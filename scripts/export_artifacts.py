"""Maintainer utility: export a verified training archive without retraining."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import scipy.sparse as sp
from harness.params import WORKLOADS
from harness.model import predict
from harness.metrics import quality
from harness.utils import sha256, write_json


def coo(matrix):
    matrix = matrix.tocoo()
    return dict(rows=matrix.row.tolist(), cols=matrix.col.tolist(), values=matrix.data.tolist())


def export(archive, checksum, destination, workload):
    archive, destination = Path(archive), Path(destination)
    if sha256(archive).lower() != checksum.lower():
        raise ValueError('Training archive checksum mismatch')
    if destination.exists():
        raise ValueError('Destination already exists')
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers()
        if any(not m.isfile() and not m.isdir() for m in members):
            raise ValueError('Archive may only contain ordinary files/directories')
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive entries')
        manifests = [name for name in names if name.endswith('/upload_manifest.json')]
        if len(manifests) != 1:
            raise ValueError('Expected one upload manifest')
        prefix = manifests[0].rsplit('/', 1)[0] + '/'
        def raw(name):
            return stream.extractfile(prefix + name).read()
        upload = json.loads(raw('upload_manifest.json'))
        for name, digest in upload['files'].items():
            if Path(name).name != name or '/' in name or '\\' in name or name in ('', '.', '..'):
                raise ValueError('Invalid upload filename')
            if hashlib.sha256(raw(name)).hexdigest() != digest:
                raise ValueError('Upload manifest checksum mismatch: ' + name)
        def checked(name):
            if name not in upload['files']:
                raise ValueError('Source file is not covered by upload manifest: ' + name)
            return raw(name)
        x = np.load(io.BytesIO(checked('features.npy')), allow_pickle=False).astype(np.float64)
        labels = np.load(io.BytesIO(checked('labels.npy')), allow_pickle=False)
        schema = json.loads(checked('feature_schema.json'))
        splits = json.loads(checked('splits.json'))
        weights = json.loads(checked('weights.json'))
        graph = coo(sp.load_npz(io.BytesIO(checked('network_adjacency_normalized.npz'))))
        training_report = json.loads(checked('baseline_metrics.json'))
        activation = WORKLOADS[workload]['activation']
        if upload['activation'] != activation:
            raise ValueError('Wrong source activation')
        data = dict(features=x.tolist(), labels=labels.tolist(), feature_names=schema['feature_names'],
                    sensitive_indices=schema['sensitive_feature_indices'], splits=splits,
                    adjacency=graph, preprocessing=schema)
        public = dict(workload=workload, adjacency=graph, weights=weights,
                      sensitive_indices=data['sensitive_indices'], activation=activation)
        if workload == 'tam':
            data['scoring_adjacency'] = coo(sp.load_npz(io.BytesIO(checked('scoring_adjacency.npz'))))
            public.update(scoring_adjacency=data['scoring_adjacency'], norm_epsilon=1e-12)
        scores = predict(x, public)
        # Compare with the saved training run before exporting a float64 reference.
        import csv
        original_scores = np.array([float(row['score']) for row in csv.DictReader(io.StringIO(checked('anomaly_scores.csv').decode()))])
        np.testing.assert_allclose(scores, original_scores, atol=1e-4, rtol=1e-5)
        threshold = training_report['validation']['threshold']
        reference = dict(scores=scores.tolist(), threshold=threshold,
                         validation=quality(labels[splits['val']], scores[splits['val']], threshold),
                         test=quality(labels[splits['test']], scores[splits['test']], threshold),
                         note='Float64 inference from frozen float32 weights; no retraining or threshold tuning')
        destination.mkdir(parents=True)
        hashes = {}
        def save(name, content, compress=False):
            hashes[name] = hashlib.sha256(content).hexdigest()
            target = destination / (name + '.gz' if compress else name)
            target.write_bytes(gzip.compress(content, mtime=0) if compress else content)
        def json_bytes(value):
            return (json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n').encode()
        for name, value, compress in [('data.json', data, True), ('weights.json', weights, False),
                                       ('reference.json', reference, True)]:
            save(name, json_bytes(value), compress)
        # Preserve numerical provenance without local runtime policy or host details.
        save('baseline_metrics.json', json_bytes(training_report))
        save('reload_verification.json', checked('reload_verification.json'))
        raw_logs = ['transactions.csv'] if workload == 'gcn' else ['accounts.csv', 'transfer_edges.csv']
        for name in raw_logs:
            save(name, checked(name), True)
        checkpoint = 'scam_list_' + workload + '.pt'
        save(checkpoint, checked(checkpoint))
        env = json.loads(checked('environment.json'))
        save('software_versions.json', json_bytes({k: env[k] for k in
             ('python', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'scikit_learn', 'torch') if k in env}))
        manifest = dict(format_version=2, id=WORKLOADS[workload]['artifact_id'], workload=workload,
                        purpose='frozen_inference_benchmark', activation=activation,
                        atol=0.001, rtol=0.001, source_archive_sha256=checksum.lower(),
                        original_model_sha256=hashlib.sha256(checked(checkpoint)).hexdigest(),
                        source_reference_max_absolute_error=float(np.max(np.abs(scores - original_scores))),
                        sha256=hashes)
        if workload == 'tam':
            manifest['norm_epsilon'] = 1e-12
        write_json(destination / 'manifest.json', manifest)
        print(json.dumps(dict(workload=workload, manifest_sha256=sha256(destination / 'manifest.json'),
                              test=reference['test'], source_error=manifest['source_reference_max_absolute_error']), indent=2))
        return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--workload', required=True, choices=sorted(WORKLOADS))
    a = p.parse_args()
    export(a.archive, a.sha256, a.out, a.workload)
