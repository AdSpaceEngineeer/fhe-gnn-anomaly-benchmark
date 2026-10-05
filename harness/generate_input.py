"""Load a frozen workload and verify its hashes and reference predictions."""
from pathlib import Path
import numpy as np
from harness.model import predict
from harness.workloads import gcn, tam
from harness.params import ROOT, WORKLOADS
from harness.utils import read_json, sha256, artifact_sha256, read_artifact_json


def load_bundle(path, workload):
    policy = WORKLOADS[workload]
    path = Path(path).resolve()
    manifest = read_json(path / 'manifest.json')
    if manifest.get('format_version') != 2 or manifest.get('workload') != workload:
        raise ValueError('Artifact format/workload mismatch')
    required = {'data.json', 'weights.json', 'reference.json'}
    if not required.issubset(manifest.get('sha256', {})):
        raise ValueError('Incomplete artifact manifest')
    for name, expected in manifest['sha256'].items():
        if name in ('', '.', '..') or Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError('Artifact names must be simple filenames')
        if artifact_sha256(path / name) != expected:
            raise ValueError('Artifact checksum mismatch: ' + name)
    data, weights, reference = [read_artifact_json(path / name) for name in ('data.json', 'weights.json', 'reference.json')]
    x = np.asarray(data['features'], dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != policy['features'] or not np.isfinite(x).all():
        raise ValueError('Invalid feature matrix')
    n, f = x.shape
    labels = np.asarray(data['labels'])
    if labels.shape != (n,) or not np.isin(labels, [0, 1]).all():
        raise ValueError('Expected N binary labels')
    names, sensitive = data['feature_names'], data['sensitive_indices']
    if (len(names) != f or len(set(names)) != f or len(sensitive) != 3 or
        any(type(i) is not int or not 0 <= i < f for i in sensitive) or
        len(set(sensitive)) != 3 or [names[i] for i in sensitive] != policy['sensitive']):
        raise ValueError('Incorrect sensitive feature schema')
    splits = data['splits']
    combined = [i for split in ('train', 'val', 'test') for i in splits[split]]
    if any(type(i) is not int for i in combined) or sorted(combined) != list(range(n)) or not splits['test']:
        raise ValueError('Splits must form a disjoint partition with test nodes')
    if manifest['activation'] != policy['activation']:
        raise ValueError('Activation does not match workload')
    (gcn if workload == 'gcn' else tam).validate(weights, f)
    for value in (reference['threshold'], manifest['atol'], manifest['rtol']):
        if isinstance(value, bool) or not np.isfinite(value):
            raise ValueError('Threshold/tolerances must be finite numbers')
    if manifest['atol'] < 0 or manifest['rtol'] < 0:
        raise ValueError('Negative tolerance')
    public_indices = [i for i in range(f) if i not in sensitive]
    public = dict(workload=workload, x_public=x[:, public_indices].tolist(), public_indices=public_indices,
                  sensitive_indices=sensitive, feature_count=f, node_count=n,
                  adjacency=data['adjacency'], weights=weights, activation=manifest['activation'])
    if workload == 'tam':
        public.update(scoring_adjacency=data['scoring_adjacency'], norm_epsilon=manifest['norm_epsilon'])
    expected = predict(x, public)
    stored = np.asarray(reference['scores'])
    if stored.shape != (n,) or not np.isfinite(stored).all() or not np.allclose(expected, stored, atol=1e-8, rtol=1e-7):
        raise ValueError('Frozen scores do not match reference inference')
    registry = read_json(ROOT / 'artifacts' / 'registry.json')
    official = registry.get(manifest['id']) == sha256(path / 'manifest.json')
    return dict(manifest=manifest, manifest_sha256=sha256(path / 'manifest.json'), registered=official,
                data=data, reference=reference, public=public, sensitive=x[:, sensitive].tolist())
