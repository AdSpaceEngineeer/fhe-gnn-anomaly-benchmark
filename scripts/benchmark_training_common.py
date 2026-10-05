"""Shared training, verification and archive helpers."""
import hashlib
import json
import platform
import tarfile
from pathlib import Path


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def split_nodes(labels, seed):
    import numpy as np
    from sklearn.model_selection import train_test_split
    train_val, test = train_test_split(np.arange(len(labels)), test_size=0.20,
                                       random_state=seed, stratify=labels)
    train, val = train_test_split(train_val, test_size=0.20,
                                random_state=seed + 1, stratify=labels[train_val])
    return dict(train=train, val=val, test=test)


def metrics(labels, scores, threshold):
    import numpy as np
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, average_precision_score)
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite scores.')
    predictions = scores >= threshold
    return dict(threshold=float(threshold),
                accuracy=float(accuracy_score(labels, predictions)),
                precision=float(precision_score(labels, predictions, zero_division=0)),
                recall=float(recall_score(labels, predictions, zero_division=0)),
                f1=float(f1_score(labels, predictions, zero_division=0)),
                roc_auc=float(roc_auc_score(labels, scores)),
                average_precision=float(average_precision_score(labels, scores)))


def choose_threshold(labels, scores):
    import numpy as np
    from sklearn.metrics import f1_score
    candidates = np.unique(np.quantile(scores, np.linspace(0.50, 0.995, 200)))
    # Identical quantile search/tie rule to the GCN reference trainer.
    return float(max(candidates, key=lambda t: f1_score(labels, scores >= t, zero_division=0)))


def environment():
    import numpy, pandas, scipy, sklearn, torch
    return dict(python=platform.python_version(), numpy=numpy.__version__,
                pandas=pandas.__version__, scipy=scipy.__version__,
                scikit_learn=sklearn.__version__, torch=str(torch.__version__),
                device='cpu',
                torch_threads=torch.get_num_threads(),
                torch_interop_threads=torch.get_num_interop_threads())


def package(outdir, filenames, metadata):
    outdir = Path(outdir)
    manifest = dict(metadata, files={name: sha256(outdir / name) for name in filenames})
    dump(outdir / 'upload_manifest.json', manifest)
    archive = Path(str(outdir) + '.tar.gz')
    with tarfile.open(archive, 'x:gz') as stream:
        for name in filenames + ['upload_manifest.json']:
            stream.add(outdir / name, arcname=outdir.name + '/' + name, recursive=False)
    checksum = sha256(archive)
    with Path(str(archive) + '.sha256').open('x') as stream:
        stream.write(checksum + '  ' + archive.name + '\n')
    print('UPLOAD:', archive.resolve(), flush=True)
    print('SHA256:', checksum, flush=True)
