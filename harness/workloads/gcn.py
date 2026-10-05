"""Four-layer ReLU attribute autoencoder."""
import numpy as np

LAYERS = ('encoder_1', 'encoder_2', 'decoder_1', 'decoder_2')


def validate(weights, feature_count):
    previous = feature_count
    for name in LAYERS:
        w, b = np.asarray(weights[name + '.weight']), np.asarray(weights[name + '.bias'])
        if w.ndim != 2 or w.shape[0] != previous or b.shape != (w.shape[1],):
            raise ValueError('Invalid GCN layer: ' + name)
        if not np.isfinite(w).all() or not np.isfinite(b).all():
            raise ValueError('Non-finite GCN weights')
        previous = w.shape[1]
    if previous != feature_count:
        raise ValueError('GCN must reconstruct every input feature')


def predict(features, adjacency, public):
    h = np.asarray(features, dtype=np.float64)
    weights = public['weights']
    for index, name in enumerate(LAYERS):
        h = adjacency @ (h @ np.asarray(weights[name + '.weight'])) + np.asarray(weights[name + '.bias'])
        if index < 3:
            h = np.maximum(h, 0.0)
    sensitive = public['sensitive_indices']
    return np.mean((h[:, sensitive] - features[:, sensitive]) ** 2, axis=1)
