"""Two-layer PReLU encoder and neighbor cosine-affinity scoring."""
import numpy as np


def validate(weights, feature_count):
    previous = feature_count
    for layer in (1, 2):
        w = np.asarray(weights[f'gcn{layer}.weight'])
        b = np.asarray(weights[f'gcn{layer}.bias'])
        a = np.asarray(weights[f'prelu{layer}.weight'])
        if w.ndim != 2 or w.shape[0] != previous or b.shape != (w.shape[1],) or a.shape != (1,):
            raise ValueError('Invalid TAM layer')
        if any(not np.isfinite(value).all() for value in (w, b, a)):
            raise ValueError('Non-finite TAM weights')
        previous = w.shape[1]


def predict(features, adjacency, scoring, public):
    h = np.asarray(features, dtype=np.float64)
    weights = public['weights']
    for layer in (1, 2):
        h = adjacency @ (h @ np.asarray(weights[f'gcn{layer}.weight'])) + np.asarray(weights[f'gcn{layer}.bias'])
        h = np.where(h >= 0, h, float(weights[f'prelu{layer}.weight'][0]) * h)
    # Standard normalization guard, part of the frozen model definition.
    h = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), public['norm_epsilon'])
    degree = np.asarray(scoring.sum(axis=1)).ravel()
    if (degree <= 0).any():
        raise ValueError('Scoring graph requires positive degree')
    return 1.0 - np.sum(h * (scoring @ h), axis=1) / degree
