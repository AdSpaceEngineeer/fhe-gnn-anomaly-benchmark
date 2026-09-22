"""Frozen four-layer Scam_List_GCN inference; no training dependency."""
import numpy as np

LAYERS = ("encoder_1", "encoder_2", "decoder_1", "decoder_2")


def predict(features, adjacency, weights, sensitive_indices, activation="poly2"):
    h = np.asarray(features, dtype=np.float64)
    for index, name in enumerate(LAYERS):
        h = adjacency @ (h @ np.asarray(weights[name + ".weight"]))
        h = h + np.asarray(weights[name + ".bias"])
        if index < 3:
            if activation == "poly2":
                h = h + 0.125 * h * h
            elif activation == "relu":
                h = np.maximum(h, 0)
            else:
                raise ValueError("Unsupported activation: " + activation)
    error = h[:, sensitive_indices] - features[:, sensitive_indices]
    return np.mean(error * error, axis=1)
