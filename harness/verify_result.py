"""Separate numerical agreement from anomaly-detection quality."""
import numpy as np
from harness.metrics import quality


def verify(scores, reference, labels, indices, threshold, atol, rtol):
    scores, reference = np.asarray(scores, dtype=float), np.asarray(reference, dtype=float)
    if scores.shape != reference.shape or scores.ndim != 1:
        raise ValueError("Expected one score per graph node, in the published node order")
    if not np.isfinite(scores).all():
        raise ValueError("Submission scores contain NaN or infinity")
    error = np.abs(scores - reference)
    indices = np.asarray(indices, dtype=int)
    return {"passed": bool(np.all(error <= atol + rtol * np.abs(reference))),
            "absolute_tolerance": atol, "relative_tolerance": rtol,
            "max_absolute_error": float(error.max()), "mean_absolute_error": float(error.mean()),
            "prediction_agreement": float(np.mean((scores >= threshold) == (reference >= threshold))),
            "quality": quality(np.asarray(labels)[indices], scores[indices], threshold),
            "plaintext_quality": quality(np.asarray(labels)[indices], reference[indices], threshold)}
