"""Quality-retention and Pareto-frontier reference metrics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    """Binary anomaly-detection metrics with anomaly label ``1``."""

    recall: float
    f1: float
    accuracy: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int


def binary_metrics(
    labels: Iterable[int],
    scores: Iterable[float],
    *,
    threshold: float = 0.5,
) -> BinaryMetrics:
    """Calculate Recall, F1, and Accuracy for binary anomaly scores."""

    expected = list(labels)
    observed = list(scores)
    if not expected:
        raise ValueError("labels must not be empty")
    if len(expected) != len(observed):
        raise ValueError("labels and scores must have the same length")
    if any(label not in (0, 1) for label in expected):
        raise ValueError("labels must contain only 0 and 1")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    if any(not math.isfinite(score) or score < 0 or score > 1 for score in observed):
        raise ValueError("scores must lie in [0, 1]")

    predicted = [int(score >= threshold) for score in observed]
    tp = int(sum(bool(label == 1 and pred == 1) for label, pred in zip(expected, predicted, strict=True)))
    fp = int(sum(bool(label == 0 and pred == 1) for label, pred in zip(expected, predicted, strict=True)))
    fn = int(sum(bool(label == 1 and pred == 0) for label, pred in zip(expected, predicted, strict=True)))
    tn = int(sum(bool(label == 0 and pred == 0) for label, pred in zip(expected, predicted, strict=True)))
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return BinaryMetrics(
        recall=float(recall),
        f1=float(f1),
        accuracy=float((tp + tn) / len(expected)),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
    )


@dataclass(frozen=True, slots=True)
class MetricPoint:
    """One measured identifier-retention configuration."""

    retention: float
    recall_encrypted: float
    f1_encrypted: float
    recall_plaintext_full: float
    f1_plaintext_full: float
    latency_seconds: float
    storage_bytes: int
    communication_bytes: int

    @property
    def quality(self) -> float:
        return quality_retention(
            recall_encrypted=self.recall_encrypted,
            f1_encrypted=self.f1_encrypted,
            recall_plaintext_full=self.recall_plaintext_full,
            f1_plaintext_full=self.f1_plaintext_full,
        )


def quality_retention(
    *,
    recall_encrypted: float,
    f1_encrypted: float,
    recall_plaintext_full: float,
    f1_plaintext_full: float,
) -> float:
    """Compute Q(k), the minimum retained recall/F1 ratio."""

    values = (
        recall_encrypted,
        f1_encrypted,
        recall_plaintext_full,
        f1_plaintext_full,
    )
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in values):
        raise ValueError("quality metrics must lie in [0, 1]")
    if recall_plaintext_full == 0 or f1_plaintext_full == 0:
        raise ValueError("full-plaintext recall and F1 must be positive")
    return min(
        recall_encrypted / recall_plaintext_full,
        f1_encrypted / f1_plaintext_full,
    )


def _dominates(left: MetricPoint, right: MetricPoint) -> bool:
    no_worse = (
        left.quality >= right.quality
        and left.retention <= right.retention
        and left.latency_seconds <= right.latency_seconds
        and left.storage_bytes <= right.storage_bytes
        and left.communication_bytes <= right.communication_bytes
    )
    strictly_better = (
        left.quality > right.quality
        or left.retention < right.retention
        or left.latency_seconds < right.latency_seconds
        or left.storage_bytes < right.storage_bytes
        or left.communication_bytes < right.communication_bytes
    )
    return no_worse and strictly_better


def pareto_frontier(points: list[MetricPoint]) -> list[MetricPoint]:
    """Return non-dominated configurations sorted by identifier retention."""

    for point in points:
        if not 0 < point.retention <= 1:
            raise ValueError("retention must be in the interval (0, 1]")
        if point.latency_seconds < 0:
            raise ValueError("latency must be non-negative")
        if point.storage_bytes < 0 or point.communication_bytes < 0:
            raise ValueError("byte counts must be non-negative")

    frontier = [
        candidate
        for candidate in points
        if not any(
            other is not candidate and _dominates(other, candidate)
            for other in points
        )
    ]
    return sorted(frontier, key=lambda point: (point.retention, -point.quality))
