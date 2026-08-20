"""Quality-retention and Pareto-frontier reference metrics."""

from __future__ import annotations

from dataclasses import dataclass


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
    if any(value < 0 or value > 1 for value in values):
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
