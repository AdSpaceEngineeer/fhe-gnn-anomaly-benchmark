import unittest

from fhe_gnn_anomaly_benchmark.metrics import (
    MetricPoint,
    pareto_frontier,
    quality_retention,
)


def _point(retention: float, quality: float, latency: float) -> MetricPoint:
    return MetricPoint(
        retention=retention,
        recall_encrypted=quality,
        f1_encrypted=quality,
        recall_plaintext_full=1.0,
        f1_plaintext_full=1.0,
        latency_seconds=latency,
        storage_bytes=int(retention * 1_000),
        communication_bytes=int(retention * 500),
    )


class MetricTests(unittest.TestCase):
    def test_quality_retention_uses_weaker_metric(self) -> None:
        self.assertAlmostEqual(
            quality_retention(
                recall_encrypted=0.9,
                f1_encrypted=0.8,
                recall_plaintext_full=1.0,
                f1_plaintext_full=1.0,
            ),
            0.8,
        )

    def test_quality_rejects_zero_baseline(self) -> None:
        with self.assertRaises(ValueError):
            quality_retention(
                recall_encrypted=0.9,
                f1_encrypted=0.8,
                recall_plaintext_full=0,
                f1_plaintext_full=1.0,
            )

    def test_frontier_removes_dominated_point(self) -> None:
        efficient = _point(0.4, 0.9, 2.0)
        dominated = MetricPoint(
            retention=0.6,
            recall_encrypted=0.85,
            f1_encrypted=0.85,
            recall_plaintext_full=1.0,
            f1_plaintext_full=1.0,
            latency_seconds=3.0,
            storage_bytes=700,
            communication_bytes=400,
        )
        high_quality = _point(1.0, 1.0, 5.0)

        self.assertEqual(
            pareto_frontier([dominated, high_quality, efficient]),
            [efficient, high_quality],
        )


if __name__ == "__main__":
    unittest.main()
