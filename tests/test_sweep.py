import tempfile
import unittest
from pathlib import Path

from fhe_gnn_anomaly_benchmark.sweep import _write_q_svg


class SweepTests(unittest.TestCase):
    def test_q_plot_contains_declared_points(self) -> None:
        points = [
            {"retained_characters": 5, "retention_realized": 5 / 22, "q_median": 0.9},
            {"retained_characters": 22, "retention_realized": 1.0, "q_median": 1.0},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "q.svg"
            _write_q_svg(points, path)
            content = path.read_text(encoding="utf-8")
            self.assertIn("Identifier Truncation Robustness", content)
            self.assertIn("k=5; Q=0.900", content)
            self.assertIn("k=22; Q=1.000", content)


if __name__ == "__main__":
    unittest.main()
