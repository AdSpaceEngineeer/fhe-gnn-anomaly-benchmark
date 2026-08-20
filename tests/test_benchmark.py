import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from fhe_gnn_anomaly_benchmark.baseline import (
    evaluate_plaintext_baseline,
    train_plaintext_baseline,
)
from fhe_gnn_anomaly_benchmark.benchmark import run_benchmark
from fhe_gnn_anomaly_benchmark.identifiers import generate_identifiers
from fhe_gnn_anomaly_benchmark.validation import (
    validate_comparison_files,
    validate_result_semantics,
)


class BenchmarkTests(unittest.TestCase):
    def test_plaintext_dry_run_exercises_complete_protocol(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        manifest = repository / "submissions" / "plaintext_reference" / "submission.json"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            features = np.array([[index / 12, (index % 3) / 3] for index in range(12)])
            labels = np.array([0, 1] * 6, dtype=np.int8)
            edges = np.array(
                [[source, target] for source in range(12) for target in range(12) if source != target and source // 2 == target // 2],
                dtype=np.int64,
            )
            np.savez_compressed(dataset / "public_graph.npz", features=features, edges=edges)
            np.savez_compressed(
                dataset / "harness_ground_truth.npz",
                labels=labels,
                train_indices=np.array([0, 1, 2, 3]),
                validation_indices=np.array([4, 5, 6, 7]),
                test_indices=np.array([8, 9, 10, 11]),
            )
            identifiers = generate_identifiers(range(12), [index // 2 for index in range(12)], seed=5)
            (dataset / "client_identifiers.txt").write_text(
                "\n".join(identifiers) + "\n", encoding="utf-8"
            )
            (dataset / "dataset.json").write_text(
                json.dumps({"dataset": "synthetic-test", "source_sha256": "test"}),
                encoding="utf-8",
            )
            model_path = root / "model.npz"
            train_plaintext_baseline(dataset, model_path, hidden_features=4, epochs=10, seed=9)
            self.assertTrue(model_path.with_suffix(".training.json").is_file())
            baseline_path = root / "baseline.json"
            baseline = evaluate_plaintext_baseline(
                dataset, model_path, baseline_path, retentions=(0.5, 1.0)
            )
            self.assertTrue(baseline_path.is_file())
            self.assertEqual(baseline["evaluation"]["test_nodes"], 4)
            self.assertEqual(len(baseline["evaluation"]["truncation_sweep"]), 2)
            inherited = os.environ.get("PYTHONPATH", "")
            source_path = str(repository / "src")
            python_path = source_path if not inherited else source_path + os.pathsep + inherited
            with patch.dict(os.environ, {"PYTHONPATH": python_path}):
                paths = run_benchmark(
                    manifest,
                    dataset,
                    model_path,
                    root / "results",
                    retention=0.5,
                    batch_size=4,
                    num_runs=1,
                    seed=13,
                )
            self.assertEqual(len(paths), 1)
            result = json.loads(paths[0].read_text(encoding="utf-8"))
            validate_result_semantics(result)
            self.assertFalse(result["conformance"]["is_fhe"])
            self.assertFalse(result["conformance"]["eligible_for_fhe_comparison"])
            self.assertEqual(
                result["model"]["adapter"], "polynomial_message_passing_v1"
            )
            self.assertFalse(result["model"]["recommended_baseline"])
            self.assertEqual(len(result["model"]["sha256"]), 64)
            self.assertEqual(result["run"]["retained_characters"], 11)
            self.assertEqual(result["run"]["batch_size"], 4)
            self.assertAlmostEqual(
                result["quality"]["protected"]["f1"],
                result["quality"]["plaintext_truncated"]["f1"],
            )
            self.assertGreater(result["performance"]["online_latency_seconds"], 0)
            self.assertIn("server_encrypted_compute", result["performance"]["stage_seconds"])

            matching = root / "matching-result.json"
            matching.write_text(json.dumps(result), encoding="utf-8")
            validate_comparison_files([paths[0], matching])
            result["model"]["sha256"] = "0" * 64
            mismatched = root / "mismatched-result.json"
            mismatched.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model.sha256 differs"):
                validate_comparison_files([paths[0], mismatched])


if __name__ == "__main__":
    unittest.main()
