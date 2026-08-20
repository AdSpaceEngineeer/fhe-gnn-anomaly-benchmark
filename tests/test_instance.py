import tempfile
import unittest
from pathlib import Path

import numpy as np

from fhe_gnn_anomaly_benchmark.instance import prepare_inference_instance


class InstanceTests(unittest.TestCase):
    def test_instance_contains_complete_incoming_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            output = root / "instance"
            dataset.mkdir()
            features = np.arange(16, dtype=float).reshape(8, 2)
            edges = np.array(
                [[0, 4], [1, 4], [2, 5], [3, 5], [4, 0], [5, 2], [6, 7], [7, 6]]
            )
            labels = np.array([0, 0, 0, 0, 1, 1, 0, 0])
            np.savez_compressed(dataset / "public_graph.npz", features=features, edges=edges)
            np.savez_compressed(
                dataset / "harness_ground_truth.npz",
                labels=labels,
                train_indices=np.array([0, 4]),
                validation_indices=np.array([1, 5]),
                test_indices=np.array([2, 3, 4, 6]),
            )
            (dataset / "client_identifiers.txt").write_text(
                "\n".join(f"GB00TEST000000000000{i:02d}" for i in range(8)) + "\n",
                encoding="utf-8",
            )
            metadata = prepare_inference_instance(dataset, output, batch_size=1, seed=3)
            self.assertEqual(metadata.batch_size, 1)
            with np.load(output / "public_graph.npz") as instance:
                originals = set(instance["original_node_indices"].tolist())
                self.assertEqual(originals, {0, 1, 4})
                self.assertEqual(instance["target_indices"].shape, (1,))
            np.testing.assert_array_equal(np.load(output / "harness_labels.npy"), np.array([1]))


if __name__ == "__main__":
    unittest.main()
