import tempfile
import unittest
from pathlib import Path

import numpy as np

from fhe_gnn_anomaly_benchmark.identifiers import encode_identifier
from fhe_gnn_anomaly_benchmark.model import (
    IDENTIFIER_FEATURE_COUNT,
    PolynomialMessagePassingGNN,
    identifier_feature_matrix,
    mean_neighbour_features,
)


class ModelTests(unittest.TestCase):
    def test_identifier_histogram_preserves_prefix_suffix_sides(self) -> None:
        encoded = encode_identifier("GB12ABCD12345678901234", retained_characters=4)
        matrix = identifier_feature_matrix([encoded])
        self.assertEqual(matrix.shape, (1, IDENTIFIER_FEATURE_COUNT))
        self.assertAlmostEqual(float(matrix.sum()), 4 / 22)
        self.assertEqual(np.count_nonzero(matrix[:, :36]), 2)
        self.assertEqual(np.count_nonzero(matrix[:, 36:]), 2)

    def test_mean_neighbour_aggregation(self) -> None:
        features = np.array([[1.0], [3.0], [7.0]])
        edges = np.array([[0, 1], [2, 1]])
        aggregated = mean_neighbour_features(features, edges)
        np.testing.assert_array_equal(aggregated, np.array([[0.0], [4.0], [0.0]]))

    def test_model_round_trip(self) -> None:
        features = np.array(
            [[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [0.9, 1.0], [0.2, 0.1], [0.8, 0.9]]
        )
        labels = np.array([0, 0, 1, 1, 0, 1])
        edges = np.array([[0, 1], [1, 0], [2, 3], [3, 2], [4, 5], [5, 4]])
        model = PolynomialMessagePassingGNN.initialize(2, 4, seed=11)
        model.fit(features, edges, labels, np.array([0, 2, 4, 5]), np.array([1, 3]), epochs=10)
        scores = model.scores(features, edges)
        self.assertEqual(scores.shape, (6,))
        self.assertTrue(np.all((scores >= 0) & (scores <= 1)))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.npz"
            model.save(path)
            restored = PolynomialMessagePassingGNN.load(path)
            np.testing.assert_allclose(restored.logits(features, edges), model.logits(features, edges))
            self.assertEqual(restored.threshold, model.threshold)


if __name__ == "__main__":
    unittest.main()
