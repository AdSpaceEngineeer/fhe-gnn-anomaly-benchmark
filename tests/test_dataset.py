import tempfile
import unittest
from pathlib import Path

import numpy as np

from fhe_gnn_anomaly_benchmark.dataset import (
    GraphDataset,
    load_yelpchi,
    relational_group_ids,
    stratified_split,
)


class DatasetTests(unittest.TestCase):
    def test_graph_validation(self) -> None:
        graph = GraphDataset(
            features=np.ones((4, 2)),
            labels=np.array([0, 0, 1, 1]),
            edges=np.array([[0, 1], [1, 0], [2, 3], [3, 2]]),
        )
        graph.validate()

    def test_relational_groups_are_label_free_and_deterministic(self) -> None:
        edges = np.array([[0, 2], [2, 0], [1, 2], [2, 1], [3, 4]])
        groups = relational_group_ids(5, edges)
        np.testing.assert_array_equal(groups, np.array([0, 1, 0, 3, 3]))

    def test_stratified_split_is_disjoint_and_complete(self) -> None:
        labels = np.array([0] * 10 + [1] * 10)
        first = stratified_split(labels, seed=7)
        second = stratified_split(labels, seed=7)
        np.testing.assert_array_equal(first.train, second.train)
        combined = np.concatenate((first.train, first.validation, first.test))
        self.assertEqual(set(combined), set(range(20)))
        self.assertEqual(len(combined), len(set(combined)))
        for values in (first.train, first.validation, first.test):
            self.assertEqual(set(labels[values]), {0, 1})

    def test_missing_source_is_reported_before_scipy_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "YelpChi.mat"
            try:
                load_yelpchi(missing)
            except RuntimeError as error:
                self.assertIn("optional 'yelpchi' dependencies", str(error))
            except FileNotFoundError:
                pass
            else:
                self.fail("missing YelpChi source should fail")


if __name__ == "__main__":
    unittest.main()
