"""Build fixed one-hop node-inference instances from prepared YelpChi data."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class InstanceMetadata:
    batch_size: int
    subgraph_nodes: int
    subgraph_edges: int
    seed: int
    generation_seconds: float


def _select_targets(test_indices: np.ndarray, labels: np.ndarray, batch_size: int, seed: int) -> np.ndarray:
    if batch_size < 1 or batch_size > len(test_indices):
        raise ValueError("batch_size must be between 1 and the test-split size")
    generator = np.random.default_rng(seed)
    test = np.asarray(test_indices, dtype=np.int64)
    if batch_size == 1:
        anomalies = test[labels[test] == 1]
        if not len(anomalies):
            raise ValueError("single-anomaly instance requires an anomaly in the test split")
        return np.array([anomalies[generator.integers(0, len(anomalies))]], dtype=np.int64)

    positives = test[labels[test] == 1]
    negatives = test[labels[test] == 0]
    positive_count = max(1, round(batch_size * len(positives) / len(test)))
    positive_count = min(positive_count, len(positives), batch_size - 1)
    negative_count = batch_size - positive_count
    if negative_count > len(negatives):
        negative_count = len(negatives)
        positive_count = batch_size - negative_count
    if positive_count > len(positives):
        raise ValueError("test split cannot supply the requested stratified batch")
    chosen = np.concatenate(
        (
            generator.choice(positives, positive_count, replace=False),
            generator.choice(negatives, negative_count, replace=False),
        )
    ).astype(np.int64)
    generator.shuffle(chosen)
    return chosen


def prepare_inference_instance(
    dataset_directory: str | Path,
    output_directory: str | Path,
    *,
    batch_size: int,
    seed: int,
) -> InstanceMetadata:
    """Materialize target nodes and their complete incoming one-hop context."""

    start = time.perf_counter()
    dataset = Path(dataset_directory)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(dataset / "public_graph.npz") as public:
        features = public["features"]
        edges = public["edges"].astype(np.int64)
    with np.load(dataset / "harness_ground_truth.npz") as private:
        labels = private["labels"].astype(np.int8)
        test_indices = private["test_indices"].astype(np.int64)
    identifiers = (dataset / "client_identifiers.txt").read_text(encoding="utf-8").splitlines()
    if len(features) != len(labels) or len(identifiers) != len(labels):
        raise ValueError("prepared public, harness, and client artifacts are not aligned")

    targets = _select_targets(test_indices, labels, batch_size, seed)
    incoming_sources = edges[np.isin(edges[:, 1], targets), 0]
    selected_nodes = np.unique(np.concatenate((targets, incoming_sources))).astype(np.int64)
    local_index = np.full(len(labels), -1, dtype=np.int64)
    local_index[selected_nodes] = np.arange(len(selected_nodes))
    edge_mask = np.isin(edges[:, 0], selected_nodes) & np.isin(edges[:, 1], selected_nodes)
    local_edges = local_index[edges[edge_mask]]
    local_targets = local_index[targets]

    np.savez_compressed(
        output / "public_graph.npz",
        features=features[selected_nodes],
        edges=local_edges,
        target_indices=local_targets,
        original_node_indices=selected_nodes,
    )
    np.save(output / "harness_labels.npy", labels[targets])
    (output / "client_identifiers.txt").write_text(
        "\n".join(identifiers[index] for index in selected_nodes) + "\n",
        encoding="utf-8",
    )
    elapsed = time.perf_counter() - start
    metadata = InstanceMetadata(
        batch_size=batch_size,
        subgraph_nodes=len(selected_nodes),
        subgraph_edges=len(local_edges),
        seed=seed,
        generation_seconds=elapsed,
    )
    (output / "instance.json").write_text(
        json.dumps(
            {
                "batch_size": metadata.batch_size,
                "subgraph_nodes": metadata.subgraph_nodes,
                "subgraph_edges": metadata.subgraph_edges,
                "seed": metadata.seed,
                "generation_seconds": metadata.generation_seconds,
                "target_selection": "single anomaly or stratified deterministic test batch",
                "context": "complete incoming one-hop neighbourhood",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata
