"""Training and evaluation helpers for the frozen plaintext GNN baseline."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from .identifiers import encode_identifiers, retained_character_count
from .metrics import BinaryMetrics, binary_metrics, quality_retention
from .model import PolynomialMessagePassingGNN, identifier_feature_matrix
from .model_bundle import ModelBundle


def _read_identifiers(path: Path) -> list[str]:
    identifiers = path.read_text(encoding="utf-8").splitlines()
    if not identifiers:
        raise ValueError(f"identifier file is empty: {path}")
    return identifiers


def plaintext_feature_matrix(
    public_features: np.ndarray,
    identifiers: list[str],
    *,
    retention: float,
) -> np.ndarray:
    if len(public_features) != len(identifiers):
        raise ValueError("public features and identifiers are not aligned")
    encoded = encode_identifiers(identifiers, retention=retention)
    return np.concatenate((public_features, identifier_feature_matrix(encoded)), axis=1)


def train_plaintext_baseline(
    dataset_directory: str | Path,
    model_path: str | Path,
    *,
    hidden_features: int = 16,
    epochs: int = 200,
    learning_rate: float = 0.01,
    seed: int = 2026,
) -> BinaryMetrics:
    """Train frozen weights on full identifiers; training is not benchmarked."""

    dataset = Path(dataset_directory)
    with np.load(dataset / "public_graph.npz") as public:
        public_features = public["features"]
        edges = public["edges"]
    with np.load(dataset / "harness_ground_truth.npz") as harness:
        labels = harness["labels"].astype(np.int8)
        train_indices = harness["train_indices"].astype(np.int64)
        validation_indices = harness["validation_indices"].astype(np.int64)
        test_indices = harness["test_indices"].astype(np.int64)
    identifiers = _read_identifiers(dataset / "client_identifiers.txt")
    features = plaintext_feature_matrix(public_features, identifiers, retention=1.0)
    model = PolynomialMessagePassingGNN.initialize(
        features.shape[1], hidden_features, seed=seed
    )
    model.fit(
        features,
        edges,
        labels,
        train_indices,
        validation_indices,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    model.save(model_path)
    test_metrics = binary_metrics(
        labels[test_indices],
        model.scores(features, edges)[test_indices],
        threshold=model.threshold,
    )
    dataset_metadata_path = dataset / "dataset.json"
    dataset_metadata = (
        json.loads(dataset_metadata_path.read_text(encoding="utf-8"))
        if dataset_metadata_path.is_file()
        else {}
    )
    model_file = Path(model_path)
    metadata = {
        "architecture": "one-hop mean-aggregation polynomial message-passing GNN",
        "activation": f"z + {model.activation_quadratic} * z^2",
        "dataset_source_sha256": dataset_metadata.get("source_sha256"),
        "epochs": epochs,
        "hidden_features": hidden_features,
        "learning_rate": learning_rate,
        "model_sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "seed": seed,
        "split_nodes": {
            "train": int(len(train_indices)),
            "validation": int(len(validation_indices)),
            "test": int(len(test_indices)),
        },
        "threshold_selection": "maximum F1 on the validation split; recall breaks ties",
        "frozen_threshold": model.threshold,
        "test_metrics": asdict(test_metrics),
    }
    training_path = model_file.with_suffix(".training.json")
    training_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return test_metrics


def evaluate_plaintext_baseline(
    dataset_directory: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    *,
    retentions: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
) -> dict[str, object]:
    """Evaluate full-test quality and identifier truncation without timing FHE."""

    if not retentions:
        raise ValueError("at least one retention level is required")
    dataset = Path(dataset_directory)
    model_bundle = ModelBundle.load(model_path)
    model_file = model_bundle.artifact_path
    with np.load(dataset / "public_graph.npz") as public:
        public_features = public["features"]
        edges = public["edges"]
    with np.load(dataset / "harness_ground_truth.npz") as harness:
        labels = harness["labels"].astype(np.int8)
        test_indices = harness["test_indices"].astype(np.int64)
    identifiers = _read_identifiers(dataset / "client_identifiers.txt")
    full_length = len(identifiers[0])
    if any(len(identifier) != full_length for identifier in identifiers):
        raise ValueError("all identifiers must have the same length")

    model = PolynomialMessagePassingGNN.load(model_file)
    full_features = plaintext_feature_matrix(public_features, identifiers, retention=1.0)
    full_scores = model.scores(full_features, edges)[test_indices]
    full_metrics = binary_metrics(labels[test_indices], full_scores, threshold=model.threshold)
    if full_metrics.recall == 0 or full_metrics.f1 == 0:
        raise ValueError("full-identifier Recall and F1 must be non-zero to define Q(k)")

    sweep: list[dict[str, object]] = []
    seen_counts: set[int] = set()
    for retention in retentions:
        retained = retained_character_count(full_length, retention)
        if retained in seen_counts:
            continue
        seen_counts.add(retained)
        truncated_features = plaintext_feature_matrix(
            public_features, identifiers, retention=retention
        )
        truncated_scores = model.scores(truncated_features, edges)[test_indices]
        metrics = binary_metrics(
            labels[test_indices], truncated_scores, threshold=model.threshold
        )
        sweep.append(
            {
                "retention_requested": float(retention),
                "retained_characters": retained,
                "retention_realized": retained / full_length,
                "metrics": asdict(metrics),
                "q": quality_retention(
                    recall_encrypted=metrics.recall,
                    f1_encrypted=metrics.f1,
                    recall_plaintext_full=full_metrics.recall,
                    f1_plaintext_full=full_metrics.f1,
                ),
            }
        )

    dataset_metadata = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    training_path = model_file.with_suffix(".training.json")
    training_metadata = (
        json.loads(training_path.read_text(encoding="utf-8"))
        if training_path.is_file()
        else None
    )
    report: dict[str, object] = {
        "schema_version": "0.1.0",
        "dataset": dataset_metadata,
        "model": {
            "architecture": "one-hop mean-aggregation polynomial message-passing GNN",
            "activation": f"z + {model.activation_quadratic} * z^2",
            "input_features": int(model.self_weights.shape[0]),
            "hidden_features": int(model.self_weights.shape[1]),
            "model_sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(),
            "frozen_threshold": model.threshold,
            "training": training_metadata,
        },
        "evaluation": {
            "split": "test",
            "test_nodes": int(len(test_indices)),
            "test_anomalies": int(labels[test_indices].sum()),
            "full_identifier_metrics": asdict(full_metrics),
            "truncation_sweep": sweep,
        },
        "scope_note": (
            "Plaintext quality evaluation only. FHE latency, memory, storage, "
            "communication, and key overhead are submission-specific measurements."
        ),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def score_plaintext_instance(
    instance_directory: str | Path,
    model_path: str | Path,
    *,
    retention: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return target labels, scores, and the frozen postprocessing threshold."""

    instance = Path(instance_directory)
    with np.load(instance / "public_graph.npz") as public:
        public_features = public["features"]
        edges = public["edges"]
        targets = public["target_indices"].astype(np.int64)
    labels = np.load(instance / "harness_labels.npy").astype(np.int8)
    identifiers = _read_identifiers(instance / "client_identifiers.txt")
    features = plaintext_feature_matrix(public_features, identifiers, retention=retention)
    model = PolynomialMessagePassingGNN.load(ModelBundle.load(model_path).artifact_path)
    return labels, model.scores(features, edges)[targets], model.threshold
