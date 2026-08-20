"""YelpChi loading and deterministic benchmark-data preparation."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .identifiers import generate_identifiers


@dataclass(frozen=True, slots=True)
class GraphDataset:
    """Public graph data loaded from YelpChi."""

    features: np.ndarray
    labels: np.ndarray
    edges: np.ndarray

    def validate(self) -> None:
        if self.features.ndim != 2:
            raise ValueError("features must be a two-dimensional matrix")
        if self.labels.ndim != 1:
            raise ValueError("labels must be a one-dimensional vector")
        if len(self.features) != len(self.labels):
            raise ValueError("features and labels must contain the same number of nodes")
        if self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise ValueError("edges must have shape (edge_count, 2)")
        if len(self.edges) and (self.edges.min() < 0 or self.edges.max() >= len(self.labels)):
            raise ValueError("edge endpoint lies outside the node range")
        if not np.isin(self.labels, [0, 1]).all():
            raise ValueError("YelpChi labels must be binary")


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def _dense_array(value: object, *, dtype: np.dtype) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()  # type: ignore[union-attr]
    return np.asarray(value, dtype=dtype)


def load_yelpchi(path: str | Path) -> GraphDataset:
    """Load canonical CARE-GNN ``YelpChi.mat`` or ``YelpChi.zip``.

    SciPy is an optional dependency because it is only needed to ingest the
    source MATLAB file; prepared benchmark artifacts use NumPy and text files.
    The loader uses ``features``, ``label``, and the homogeneous adjacency
    matrix ``homo`` documented by the CARE-GNN reference implementation.
    """

    try:
        from scipy.io import loadmat
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "YelpChi preparation requires the optional 'yelpchi' dependencies"
        ) from error

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source, "r") as archive:
            matches = [name for name in archive.namelist() if Path(name).name == "YelpChi.mat"]
            if len(matches) != 1:
                raise ValueError(
                    "YelpChi.zip must contain exactly one file named YelpChi.mat"
                )
            with archive.open(matches[0]) as matlab_file:
                material = loadmat(io.BytesIO(matlab_file.read()))
    else:
        material = loadmat(source)
    required = {"features", "label", "homo"}
    missing = sorted(required - material.keys())
    if missing:
        raise ValueError(f"YelpChi file is missing required fields: {missing}")

    features = _dense_array(material["features"], dtype=np.float64)
    labels = _dense_array(material["label"], dtype=np.int8).reshape(-1)
    adjacency = material["homo"]
    if hasattr(adjacency, "tocoo"):
        coordinate = adjacency.tocoo()
        edges = np.column_stack((coordinate.row, coordinate.col)).astype(np.int64)
    else:
        rows, columns = np.nonzero(np.asarray(adjacency))
        edges = np.column_stack((rows, columns)).astype(np.int64)
    edges = edges[edges[:, 0] != edges[:, 1]]

    dataset = GraphDataset(features=features, labels=labels, edges=edges)
    dataset.validate()
    return dataset


def relational_group_ids(node_count: int, edges: np.ndarray) -> np.ndarray:
    """Assign a label-free group from each node's closed one-hop neighbourhood.

    The group is the smallest node index in the closed neighbourhood. It is
    derived solely from public topology and never from anomaly labels.
    """

    if node_count < 1:
        raise ValueError("node_count must be positive")
    normalized = np.asarray(edges, dtype=np.int64)
    if normalized.ndim != 2 or normalized.shape[1] != 2:
        raise ValueError("edges must have shape (edge_count, 2)")
    if len(normalized) and (normalized.min() < 0 or normalized.max() >= node_count):
        raise ValueError("edge endpoint lies outside the node range")
    groups = np.arange(node_count, dtype=np.int64)
    for source, target in normalized:
        groups[source] = min(groups[source], target)
        groups[target] = min(groups[target], source)
    return groups


def stratified_split(
    labels: np.ndarray,
    *,
    seed: int,
    train_fraction: float = 0.4,
    validation_fraction: float = 0.2,
) -> DatasetSplit:
    """Create reproducible 40/20/40 splits within each binary class."""

    normalized = np.asarray(labels, dtype=np.int8).reshape(-1)
    if not len(normalized):
        raise ValueError("labels must not be empty")
    if not np.isin(normalized, [0, 1]).all():
        raise ValueError("labels must contain only 0 and 1")
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("split fractions must lie in [0, 1]")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test split")

    generator = np.random.default_rng(seed)
    partitions: dict[str, list[np.ndarray]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for label in (0, 1):
        indices = np.flatnonzero(normalized == label)
        if len(indices) < 3:
            raise ValueError("each class needs at least three nodes for stratification")
        generator.shuffle(indices)
        train_end = max(1, int(len(indices) * train_fraction))
        validation_end = max(train_end + 1, train_end + int(len(indices) * validation_fraction))
        validation_end = min(validation_end, len(indices) - 1)
        partitions["train"].append(indices[:train_end])
        partitions["validation"].append(indices[train_end:validation_end])
        partitions["test"].append(indices[validation_end:])

    def combine(name: str) -> np.ndarray:
        values = np.concatenate(partitions[name]).astype(np.int64)
        generator.shuffle(values)
        return values

    return DatasetSplit(
        train=combine("train"),
        validation=combine("validation"),
        test=combine("test"),
    )


def prepare_yelpchi(
    source_path: str | Path,
    output_directory: str | Path,
    *,
    seed: int = 2026,
) -> dict[str, object]:
    """Create separated public-server and private-client benchmark artifacts."""

    source = Path(source_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_yelpchi(source)
    splits = stratified_split(dataset.labels, seed=seed)
    groups = relational_group_ids(len(dataset.labels), dataset.edges)
    identifiers = generate_identifiers(range(len(groups)), groups, seed=seed)

    public_path = output / "public_graph.npz"
    harness_path = output / "harness_ground_truth.npz"
    client_path = output / "client_identifiers.txt"
    np.savez_compressed(
        public_path,
        features=dataset.features,
        edges=dataset.edges,
    )
    np.savez_compressed(
        harness_path,
        labels=dataset.labels,
        train_indices=splits.train,
        validation_indices=splits.validation,
        test_indices=splits.test,
    )
    client_path.write_text("\n".join(identifiers) + "\n", encoding="utf-8")
    metadata: dict[str, object] = {
        "dataset": "YelpChi",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "seed": seed,
        "node_count": int(len(dataset.labels)),
        "edge_count": int(len(dataset.edges)),
        "feature_count": int(dataset.features.shape[1]),
        "anomaly_count": int(dataset.labels.sum()),
        "identifier_format": "synthetic GB IBAN-like, ISO 13616 mod-97 valid",
        "identifier_length": len(identifiers[0]),
        "group_rule": "minimum node index in closed one-hop public neighbourhood",
        "split_rule": "stratified 40/20/40 with deterministic seed",
        "public_artifact": public_path.name,
        "harness_artifact": harness_path.name,
        "client_artifact": client_path.name,
    }
    (output / "dataset.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata
