"""Load and validate a frozen bundle; no generation or training during a run."""
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from harness.model import LAYERS, predict
from harness.params import ROOT, SENSITIVE_FIELDS
from harness.utils import read_json, sha256, artifact_sha256, read_artifact_json

REQUIRED = {"data.json", "weights.json", "reference.json"}


def load_bundle(path):
    path = Path(path).resolve()
    manifest = read_json(path / "manifest.json")
    if manifest.get("format_version") != 1 or not REQUIRED.issubset(manifest.get("sha256", {})):
        raise ValueError("Invalid artifact manifest")
    for name, expected in manifest["sha256"].items():
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("Artifact names must be simple filenames")
        if artifact_sha256(path / name) != expected:
            raise ValueError("Artifact checksum mismatch: " + name)
    checkpoint = path / "scam_list_gcn.pt"
    if checkpoint.exists() and sha256(checkpoint) != manifest.get("original_model_sha256"):
        raise ValueError("Original model checkpoint checksum mismatch")
    data, weights, reference = [read_artifact_json(path / name) for name in ("data.json", "weights.json", "reference.json")]
    x = np.asarray(data["features"], dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Features must be a finite N x F matrix")
    n, f = x.shape
    labels = np.asarray(data["labels"])
    if labels.shape != (n,) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected N binary labels")
    names, sensitive = data["feature_names"], data["sensitive_indices"]
    if len(names) != f or len(set(names)) != f or [names[i] for i in sensitive] != SENSITIVE_FIELDS:
        raise ValueError("Incorrect feature schema / sensitive columns")
    splits = data["splits"]
    combined = [i for split in ("train", "val", "test") for i in splits[split]]
    if sorted(combined) != list(range(n)) or not splits["test"]:
        raise ValueError("Splits must form a disjoint partition and include test nodes")
    graph = data["adjacency"]
    a = sp.coo_matrix((graph["values"], (graph["rows"], graph["cols"])), shape=(n, n)).tocsr()
    if not np.isfinite(a.data).all() or (a.data < 0).any():
        raise ValueError("Invalid normalized adjacency")
    previous = f
    for name in LAYERS:
        w, b = np.asarray(weights[name + ".weight"]), np.asarray(weights[name + ".bias"])
        if w.ndim != 2 or w.shape[0] != previous or b.shape != (w.shape[1],) or not np.isfinite(w).all() or not np.isfinite(b).all():
            raise ValueError("Invalid frozen layer: " + name)
        previous = w.shape[1]
    if previous != f or manifest["activation"] not in ("poly2", "relu"):
        raise ValueError("Model output/activation does not match workload")
    for value in (reference["threshold"], manifest["atol"], manifest["rtol"]):
        if not np.isfinite(value):
            raise ValueError("Threshold and tolerances must be finite")
    if manifest["atol"] < 0 or manifest["rtol"] < 0:
        raise ValueError("Negative tolerance")
    expected = predict(x, a, weights, sensitive, manifest["activation"])
    stored = np.asarray(reference["scores"])
    if stored.shape != (n,) or not np.isfinite(stored).all() or not np.allclose(expected, stored, atol=1e-8, rtol=1e-7):
        raise ValueError("Frozen plaintext scores do not match model/data")
    registry_path = ROOT / "artifacts" / "registry.json"
    registry = read_json(registry_path) if registry_path.exists() else {}
    official = registry.get(manifest["id"]) == sha256(path / "manifest.json")
    public_indices = [i for i in range(f) if i not in sensitive]
    public = {"x_public": x[:, public_indices].tolist(), "public_indices": public_indices,
              "sensitive_indices": sensitive, "feature_count": f, "node_count": n,
              "adjacency": graph, "weights": weights, "activation": manifest["activation"]}
    return {"manifest": manifest, "manifest_sha256": sha256(path / "manifest.json"),
            "registered": official, "data": data, "reference": reference,
            "public": public, "sensitive": x[:, sensitive].tolist()}
