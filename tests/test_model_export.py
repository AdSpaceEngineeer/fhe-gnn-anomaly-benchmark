"""Maintainer checks; skipped when training-only dependencies are absent."""
from pathlib import Path
import importlib.util
import numpy as np
import pytest
import scipy.sparse as sp
from harness.params import ROOT
from harness.model import predict
from harness.generate_input import load_bundle
from harness.utils import write_json, sha256


def test_torch_and_numpy_model_agree_and_export(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pandas")
    spec = importlib.util.spec_from_file_location("training", ROOT / "scripts/scam_list_gcn.py")
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    torch.set_num_threads(1)
    b = load_bundle(ROOT / "tests/fixtures/arithmetic")
    data = b["data"]
    x = np.asarray(data["features"], dtype=np.float32)
    g = data["adjacency"]
    a = sp.coo_matrix((g["values"], (g["rows"], g["cols"])), shape=(3, 3), dtype=np.float32)
    model = training.Scam_List_GCN(8, 2, 2)
    weights = b["public"]["weights"]
    model.load_state_dict({key: torch.tensor(value, dtype=torch.float32) for key, value in weights.items()})
    with torch.no_grad():
        actual = training.reconstruction_scores(torch.tensor(x),
                  model(torch.tensor(x), training.scipy_to_torch_sparse(a, torch.device("cpu"))), [4, 6, 7])
    assert np.allclose(actual, b["reference"]["scores"], atol=1e-6)
    source = tmp_path / "run"
    source.mkdir()
    np.save(source / "features.npy", x)
    np.save(source / "labels.npy", data["labels"])
    sp.save_npz(source / "network_adjacency_normalized.npz", a)
    write_json(source / "splits.json", {"train": [0], "val": [1], "test": [2]})
    write_json(source / "feature_schema.json", {"feature_names": data["feature_names"], "sensitive_feature_indices": [4,6,7]})
    torch.save({"model_state_dict": model.state_dict(), "config": {"activation": "poly2"}}, source / "scam_list_gcn.pt")
    write_json(source / "baseline_metrics.json", {"model_sha256": sha256(source / "scam_list_gcn.pt"),
                                                  "validation": {"threshold": 1.}})
    (source / "transactions.csv").write_text("event_id\n1\n2\n3\n")
    from scripts.export_artifacts import export
    target = tmp_path / "export"
    export(source, target, "test-only")
    restored = load_bundle(target)
    assert not restored["registered"]
    assert restored["reference"]["threshold"] == 1.
    assert np.allclose(restored["reference"]["scores"], actual, atol=1e-6)
