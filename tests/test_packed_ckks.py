"""Routing/algebra tests; fixtures do not define additional benchmark workloads."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np
import pytest
import scipy.sparse as sp
from harness.model import predict
from harness.params import ROOT
from harness.utils import load_adapter, read_json

pytest.importorskip("tenseal")
sys.path.insert(0, str(ROOT / "submissions/toy_ckks"))
import packed_ckks as packed


def test_copied_submission_describe_without_key_generation(tmp_path):
    import shutil
    import subprocess
    folder = tmp_path / "my_method"
    shutil.copytree(ROOT / "submissions/toy_ckks", folder)
    output = tmp_path / "description"
    output.mkdir()
    result = subprocess.run(
        [sys.executable, "-m", "harness.worker", "describe", "--adapter", str(folder / "adapter.py"),
         "--io", str(output), "--threads", "1"], cwd=ROOT,
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    description = read_json(output / "description.json")
    assert description["name"] == "my_method"
    assert description["security"]["classical_bits"] == 128
    assert description["parameters"]["poly_modulus_degree"] == 32768


def test_feature_projection_masks_do_not_mix_events():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(packed.ROWS, packed.STRIDE))
    for ni, no in ((8, 64), (64, 32), (32, 64), (64, 8)):
        w = rng.normal(size=(ni, no))
        actual = sum(np.roll(x.ravel(), -shift) * mask for shift, mask in packed.feature_diagonals(w))
        expected = np.zeros_like(x)
        expected[:, :no] = x[:, :ni] @ w
        assert np.allclose(actual.reshape(x.shape), expected, atol=1e-12)


def test_graph_aggregation_crosses_ciphertext_boundaries():
    rng = np.random.default_rng(43)
    n = packed.ROWS + 3
    a = sp.coo_matrix((rng.uniform(.1, 1, size=10),
                     ([0, 1, 1, 2, n - 1, n - 2, 0, n - 1, 1, 2],
                      [n - 1, n - 2, n - 2, 0, 0, 2, 1, n - 1, 1, 2])), shape=(n, n))
    x = rng.normal(size=(n, packed.STRIDE))
    padded = np.zeros((2 * packed.ROWS, packed.STRIDE))
    padded[:n] = x
    blocks = padded.reshape(2, packed.SLOTS)
    actual = np.zeros_like(blocks)
    graph = {"rows": a.row.tolist(), "cols": a.col.tolist(), "values": a.data.tolist()}
    for src, shift, dst, mask in packed.graph_terms(graph):
        actual[dst] += np.roll(blocks[src], -shift) * mask
    assert np.allclose(actual.reshape(-1, packed.STRIDE)[:n], a @ x, atol=1e-12)
    assert np.all(actual.reshape(-1, packed.STRIDE)[n:] == 0)


@pytest.mark.skipif(os.environ.get("RUN_TRAINED_CKKS_TEST") != "1", reason="Opt-in trained-width CKKS test")
def test_real_ckks_with_frozen_weights_and_cross_block_edges(tmp_path):
    """Real published weights on an internal graph; NOT a full-workload result."""
    import shutil
    copy = tmp_path / "my_method"
    shutil.copytree(ROOT / "submissions/toy_ckks", copy)
    adapter = load_adapter(copy / "adapter.py")
    assert adapter.describe()["name"] == "my_method"
    rng = np.random.default_rng(44)
    n = packed.ROWS + 1
    x = rng.uniform(-1, 1, size=(n, 8))
    row = np.r_[np.arange(n), np.arange(n), np.arange(n)]
    col = np.r_[np.arange(n), (np.arange(n) + 1) % n, (np.arange(n) - 1) % n]
    a = sp.coo_matrix((np.full(len(row), 1 / 3), (row, col)), shape=(n, n))
    weights = read_json(ROOT / "artifacts/scam-list-gcn-100k-v1/weights.json")
    public = {"node_count": n, "feature_count": 8, "sensitive_indices": [4, 6, 7],
              "public_indices": [0, 1, 2, 3, 5], "x_public": x[:, [0, 1, 2, 3, 5]].tolist(),
              "weights": weights, "activation": "poly2",
              "adjacency": {"rows": row.tolist(), "cols": col.tolist(), "values": a.data.tolist()}}
    private, keys = adapter.keygen(2)
    from harness.security import inspect_public_context
    assert inspect_public_context(adapter.describe(), keys, 2)["eligible"]
    encrypted = adapter.encrypt(x[:, [4, 6, 7]], private, 2)
    intermediate = tmp_path / "intermediate"
    intermediate.mkdir()
    result = adapter.evaluate(encrypted, public, keys, 2, intermediate)
    actual = np.asarray(adapter.decrypt(result, private, 2))
    expected = predict(x, a, weights, [4, 6, 7])
    assert actual.shape == (n,)
    assert np.all(np.abs(actual - expected) <= .001 + .001 * np.abs(expected))
    print("trained-weight fixture max error:", np.max(np.abs(actual - expected)))
