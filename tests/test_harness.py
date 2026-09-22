import copy
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pytest
from harness.params import ROOT
from harness.generate_input import load_bundle
from harness.model import predict
from harness.security import validate_description
from harness.utils import read_json, write_json, write_blobs, sha256
from harness.verify_result import verify


def test_frozen_toy_integrity():
    b = load_bundle(ROOT / "tests/fixtures/arithmetic")
    assert not b["registered"]  # legacy arithmetic fixture is not a benchmark workload
    assert b["public"]["sensitive_indices"] == [4, 6, 7]
    assert "features" not in b["public"] and "labels" not in b["public"]


def test_tampered_artifact_rejected(tmp_path):
    shutil.copytree(ROOT / "tests/fixtures/arithmetic", tmp_path / "toy")
    p = tmp_path / "toy" / "weights.json"
    p.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(tmp_path / "toy")


def test_compressed_bundle_preserves_identity(tmp_path):
    folder = tmp_path / "toy"
    shutil.copytree(ROOT / "tests/fixtures/arithmetic", folder)
    original = load_bundle(folder)
    path = folder / "data.json"
    with gzip.open(str(path) + ".gz", "wb") as handle:
        handle.write(path.read_bytes())
    with pytest.raises(ValueError, match="Ambiguous"):
        load_bundle(folder)
    path.unlink()
    compressed = load_bundle(folder)
    assert not compressed["registered"]
    assert compressed["manifest_sha256"] == original["manifest_sha256"]
    assert compressed["reference"] == original["reference"]
    assert compressed["data"] == original["data"]
    with gzip.open(str(path) + ".gz", "wb") as handle:
        handle.write(b"{}")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(folder)


def test_frozen_trained_baseline():
    from harness.metrics import quality
    folder = ROOT / "artifacts" / "scam-list-gcn-100k-v1"
    bundle = load_bundle(folder)  # checks all hashes and all 100,000 reference scores
    assert bundle["registered"]
    assert bundle["public"]["node_count"] == 100000
    assert bundle["public"]["sensitive_indices"] == [4, 6, 7]
    assert sha256(folder / "scam_list_gcn.pt") == bundle["manifest"]["original_model_sha256"]
    data, reference = bundle["data"], bundle["reference"]
    assert [len(data["splits"][s]) for s in ("train", "val", "test")] == [64000, 16000, 20000]
    indices = data["splits"]["test"]
    actual = quality(np.asarray(data["labels"])[indices], np.asarray(reference["scores"])[indices], reference["threshold"])
    assert actual["f1"] == pytest.approx(0.9152542372881356)
    assert actual["recall"] == pytest.approx(0.8977832512315271)
    assert actual["accuracy"] == pytest.approx(0.99325)
    assert actual == pytest.approx(reference["test"], abs=1e-14)


def test_tampered_checkpoint_rejected(tmp_path):
    folder = tmp_path / "toy"
    shutil.copytree(ROOT / "tests/fixtures/arithmetic", folder)
    (folder / "scam_list_gcn.pt").write_bytes(b"not the expected checkpoint")
    with pytest.raises(ValueError, match="checkpoint checksum"):
        load_bundle(folder)


def test_model_matches_hand_calculation():
    from harness.model import LAYERS
    w = {}
    for name in LAYERS:
        w[name + ".weight"], w[name + ".bias"] = [[2.]], [0.]
    x = np.array([[1.]])
    h = 1.
    for _ in range(3):
        h *= 2
        h += .125 * h*h
    expected = (2*h - 1)**2
    assert predict(x, np.eye(1), w, [0])[0] == pytest.approx(expected)


@pytest.mark.parametrize("scores", [[float("nan")], [float("inf")], [[1.]], [1., 2.]])
def test_bad_scores_rejected(scores):
    with pytest.raises(ValueError):
        verify(scores, [1.], [1], [0], .5, .001, .001)


def test_quality_is_separate_from_numerical_agreement():
    result = verify([2., 0.], [1., 0.], [1, 0], [0, 1], .5, .001, .001)
    assert not result["passed"]
    assert result["quality"]["f1"] == 1.


def description():
    return {"is_fhe": True, "security": {"classical_bits": 128, "validator": "seal_tc128", "evidence": "SEAL"},
            "parameters": {"poly_modulus_degree": 8192, "coeff_mod_bit_sizes": [60, 40, 40, 60]},
            "encoding": "CKKS", "packing": "scalar", "activation": "poly2"}


def test_weak_security_rejected():
    d = description()
    d["security"]["classical_bits"] = 127
    with pytest.raises(ValueError, match="128"):
        validate_description(d)


def test_insecure_modulus_rejected():
    d = description()
    d["parameters"]["coeff_mod_bit_sizes"] = [60, 40, 40, 40, 60]
    with pytest.raises(ValueError, match="bound"):
        validate_description(d)


def test_unknown_security_needs_review():
    d = description()
    d["security"]["validator"] = "novel_scheme"
    assert not validate_description(d)["eligible"]


def test_debug_is_never_fhe():
    with pytest.raises(ValueError, match="debug"):
        validate_description({"is_fhe": False})
    assert not validate_description({"is_fhe": False}, True)["eligible"]


def test_payload_paths_cannot_escape(tmp_path):
    with pytest.raises(ValueError):
        write_blobs(tmp_path, {"../secret": b"x"})


def test_pipeline_repeat_and_client_server_split(tmp_path):
    out = tmp_path / "run"
    command = [sys.executable, str(ROOT / "harness/run_submission.py"), "--submission", "plaintext_debug",
               "--artifacts", str(ROOT / "tests/fixtures/arithmetic"),
               "--debug-plaintext", "--num-runs", "2", "--threads", "1", "--out", str(out)]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    r = read_json(out / "report.json")
    assert r["status"] == "passed" and len(r["runs"]) == 2
    assert not r["eligible_for_comparison"]
    assert (out / "comparison.md").is_file()
    for run in r["runs"]:
        assert run["verification"]["max_absolute_error"] < 1e-9
        assert run["stages"]["evaluate"]["wall_seconds"] > 0
    public = read_json(out / "io/run-000/server/public.json")
    assert "labels" not in public and "sensitive" not in public
    assert (out / "io/run-000/client/input.json").is_file()
    again = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert again.returncode != 0  # protect existing reports


def test_wrong_submission_fails_with_report(tmp_path):
    folder = tmp_path / "bad"
    folder.mkdir()
    # Minimal bad adapter uses the honest debug implementation but emits NaN.
    adapter = folder / "adapter.py"
    adapter.write_text("from submissions.plaintext_debug.adapter import Adapter as Base\n"
                       "class Adapter(Base):\n"
                       "    def decrypt(self, *args): return [float('nan')]*3\n")
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(ROOT / "harness/run_submission.py"), "--submission", str(adapter),
                        "--artifacts", str(ROOT / "tests/fixtures/arithmetic"),
                        "--debug-plaintext", "--out", str(out)], capture_output=True, text=True, timeout=120)
    assert r.returncode != 0
    assert read_json(out / "report.json")["status"] == "error"


@pytest.mark.parametrize("valid", [True, False])
def test_optional_timings_end_to_end(tmp_path, valid):
    folder = tmp_path / "timed_debug"
    folder.mkdir()
    timings = '{"Encrypted computation": 0.01, "I/O": 0.02}' if valid else '{"I/O": -1}'
    adapter = folder / "adapter.py"
    adapter.write_text(
        "from submissions.plaintext_debug.adapter import Adapter as Base\n"
        "class Adapter(Base):\n"
        "    def evaluate(self, encrypted, public, keys, threads, intermediate_dir):\n"
        "        result = super().evaluate(encrypted, public, keys, threads, intermediate_dir)\n"
        f"        (intermediate_dir / 'server_reported_steps.json').write_text({timings!r})\n"
        "        return result\n", encoding="utf-8")
    out = tmp_path / "results"
    result = subprocess.run(
        [sys.executable, str(ROOT / "harness/run_submission.py"), "--submission", str(adapter),
         "--artifacts", str(ROOT / "tests/fixtures/arithmetic"), "--debug-plaintext",
         "--threads", "1", "--out", str(out)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    report = read_json(out / "report.json")
    run = report["runs"][0]
    assert run["verification"]["passed"]
    assert run["storage_bytes"]["persisted_intermediates"] == 0
    stage = run["stages"]["evaluate"]
    assert stage["harness_file_io_seconds"] >= 0
    assert stage["adapter_call_seconds"] > 0
    assert stage["wall_seconds"] >= stage["adapter_call_seconds"]
    table = (out / "comparison.md").read_text(encoding="utf-8")
    if valid:
        assert run["server_reported_steps"] == json.loads(timings)
        assert not run["server_timing_warnings"]
        assert "| Encrypted computation | 0.01 | 1/1 |" in table
    else:
        assert run["server_reported_steps"] is None
        assert run["server_timing_warnings"]
        assert "Ignored optional server timings" in table
