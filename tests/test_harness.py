import copy
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
from harness.utils import read_json, write_json, write_blobs
from harness.verify_result import verify


def test_frozen_toy_integrity():
    b = load_bundle(ROOT / "artifacts" / "toy-v1")
    assert b["registered"]
    assert b["public"]["sensitive_indices"] == [4, 6, 7]
    assert "features" not in b["public"] and "labels" not in b["public"]


def test_tampered_artifact_rejected(tmp_path):
    shutil.copytree(ROOT / "artifacts" / "toy-v1", tmp_path / "toy")
    p = tmp_path / "toy" / "weights.json"
    p.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(tmp_path / "toy")


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
               "--debug-plaintext", "--num-runs", "2", "--threads", "1", "--out", str(out)]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    r = read_json(out / "report.json")
    assert r["status"] == "passed" and len(r["runs"]) == 2
    assert not r["eligible_for_comparison"]
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
                        "--debug-plaintext", "--out", str(out)], capture_output=True, text=True, timeout=120)
    assert r.returncode != 0
    assert read_json(out / "report.json")["status"] == "error"
