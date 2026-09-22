"""Actual encryption test; opt in with RUN_FHE_TESTS=1."""
import os
import subprocess
import sys
import pytest
from harness.params import ROOT
from harness.utils import read_json


@pytest.mark.skipif(os.environ.get("RUN_FHE_TESTS") != "1", reason="Set RUN_FHE_TESTS=1 for real CKKS")
def test_real_ckks_pipeline(tmp_path):
    pytest.importorskip("tenseal")
    out = tmp_path / "ckks"
    result = subprocess.run([sys.executable, str(ROOT / "harness/run_submission.py"),
                             "--submission", "toy_ckks", "--out", str(out), "--threads", "2"],
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    report = read_json(out / "report.json")
    assert report["security"]["status"] == "seal_tc128_context_checked"
    assert report["runs"][0]["verification"]["passed"]
    import tenseal as ts
    public = ts.context_from((out / "io/server/keys/context.bin").read_bytes(), n_threads=2)
    assert not public.has_secret_key()
    assert report["runs"][0]["communication_bytes"]["client_to_server_input"] > 0


@pytest.mark.skipif(os.environ.get("RUN_FHE_TESTS") != "1", reason="Set RUN_FHE_TESTS=1")
def test_actual_context_rejects_secret_key_and_parameter_mismatch():
    ts = pytest.importorskip("tenseal")
    from harness.security import inspect_public_context
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                     coeff_mod_bit_sizes=[60,40,40,60], n_threads=1)
    d = {"security": {"validator": "seal_tc128"},
         "parameters": {"poly_modulus_degree": 8192, "coeff_mod_bit_sizes": [60,40,40,60]}}
    private = ctx.serialize(save_secret_key=True)
    with pytest.raises(ValueError, match="secret key"):
        inspect_public_context(d, {"context.bin": private}, 1)
    public = ctx.serialize(save_secret_key=False)
    d["parameters"]["coeff_mod_bit_sizes"] = [60,30,30,60]
    with pytest.raises(ValueError, match="differs"):
        inspect_public_context(d, {"context.bin": public}, 1)
