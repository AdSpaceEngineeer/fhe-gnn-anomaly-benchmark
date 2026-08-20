import json
import sys
import tempfile
import unittest
from pathlib import Path

from fhe_gnn_anomaly_benchmark.harness import (
    SubmissionManifest,
    artifact_sizes,
    run_stage,
)


class HarnessTests(unittest.TestCase):
    def test_manifest_enforces_security_for_fhe_submission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "submission.json"
            material = {
                "name": "test",
                "implementation": "test backend",
                "scheme": "example",
                "claimed_security_bits": 100,
                "commands": {
                    stage: ["program"]
                    for stage in (
                        "client_key_generation",
                        "server_preprocess_model",
                        "client_preprocess_input",
                        "client_encrypt_input",
                        "server_encrypted_compute",
                        "client_decrypt_output",
                        "client_postprocess",
                    )
                },
            }
            path.write_text(json.dumps(material), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "128-bit"):
                SubmissionManifest.load(path)

    def test_stage_wall_time_and_self_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            program = (
                "import json,os;"
                "open(os.environ['FHE_BENCH_STAGE_REPORT'],'w').write("
                "json.dumps({'peak_ram_bytes':1234,'detail_seconds':{'compute':0.25}}))"
            )
            measured = run_stage(
                "server_encrypted_compute",
                [sys.executable, "-c", program],
                submission_directory=root,
                work_directory=root / "work",
            )
            self.assertGreaterEqual(measured.wall_seconds, 0)
            self.assertEqual(measured.peak_ram_bytes, 1234)
            self.assertEqual(measured.detail_seconds, {"compute": 0.25})

    def test_ephemeral_policy_requires_rotation_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "submission.json"
            material = {
                "name": "test",
                "implementation": "test backend",
                "scheme": "example",
                "claimed_security_bits": 128,
                "key_policy": "ephemeral_per_batch",
                "commands": {
                    stage: ["program"]
                    for stage in (
                        "client_key_generation",
                        "server_preprocess_model",
                        "client_preprocess_input",
                        "client_encrypt_input",
                        "server_encrypted_compute",
                        "client_decrypt_output",
                        "client_postprocess",
                    )
                },
            }
            path.write_text(json.dumps(material), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "rotation stages"):
                SubmissionManifest.load(path)

    def test_artifact_sizes_use_exact_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "ciphertexts_upload"
            artifact.mkdir()
            (artifact / "part-1.bin").write_bytes(b"123")
            (artifact / "part-2.bin").write_bytes(b"4567")
            sizes = artifact_sizes(root)
            self.assertEqual(sizes["ciphertexts_upload"], 7)
            self.assertEqual(sizes["public_keys"], 0)


if __name__ == "__main__":
    unittest.main()
