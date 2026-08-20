import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fhe_gnn_anomaly_benchmark.model_bundle import ModelBundle, write_model_bundle


class ModelBundleTests(unittest.TestCase):
    def test_bundle_registers_and_verifies_compatible_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "weights.npz"
            np.savez_compressed(model, value=np.array([1.0]))
            manifest = write_model_bundle(
                model,
                root / "model.json",
                name="alternative-training-run",
            )
            loaded = ModelBundle.load(manifest)
            self.assertEqual(loaded.name, "alternative-training-run")
            self.assertEqual(loaded.artifact_path, model.resolve())
            self.assertFalse(loaded.recommended_baseline)

            material = json.loads(manifest.read_text(encoding="utf-8"))
            material["artifact_sha256"] = "0" * 64
            manifest.write_text(json.dumps(material), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                ModelBundle.load(manifest)


if __name__ == "__main__":
    unittest.main()
