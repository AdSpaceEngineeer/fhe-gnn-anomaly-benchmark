"""Plaintext-only protocol dry run. This module is not an FHE submission."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tracemalloc
from pathlib import Path

import numpy as np

from .identifiers import EncodedIdentifier, encode_identifiers
from .model import PolynomialMessagePassingGNN, identifier_feature_matrix, sigmoid


def _environment_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing required environment variable {name}")
    return Path(value)


def _directory(name: str, *, setup: bool = False) -> Path:
    root = _environment_path("FHE_BENCH_SETUP_DIR" if setup else "FHE_BENCH_WORK_DIR")
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def client_key_generation() -> None:
    (_directory("public_keys", setup=True) / "DRY_RUN.json").write_text(
        json.dumps({"warning": "plaintext dry run; not a cryptographic key"}),
        encoding="utf-8",
    )
    (_directory("evaluation_keys", setup=True) / "DRY_RUN.json").write_text(
        json.dumps({"warning": "plaintext dry run; not an evaluation key"}),
        encoding="utf-8",
    )


def server_preprocess_model() -> None:
    model = PolynomialMessagePassingGNN.load(_environment_path("FHE_BENCH_MODEL_PATH"))
    (_directory("server_intermediates", setup=True) / "model.json").write_text(
        json.dumps(
            {
                "input_features": int(model.self_weights.shape[0]),
                "hidden_features": int(model.self_weights.shape[1]),
                "activation": f"z + {model.activation_quadratic} * z^2",
            }
        ),
        encoding="utf-8",
    )


def client_preprocess_input() -> None:
    instance = _environment_path("FHE_BENCH_INSTANCE_DIR")
    identifiers = (instance / "client_identifiers.txt").read_text(encoding="utf-8").splitlines()
    retention = float(os.environ["FHE_BENCH_RETENTION"])
    encoded = encode_identifiers(identifiers, retention=retention)
    values = np.asarray([item.values for item in encoded], dtype=np.int16)
    sides = np.asarray([item.sides for item in encoded], dtype=np.int8)
    np.savez_compressed(
        _directory("client_preprocessed") / "identifiers.npz",
        values=values,
        sides=sides,
        full_length=np.array(encoded[0].full_length, dtype=np.int16),
        retained_characters=np.array(encoded[0].retained_characters, dtype=np.int16),
    )


def client_encrypt_input() -> None:
    shutil.copyfile(
        _directory("client_preprocessed") / "identifiers.npz",
        _directory("ciphertexts_upload") / "PLAINTEXT_IDENTIFIERS.npz",
    )


def server_encrypted_compute() -> None:
    instance = _environment_path("FHE_BENCH_INSTANCE_DIR")
    with np.load(instance / "public_graph.npz") as public:
        public_features = public["features"]
        edges = public["edges"]
        targets = public["target_indices"].astype(np.int64)
    with np.load(_directory("ciphertexts_upload") / "PLAINTEXT_IDENTIFIERS.npz") as encoded:
        values = encoded["values"]
        sides = encoded["sides"]
        full_length = int(encoded["full_length"])
        retained = int(encoded["retained_characters"])
    logical = [
        EncodedIdentifier(
            values=tuple(int(value) for value in row_values),
            sides=tuple(int(side) for side in row_sides),
            retained_characters=retained,
            full_length=full_length,
        )
        for row_values, row_sides in zip(values, sides, strict=True)
    ]
    features = np.concatenate(
        (public_features, identifier_feature_matrix(logical)), axis=1
    )
    model = PolynomialMessagePassingGNN.load(_environment_path("FHE_BENCH_MODEL_PATH"))
    np.save(
        _directory("ciphertexts_download") / "PLAINTEXT_LOGITS.npy",
        model.logits(features, edges)[targets],
    )


def client_decrypt_output() -> None:
    shutil.copyfile(
        _directory("ciphertexts_download") / "PLAINTEXT_LOGITS.npy",
        _directory("decrypted_outputs") / "logits.npy",
    )


def client_postprocess() -> None:
    logits = np.load(_directory("decrypted_outputs") / "logits.npy")
    np.save(_directory("client_outputs") / "scores.npy", sigmoid(logits))


STAGES = {
    "client_key_generation": client_key_generation,
    "server_preprocess_model": server_preprocess_model,
    "client_preprocess_input": client_preprocess_input,
    "client_encrypt_input": client_encrypt_input,
    "server_encrypted_compute": server_encrypted_compute,
    "client_decrypt_output": client_decrypt_output,
    "client_postprocess": client_postprocess,
}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in STAGES:
        raise SystemExit(f"usage: python -m {__package__}.reference_submission <stage>")
    tracemalloc.start()
    STAGES[sys.argv[1]]()
    _, peak = tracemalloc.get_traced_memory()
    report = _environment_path("FHE_BENCH_STAGE_REPORT")
    report.write_text(json.dumps({"peak_ram_bytes": peak}), encoding="utf-8")


if __name__ == "__main__":
    main()
