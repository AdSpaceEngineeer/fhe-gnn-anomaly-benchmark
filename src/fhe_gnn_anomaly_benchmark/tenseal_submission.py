"""Real CKKS submission adapter implemented with open-source TenSEAL."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from .baseline import _read_identifiers
from .identifiers import encode_identifiers
from .model import (
    PolynomialMessagePassingGNN,
    identifier_feature_matrix,
    mean_neighbour_features,
    sigmoid,
)


POLY_MODULUS_DEGREE = 16384
COEFF_MOD_BIT_SIZES = (60, 40, 40, 40, 40, 60)
GLOBAL_SCALE = 2**40
DETAIL_SECONDS: dict[str, float] = {}


def _tenseal():
    try:
        import tenseal as ts
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("TenSEAL submission requires the optional 'tenseal' dependency") from error
    return ts


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


def _measure(name: str, operation):
    start = time.perf_counter()
    result = operation()
    DETAIL_SECONDS[name] = DETAIL_SECONDS.get(name, 0.0) + (time.perf_counter() - start)
    return result


def _peak_rss_bytes() -> int:
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        )
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        process = kernel32.GetCurrentProcess()
        success = psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        )
        return int(counters.PeakWorkingSetSize) if success else 0
    try:  # pragma: no cover - platform dependent
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak * 1024 if sys.platform != "darwin" else peak)
    except (ImportError, OSError):
        return 0


def client_key_generation() -> None:
    ts = _tenseal()

    def generate():
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=POLY_MODULUS_DEGREE,
            coeff_mod_bit_sizes=list(COEFF_MOD_BIT_SIZES),
        )
        context.global_scale = GLOBAL_SCALE
        context.generate_galois_keys()
        return context

    context = _measure("context_and_key_generation", generate)
    (_directory("public_keys", setup=True) / "context.bin").write_bytes(
        context.serialize(
            save_public_key=True,
            save_secret_key=False,
            save_galois_keys=False,
            save_relin_keys=False,
        )
    )
    (_directory("evaluation_keys", setup=True) / "context.bin").write_bytes(
        context.serialize(
            save_public_key=True,
            save_secret_key=False,
            save_galois_keys=True,
            save_relin_keys=True,
        )
    )
    (_directory("client_secret", setup=True) / "context.bin").write_bytes(
        context.serialize(
            save_public_key=True,
            save_secret_key=True,
            save_galois_keys=False,
            save_relin_keys=False,
        )
    )


def server_upload_evaluation_key() -> None:
    shutil.copyfile(
        _directory("evaluation_keys", setup=True) / "context.bin",
        _directory("server_intermediates", setup=True) / "evaluation_context.bin",
    )


def server_preprocess_model() -> None:
    source = _environment_path("FHE_BENCH_MODEL_PATH")
    destination = _directory("server_intermediates", setup=True) / "model.npz"
    shutil.copyfile(source, destination)
    model = PolynomialMessagePassingGNN.load(destination)
    (destination.parent / "model.json").write_text(
        json.dumps(
            {
                "source_sha256": os.environ["FHE_BENCH_MODEL_SHA256"],
                "input_features": int(model.self_weights.shape[0]),
                "hidden_features": int(model.self_weights.shape[1]),
                "activation_quadratic": model.activation_quadratic,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def client_preprocess_input() -> None:
    instance = _environment_path("FHE_BENCH_INSTANCE_DIR")
    identifiers = _read_identifiers(instance / "client_identifiers.txt")
    retention = float(os.environ["FHE_BENCH_RETENTION"])
    encoded = encode_identifiers(identifiers, retention=retention)
    features = identifier_feature_matrix(encoded)
    np.save(_directory("client_preprocessed") / "identifier_features.npy", features)


def client_encrypt_input() -> None:
    ts = _tenseal()
    public_context = ts.context_from(
        (_directory("public_keys", setup=True) / "context.bin").read_bytes()
    )
    features = np.load(
        _directory("client_preprocessed") / "identifier_features.npy"
    ).astype(np.float64)
    destination = _directory("ciphertexts_upload")

    def encrypt_all() -> None:
        for index, row in enumerate(features):
            ciphertext = ts.ckks_vector(public_context, row.tolist())
            (destination / f"node-{index:06d}.bin").write_bytes(ciphertext.serialize())

    _measure("encrypt_identifier_vectors", encrypt_all)
    (destination / "index.json").write_text(
        json.dumps(
            {
                "ciphertext_count": int(len(features)),
                "logical_features_per_node": int(features.shape[1]),
                "packing": "one encrypted 72-feature histogram per graph node",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def server_encrypted_compute() -> None:
    ts = _tenseal()
    setup = _directory("server_intermediates", setup=True)
    context = ts.context_from((setup / "evaluation_context.bin").read_bytes())
    model = PolynomialMessagePassingGNN.load(setup / "model.npz")
    instance = _environment_path("FHE_BENCH_INSTANCE_DIR")
    with np.load(instance / "public_graph.npz") as public:
        public_features = public["features"].astype(np.float64)
        edges = public["edges"].astype(np.int64)
        targets = public["target_indices"].astype(np.int64)

    public_count = public_features.shape[1]
    identifier_slice = slice(public_count, len(model.feature_mean))
    normalized_public = (
        public_features - model.feature_mean[:public_count]
    ) / model.feature_scale[:public_count]
    neighbour_public = mean_neighbour_features(normalized_public, edges)
    identifier_scale = model.feature_scale[identifier_slice]
    identifier_mean_scaled = model.feature_mean[identifier_slice] / identifier_scale
    self_identifier_weights = model.self_weights[identifier_slice] / identifier_scale[:, None]
    neighbour_identifier_weights = (
        model.neighbour_weights[identifier_slice] / identifier_scale[:, None]
    )
    self_identifier_offset = -identifier_mean_scaled @ model.self_weights[identifier_slice]
    neighbour_identifier_offset = (
        -identifier_mean_scaled @ model.neighbour_weights[identifier_slice]
    )
    uploaded = _directory("ciphertexts_upload")
    destination = _directory("ciphertexts_download")

    def load_ciphertext(index: int):
        return ts.ckks_vector_from(
            context, (uploaded / f"node-{index:06d}.bin").read_bytes()
        )

    def evaluate_targets() -> None:
        for output_index, target in enumerate(targets):
            sources = edges[edges[:, 1] == target, 0]
            self_part = load_ciphertext(int(target)).matmul(
                self_identifier_weights.tolist()
            )
            if len(sources):
                neighbour_features = load_ciphertext(int(sources[0]))
                for source in sources[1:]:
                    neighbour_features += load_ciphertext(int(source))
                neighbour_part = neighbour_features.matmul(
                    (neighbour_identifier_weights / len(sources)).tolist()
                )
                neighbour_offset = neighbour_identifier_offset
            else:
                neighbour_part = load_ciphertext(int(target)) * 0
                neighbour_part = neighbour_part.matmul(
                    neighbour_identifier_weights.tolist()
                )
                neighbour_offset = 0.0
            plaintext_z = (
                normalized_public[target] @ model.self_weights[:public_count]
                + neighbour_public[target] @ model.neighbour_weights[:public_count]
                + model.hidden_bias
                + self_identifier_offset
                + neighbour_offset
            )
            preactivation = self_part + neighbour_part + plaintext_z.tolist()
            hidden = (
                preactivation
                + preactivation.square() * model.activation_quadratic
            )
            logit = hidden.dot(model.output_weights.tolist()) + model.output_bias
            (destination / f"logit-{output_index:06d}.bin").write_bytes(
                logit.serialize()
            )

    _measure("homomorphic_gnn_evaluation", evaluate_targets)
    (destination / "index.json").write_text(
        json.dumps({"target_count": int(len(targets))}), encoding="utf-8"
    )


def client_decrypt_output() -> None:
    ts = _tenseal()
    secret_context = ts.context_from(
        (_directory("client_secret", setup=True) / "context.bin").read_bytes()
    )
    source = _directory("ciphertexts_download")
    target_count = json.loads((source / "index.json").read_text(encoding="utf-8"))[
        "target_count"
    ]

    def decrypt_all() -> np.ndarray:
        return np.asarray(
            [
                ts.ckks_vector_from(
                    secret_context,
                    (source / f"logit-{index:06d}.bin").read_bytes(),
                ).decrypt()[0]
                for index in range(target_count)
            ],
            dtype=np.float64,
        )

    logits = _measure("decrypt_logits", decrypt_all)
    np.save(_directory("decrypted_outputs") / "logits.npy", logits)


def client_postprocess() -> None:
    logits = np.load(_directory("decrypted_outputs") / "logits.npy")
    np.save(_directory("client_outputs") / "scores.npy", sigmoid(logits))


STAGES = {
    "client_key_generation": client_key_generation,
    "server_upload_evaluation_key": server_upload_evaluation_key,
    "server_preprocess_model": server_preprocess_model,
    "client_preprocess_input": client_preprocess_input,
    "client_encrypt_input": client_encrypt_input,
    "server_encrypted_compute": server_encrypted_compute,
    "client_decrypt_output": client_decrypt_output,
    "client_postprocess": client_postprocess,
}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in STAGES:
        raise SystemExit(f"usage: python -m {__package__}.tenseal_submission <stage>")
    STAGES[sys.argv[1]]()
    _environment_path("FHE_BENCH_STAGE_REPORT").write_text(
        json.dumps(
            {
                "peak_ram_bytes": _peak_rss_bytes(),
                "detail_seconds": DETAIL_SECONDS,
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
