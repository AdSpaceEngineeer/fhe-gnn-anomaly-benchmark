"""End-to-end benchmark orchestration and result reporting."""

from __future__ import annotations

import hashlib
import json
import platform
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .baseline import score_plaintext_instance
from .harness import (
    INFERENCE_STAGES,
    INITIALIZATION_STAGES,
    ROTATION_STAGES,
    StageRun,
    SubmissionManifest,
    artifact_sizes,
    run_stage_sequence,
)
from .identifiers import retained_character_count
from .instance import prepare_inference_instance
from .metrics import BinaryMetrics, binary_metrics, quality_retention
from .model_bundle import ModelBundle


def _metric_payload(metrics: BinaryMetrics) -> dict[str, float | int]:
    return asdict(metrics)


def _stage_lookup(stage_run: StageRun, name: str) -> float:
    for measurement in stage_run.measurements:
        if measurement.stage == name:
            return measurement.wall_seconds
    return 0.0


def _hardware_metadata() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "not reported",
        "python": platform.python_version(),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_unchanged_model(path: Path, expected_sha256: str) -> None:
    if _sha256(path) != expected_sha256:
        raise RuntimeError("submission modified the frozen model artifact")


def run_benchmark(
    submission_manifest_path: str | Path,
    dataset_directory: str | Path,
    model_path: str | Path,
    output_directory: str | Path,
    *,
    retention: float,
    batch_size: int,
    num_runs: int = 3,
    seed: int = 2026,
) -> list[Path]:
    """Run one retention/batch configuration and write one result per run."""

    if num_runs < 1:
        raise ValueError("num_runs must be positive")
    manifest_path = Path(submission_manifest_path).resolve()
    manifest = SubmissionManifest.load(manifest_path)
    model_bundle = ModelBundle.load(model_path)
    frozen_model = model_bundle.artifact_path
    model_sha256 = model_bundle.artifact_sha256
    output = Path(output_directory).resolve()
    session = output / "work" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )
    instance_directory = session / "instance"
    instance_metadata = prepare_inference_instance(
        dataset_directory,
        instance_directory,
        batch_size=batch_size,
        seed=seed,
    )
    identifiers = (instance_directory / "client_identifiers.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    full_length = len(identifiers[0])
    retained = retained_character_count(full_length, retention)

    setup_directory = session / "setup"
    common_environment = {
        "FHE_BENCH_INSTANCE_DIR": str(instance_directory),
        "FHE_BENCH_MODEL_PATH": str(frozen_model),
        "FHE_BENCH_MODEL_SHA256": model_sha256,
        "FHE_BENCH_MODEL_BUNDLE_PATH": str(model_bundle.manifest_path or ""),
        "FHE_BENCH_SETUP_DIR": str(setup_directory),
        "FHE_BENCH_RETENTION": str(retention),
        "FHE_BENCH_RETAINED_CHARACTERS": str(retained),
        "FHE_BENCH_FULL_LENGTH": str(full_length),
    }
    initialization = run_stage_sequence(
        manifest,
        INITIALIZATION_STAGES,
        submission_directory=manifest_path.parent,
        work_directory=setup_directory,
        environment=common_environment,
    )
    _require_unchanged_model(frozen_model, model_sha256)
    setup_sizes = artifact_sizes(setup_directory)

    labels, plaintext_full_scores, threshold = score_plaintext_instance(
        instance_directory, frozen_model, retention=1.0
    )
    _, plaintext_truncated_scores, truncated_threshold = score_plaintext_instance(
        instance_directory, frozen_model, retention=retention
    )
    if threshold != truncated_threshold:
        raise RuntimeError("plaintext baselines did not use the same frozen threshold")
    plaintext_full = binary_metrics(labels, plaintext_full_scores, threshold=threshold)
    plaintext_truncated = binary_metrics(
        labels, plaintext_truncated_scores, threshold=threshold
    )

    result_paths: list[Path] = []
    for run_index in range(1, num_runs + 1):
        run_directory = session / f"run-{run_index}"
        rotation = StageRun(())
        if manifest.key_policy == "ephemeral_per_batch" and run_index > 1:
            rotation = run_stage_sequence(
                manifest,
                ROTATION_STAGES,
                submission_directory=manifest_path.parent,
                work_directory=run_directory,
                environment=common_environment,
            )
        inference = run_stage_sequence(
            manifest,
            INFERENCE_STAGES,
            submission_directory=manifest_path.parent,
            work_directory=run_directory,
            environment=common_environment,
        )
        _require_unchanged_model(frozen_model, model_sha256)
        score_path = run_directory / "client_outputs" / "scores.npy"
        if not score_path.is_file():
            raise RuntimeError(f"submission did not produce required scores: {score_path}")
        submitted_scores = np.load(score_path).astype(np.float64).reshape(-1)
        protected = binary_metrics(labels, submitted_scores, threshold=threshold)
        q_value: float | None
        if plaintext_full.recall == 0 or plaintext_full.f1 == 0:
            q_value = None
        else:
            q_value = quality_retention(
                recall_encrypted=protected.recall,
                f1_encrypted=protected.f1,
                recall_plaintext_full=plaintext_full.recall,
                f1_plaintext_full=plaintext_full.f1,
            )

        run_sizes = artifact_sizes(run_directory)
        artifact_bytes = {
            **{f"setup_{name}": size for name, size in setup_sizes.items()},
            **{f"run_{name}": size for name, size in run_sizes.items()},
        }
        client_to_server = (
            setup_sizes["public_keys"]
            + setup_sizes["evaluation_keys"]
            + run_sizes["ciphertexts_upload"]
            + run_sizes["rotated_evaluation_keys"]
        )
        server_to_client = (
            run_sizes["ciphertexts_download"] + run_sizes["encrypted_outputs"]
        )
        stage_measurements = (
            *initialization.measurements,
            *rotation.measurements,
            *inference.measurements,
        )
        stage_seconds = {
            measurement.stage: measurement.wall_seconds
            for measurement in stage_measurements
        }
        peak_by_stage = {
            measurement.stage: measurement.peak_ram_bytes
            for measurement in stage_measurements
        }
        online_latency = inference.total_seconds
        key_generation = _stage_lookup(initialization, "client_key_generation")
        key_upload = _stage_lookup(initialization, "server_upload_evaluation_key")
        rotation_seconds = rotation.total_seconds if rotation.measurements else None
        if manifest.key_policy == "ephemeral_per_batch":
            key_overhead = (
                key_generation + key_upload
                if run_index == 1
                else (rotation_seconds or 0.0)
            ) / batch_size
        else:
            key_overhead = (key_generation + key_upload) / (batch_size * num_runs)
        result = {
            "schema_version": "0.4.0",
            "conformance": {
                "is_fhe": manifest.is_fhe,
                "meets_128_bit_security_target": (
                    manifest.is_fhe and manifest.claimed_security_bits >= 128
                ),
                "eligible_for_fhe_comparison": (
                    manifest.is_fhe and manifest.claimed_security_bits >= 128
                ),
                "warning": None
                if manifest.is_fhe
                else "Protocol dry run only; no identifier confidentiality",
            },
            "run": {
                "submission": manifest.name,
                "implementation": manifest.implementation,
                "dataset": "YelpChi",
                "run_index": run_index,
                "num_runs": num_runs,
                "seed": seed,
                "batch_size": batch_size,
                "subgraph_nodes": instance_metadata.subgraph_nodes,
                "subgraph_edges": instance_metadata.subgraph_edges,
                "retention_requested": retention,
                "retained_characters": retained,
                "full_length": full_length,
                "retention_realized": retained / full_length,
                "hardware": _hardware_metadata(),
            },
            "model": {
                "name": model_bundle.name,
                "adapter": model_bundle.adapter,
                "artifact": frozen_model.name,
                "sha256": model_sha256,
                "recommended_baseline": model_bundle.recommended_baseline,
            },
            "cryptography": {
                "scheme": manifest.scheme,
                "parameters": manifest.parameters,
                "claimed_security_bits": manifest.claimed_security_bits,
                "key_policy": manifest.key_policy,
                "additional_leakage": list(manifest.additional_leakage),
            },
            "quality": {
                "protected": _metric_payload(protected),
                "plaintext_full": _metric_payload(plaintext_full),
                "plaintext_truncated": _metric_payload(plaintext_truncated),
                "q": q_value,
                "max_abs_score_error_vs_plaintext_truncated": float(
                    np.max(np.abs(submitted_scores - plaintext_truncated_scores))
                ),
                "mean_abs_score_error_vs_plaintext_truncated": float(
                    np.mean(np.abs(submitted_scores - plaintext_truncated_scores))
                ),
                "threshold": threshold,
            },
            "performance": {
                "online_latency_seconds": online_latency,
                "initialization_seconds": initialization.total_seconds,
                "throughput_targets_per_second": batch_size / online_latency
                if online_latency
                else 0.0,
                "peak_ram_bytes": max(peak_by_stage.values(), default=0),
                "peak_ram_by_stage_bytes": peak_by_stage,
                "storage_bytes": sum(artifact_bytes.values()),
                "artifact_bytes": artifact_bytes,
                "communication_bytes": client_to_server + server_to_client,
                "communication_by_direction_bytes": {
                    "client_to_server": client_to_server,
                    "server_to_client": server_to_client,
                },
                "stage_seconds": stage_seconds,
                "harness_instance_generation_seconds": instance_metadata.generation_seconds,
                "key_generation_seconds": key_generation,
                "evaluation_key_bytes": setup_sizes["evaluation_keys"],
                "evaluation_key_upload_seconds": key_upload,
                "key_rotation_seconds": rotation_seconds,
                "amortized_key_overhead_seconds_per_target": key_overhead,
            },
        }
        result_directory = output / "measurements" / f"batch-{batch_size}" / f"k-{retained}"
        result_directory.mkdir(parents=True, exist_ok=True)
        result_path = result_directory / f"results-{run_index}.json"
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result_paths.append(result_path)
    return result_paths
