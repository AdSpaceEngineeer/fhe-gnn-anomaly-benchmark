"""Semantic result checks that complement the JSON Schema."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .metrics import quality_retention


def validate_result_semantics(result: dict[str, Any]) -> None:
    if result.get("schema_version") != "0.4.0":
        raise ValueError("unsupported result schema version")
    for section in ("conformance", "run", "model", "cryptography", "quality", "performance"):
        if not isinstance(result.get(section), dict):
            raise ValueError(f"result requires object section {section!r}")

    model = result["model"]
    if (
        not isinstance(model.get("sha256"), str)
        or len(model["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in model["sha256"])
    ):
        raise ValueError("model sha256 must be 64 lowercase hexadecimal characters")
    for field in ("name", "adapter", "artifact"):
        if not isinstance(model.get(field), str) or not model[field]:
            raise ValueError(f"model {field} must be a non-empty string")
    if not isinstance(model.get("recommended_baseline"), bool):
        raise ValueError("model recommended_baseline must be boolean")

    run = result["run"]
    retained = run["retained_characters"]
    full_length = run["full_length"]
    if not 2 <= retained <= full_length:
        raise ValueError("retained_characters must lie between 2 and full_length")
    if not math.isclose(run["retention_realized"], retained / full_length, rel_tol=1e-12):
        raise ValueError("retention_realized is inconsistent with k/L")
    if run["run_index"] > run["num_runs"]:
        raise ValueError("run_index exceeds num_runs")

    quality = result["quality"]
    for name in ("protected", "plaintext_full", "plaintext_truncated"):
        metrics = quality[name]
        confusion_total = sum(
            metrics[key]
            for key in (
                "true_positives",
                "false_positives",
                "false_negatives",
                "true_negatives",
            )
        )
        if confusion_total != run["batch_size"]:
            raise ValueError(f"{name} confusion matrix does not match batch size")
    if quality["q"] is not None:
        expected_q = quality_retention(
            recall_encrypted=quality["protected"]["recall"],
            f1_encrypted=quality["protected"]["f1"],
            recall_plaintext_full=quality["plaintext_full"]["recall"],
            f1_plaintext_full=quality["plaintext_full"]["f1"],
        )
        if not math.isclose(quality["q"], expected_q, rel_tol=1e-12):
            raise ValueError("Q(k) is inconsistent with Recall/F1 values")
    for field in (
        "max_abs_score_error_vs_plaintext_truncated",
        "mean_abs_score_error_vs_plaintext_truncated",
    ):
        if not math.isfinite(quality[field]) or quality[field] < 0:
            raise ValueError(f"{field} must be finite and non-negative")

    performance = result["performance"]
    if performance["storage_bytes"] != sum(performance["artifact_bytes"].values()):
        raise ValueError("storage_bytes is inconsistent with artifact_bytes")
    if performance["communication_bytes"] != sum(
        performance["communication_by_direction_bytes"].values()
    ):
        raise ValueError("communication_bytes is inconsistent with directional totals")
    expected_throughput = (
        run["batch_size"] / performance["online_latency_seconds"]
        if performance["online_latency_seconds"]
        else 0
    )
    if not math.isclose(
        performance["throughput_targets_per_second"], expected_throughput, rel_tol=1e-12
    ):
        raise ValueError("throughput is inconsistent with batch size and latency")

    conformance = result["conformance"]
    cryptography = result["cryptography"]
    expected_security = bool(
        conformance["is_fhe"] and cryptography["claimed_security_bits"] >= 128
    )
    if conformance["meets_128_bit_security_target"] != expected_security:
        raise ValueError("security conformance flag is inconsistent")
    if conformance["eligible_for_fhe_comparison"] and not expected_security:
        raise ValueError("ineligible submission is marked comparable")
    key_policy = cryptography["key_policy"]
    rotation = performance["key_rotation_seconds"]
    if key_policy == "ephemeral_per_batch" and run["run_index"] > 1 and rotation is None:
        raise ValueError("ephemeral run after the first must report key rotation time")
    if key_policy == "reused" and rotation is not None:
        raise ValueError("reused-key submission must not report per-batch key rotation")


def validate_result_file(path: str | Path) -> dict[str, Any]:
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("result file must contain a JSON object")
    validate_result_semantics(result)
    return result


def validate_comparison_files(paths: list[str | Path]) -> list[dict[str, Any]]:
    """Reject a direct comparison unless immutable workload fields match."""

    if len(paths) < 2:
        raise ValueError("comparison validation requires at least two result files")
    results = [validate_result_file(path) for path in paths]
    expected = results[0]
    fields = (
        ("model", "sha256"),
        ("run", "dataset"),
        ("run", "batch_size"),
        ("run", "retained_characters"),
        ("run", "full_length"),
        ("run", "seed"),
    )
    for result in results[1:]:
        for section, field in fields:
            if result[section][field] != expected[section][field]:
                raise ValueError(
                    f"non-comparable result: {section}.{field} differs "
                    f"({expected[section][field]!r} != {result[section][field]!r})"
                )
    return results
