"""Implementation-agnostic executable-stage harness."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence


INITIALIZATION_STAGES = (
    "client_key_generation",
    "server_upload_evaluation_key",
    "server_preprocess_model",
)
INFERENCE_STAGES = (
    "client_preprocess_input",
    "client_encrypt_input",
    "server_encrypted_compute",
    "client_decrypt_output",
    "client_postprocess",
)
ROTATION_STAGES = (
    "client_key_rotation",
    "server_upload_rotated_evaluation_key",
)
REQUIRED_STAGES = (
    "client_key_generation",
    "server_preprocess_model",
    *INFERENCE_STAGES,
)


@dataclass(frozen=True, slots=True)
class SubmissionManifest:
    name: str
    implementation: str
    scheme: str
    claimed_security_bits: int
    parameters: dict[str, object]
    commands: dict[str, tuple[str, ...]]
    key_policy: str = "reused"
    is_fhe: bool = True
    additional_leakage: tuple[str, ...] = ()

    @classmethod
    def load(cls, path: str | Path) -> "SubmissionManifest":
        source = Path(path)
        material = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(material, dict):
            raise ValueError("submission manifest must be a JSON object")
        commands_value = material.get("commands")
        if not isinstance(commands_value, dict):
            raise ValueError("submission manifest requires a commands object")
        commands: dict[str, tuple[str, ...]] = {}
        for stage, command in commands_value.items():
            if not isinstance(command, list) or not command or not all(
                isinstance(argument, str) and argument for argument in command
            ):
                raise ValueError(f"command for {stage!r} must be a non-empty string array")
            commands[str(stage)] = tuple(command)
        missing = sorted(set(REQUIRED_STAGES) - commands.keys())
        if missing:
            raise ValueError(f"submission manifest is missing required stages: {missing}")
        allowed = set(INITIALIZATION_STAGES) | set(INFERENCE_STAGES) | set(ROTATION_STAGES)
        unknown = sorted(commands.keys() - allowed)
        if unknown:
            raise ValueError(f"submission manifest contains unknown stages: {unknown}")

        is_fhe = bool(material.get("is_fhe", True))
        security_bits = material.get("claimed_security_bits")
        if not isinstance(security_bits, int) or security_bits < 0:
            raise ValueError("claimed_security_bits must be a non-negative integer")
        if is_fhe and security_bits < 128:
            raise ValueError("conforming FHE submissions must claim at least 128-bit security")
        key_policy = material.get("key_policy", "reused")
        if key_policy not in ("reused", "ephemeral_per_batch"):
            raise ValueError("key_policy must be 'reused' or 'ephemeral_per_batch'")
        if key_policy == "ephemeral_per_batch":
            missing_rotation = sorted(set(ROTATION_STAGES) - commands.keys())
            if missing_rotation:
                raise ValueError(
                    "ephemeral_per_batch submissions require rotation stages: "
                    f"{missing_rotation}"
                )
        parameters = material.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("parameters must be a JSON object")
        leakage = material.get("additional_leakage", [])
        if not isinstance(leakage, list) or not all(isinstance(item, str) for item in leakage):
            raise ValueError("additional_leakage must be a string array")
        required_text = ("name", "implementation", "scheme")
        for field_name in required_text:
            if not isinstance(material.get(field_name), str) or not material[field_name].strip():
                raise ValueError(f"submission manifest requires non-empty {field_name}")
        return cls(
            name=material["name"],
            implementation=material["implementation"],
            scheme=material["scheme"],
            claimed_security_bits=security_bits,
            parameters=parameters,
            commands=commands,
            key_policy=key_policy,
            is_fhe=is_fhe,
            additional_leakage=tuple(leakage),
        )


@dataclass(frozen=True, slots=True)
class StageMeasurement:
    stage: str
    wall_seconds: float
    peak_ram_bytes: int
    detail_seconds: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StageRun:
    measurements: tuple[StageMeasurement, ...]

    @property
    def total_seconds(self) -> float:
        return sum(measurement.wall_seconds for measurement in self.measurements)

    @property
    def peak_ram_bytes(self) -> int:
        return max((measurement.peak_ram_bytes for measurement in self.measurements), default=0)

    def stage_seconds(self) -> dict[str, float]:
        return {measurement.stage: measurement.wall_seconds for measurement in self.measurements}


def _expanded_command(command: Sequence[str], submission_directory: Path) -> list[str]:
    replacements = {
        "$PYTHON": sys.executable,
        "$SUBMISSION_DIR": str(submission_directory),
    }
    return [replacements.get(argument, argument) for argument in command]


def _read_stage_report(path: Path) -> tuple[int, dict[str, float]]:
    if not path.exists():
        return 0, {}
    material = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(material, dict):
        raise ValueError("stage report must be a JSON object")
    peak = material.get("peak_ram_bytes", 0)
    detail = material.get("detail_seconds", {})
    if not isinstance(peak, int) or peak < 0:
        raise ValueError("stage report peak_ram_bytes must be a non-negative integer")
    if not isinstance(detail, dict) or any(
        not isinstance(name, str) or not isinstance(value, (int, float)) or value < 0
        for name, value in detail.items()
    ):
        raise ValueError("stage report detail_seconds must map names to non-negative numbers")
    return peak, {name: float(value) for name, value in detail.items()}


def run_stage(
    stage: str,
    command: Sequence[str],
    *,
    submission_directory: str | Path,
    work_directory: str | Path,
    environment: Mapping[str, str] | None = None,
) -> StageMeasurement:
    """Execute one submission stage and independently measure wall time."""

    submission = Path(submission_directory).resolve()
    work = Path(work_directory).resolve()
    work.mkdir(parents=True, exist_ok=True)
    report = work / f"{stage}-report.json"
    report.unlink(missing_ok=True)
    stage_environment = os.environ.copy()
    if environment:
        stage_environment.update(environment)
    stage_environment.update(
        {
            "FHE_BENCH_STAGE": stage,
            "FHE_BENCH_WORK_DIR": str(work),
            "FHE_BENCH_STAGE_REPORT": str(report),
        }
    )
    start = time.perf_counter()
    completed = subprocess.run(
        _expanded_command(command, submission),
        cwd=submission,
        env=stage_environment,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - start
    (work / f"{stage}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (work / f"{stage}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"stage {stage!r} failed with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    peak, detail = _read_stage_report(report)
    return StageMeasurement(
        stage=stage,
        wall_seconds=elapsed,
        peak_ram_bytes=peak,
        detail_seconds=detail,
    )


def run_stage_sequence(
    manifest: SubmissionManifest,
    stages: Sequence[str],
    *,
    submission_directory: str | Path,
    work_directory: str | Path,
    environment: Mapping[str, str] | None = None,
) -> StageRun:
    measurements = []
    for stage in stages:
        command = manifest.commands.get(stage)
        if command is None:
            continue
        measurements.append(
            run_stage(
                stage,
                command,
                submission_directory=submission_directory,
                work_directory=work_directory,
                environment=environment,
            )
        )
    return StageRun(tuple(measurements))


def directory_size(path: str | Path) -> int:
    """Return recursive file bytes without counting directory metadata."""

    root = Path(path)
    if not root.exists():
        return 0
    if root.is_file():
        return root.stat().st_size
    return sum(item.stat().st_size for item in root.rglob("*") if item.is_file())


def artifact_sizes(work_directory: str | Path) -> dict[str, int]:
    """Measure fixed contract artifacts after a run."""

    work = Path(work_directory)
    names = (
        "public_keys",
        "evaluation_keys",
        "ciphertexts_upload",
        "server_intermediates",
        "ciphertexts_download",
        "encrypted_outputs",
        "rotated_evaluation_keys",
    )
    return {name: directory_size(work / name) for name in names}
