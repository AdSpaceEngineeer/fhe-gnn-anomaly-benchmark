"""Versioned model-bundle metadata for interchangeable trained weights."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


BUILTIN_MODEL_ADAPTER = "polynomial_message_passing_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class ModelBundle:
    name: str
    adapter: str
    artifact_path: Path
    artifact_sha256: str
    manifest_path: Path | None = None
    recommended_baseline: bool = False

    @classmethod
    def load(cls, path: str | Path) -> "ModelBundle":
        """Load a bundle manifest or wrap a legacy ``.npz`` artifact."""

        source = Path(path).resolve()
        if source.is_dir():
            source = source / "model.json"
        if source.suffix.lower() != ".json":
            if not source.is_file():
                raise FileNotFoundError(source)
            return cls(
                name=source.stem,
                adapter=BUILTIN_MODEL_ADAPTER,
                artifact_path=source,
                artifact_sha256=_sha256(source),
            )

        material = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(material, dict) or material.get("schema_version") != "0.1.0":
            raise ValueError("model bundle requires schema_version 0.1.0")
        name = material.get("name")
        adapter = material.get("adapter")
        artifact = material.get("artifact")
        expected_hash = material.get("artifact_sha256")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("model bundle requires a non-empty name")
        if adapter != BUILTIN_MODEL_ADAPTER:
            raise ValueError(f"unsupported model adapter: {adapter!r}")
        if not isinstance(artifact, str) or not artifact:
            raise ValueError("model bundle requires an artifact path")
        artifact_path = (source.parent / artifact).resolve()
        if not artifact_path.is_relative_to(source.parent.resolve()):
            raise ValueError("model artifact must remain inside its bundle directory")
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        observed_hash = _sha256(artifact_path)
        if expected_hash != observed_hash:
            raise ValueError("model artifact checksum does not match model bundle")
        recommended = material.get("recommended_baseline", False)
        if not isinstance(recommended, bool):
            raise ValueError("recommended_baseline must be boolean")
        return cls(
            name=name,
            adapter=adapter,
            artifact_path=artifact_path,
            artifact_sha256=observed_hash,
            manifest_path=source,
            recommended_baseline=recommended,
        )


def write_model_bundle(
    model_path: str | Path,
    manifest_path: str | Path,
    *,
    name: str,
    recommended_baseline: bool = False,
) -> Path:
    """Write a bundle for compatible trained weights."""

    model = Path(model_path).resolve()
    destination = Path(manifest_path).resolve()
    if model.parent != destination.parent:
        raise ValueError("model artifact and manifest must share a directory")
    material = {
        "schema_version": "0.1.0",
        "name": name,
        "adapter": BUILTIN_MODEL_ADAPTER,
        "artifact": model.name,
        "artifact_sha256": _sha256(model),
        "recommended_baseline": recommended_baseline,
    }
    destination.write_text(
        json.dumps(material, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination
