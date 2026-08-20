"""Canonical identifier-retention sweep and compact SVG reporting."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Sequence

from .benchmark import run_benchmark
from .identifiers import CANONICAL_RETENTIONS


def _write_q_svg(points: list[dict[str, float | int]], path: Path) -> None:
    width, height = 720, 440
    left, right, top, bottom = 80, 30, 40, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_values = [float(point["retention_realized"]) for point in points]
    y_values = [float(point["q_median"]) for point in points]
    y_max = max(1.0, max(y_values, default=1.0))

    def x_position(value: float) -> float:
        return left + value * plot_width

    def y_position(value: float) -> float:
        return top + (1 - value / y_max) * plot_height

    polyline = " ".join(
        f"{x_position(x):.2f},{y_position(y):.2f}" for x, y in zip(x_values, y_values)
    )
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="360" y="24" text-anchor="middle" font-family="sans-serif" font-size="18">Identifier Truncation Robustness Q(k)</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>',
    ]
    for tick in range(6):
        x_value = tick / 5
        x = x_position(x_value)
        elements.append(
            f'<text x="{x:.2f}" y="{height-bottom+24}" text-anchor="middle" font-family="sans-serif" font-size="12">{x_value:.1f}</text>'
        )
        y_value = y_max * tick / 5
        y = y_position(y_value)
        elements.append(
            f'<text x="{left-12}" y="{y+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{y_value:.2f}</text>'
        )
    if polyline:
        elements.append(
            f'<polyline points="{polyline}" fill="none" stroke="#2563eb" stroke-width="3"/>'
        )
    for point, x_value, y_value in zip(points, x_values, y_values):
        x, y = x_position(x_value), y_position(y_value)
        elements.extend(
            (
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#dc2626"/>',
                f'<text x="{x:.2f}" y="{y-10:.2f}" text-anchor="middle" font-family="sans-serif" font-size="11">k={int(point["retained_characters"])}; Q={y_value:.3f}</text>',
            )
        )
    elements.extend(
        (
            f'<text x="{left+plot_width/2:.2f}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="14">Realized retention k/L</text>',
            f'<text x="20" y="{top+plot_height/2:.2f}" transform="rotate(-90 20 {top+plot_height/2:.2f})" text-anchor="middle" font-family="sans-serif" font-size="14">Q(k)</text>',
            "</svg>",
        )
    )
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def run_retention_sweep(
    submission_manifest_path: str | Path,
    dataset_directory: str | Path,
    model_path: str | Path,
    output_directory: str | Path,
    *,
    batch_size: int,
    num_runs: int = 3,
    seed: int = 2026,
    retentions: Sequence[float] = CANONICAL_RETENTIONS,
) -> tuple[Path, Path]:
    """Run a declared retention set and write summary JSON plus Q(k) SVG."""

    if not retentions:
        raise ValueError("retention sweep must contain at least one level")
    output = Path(output_directory).resolve()
    rows: list[dict[str, float | int]] = []
    seen_k: set[int] = set()
    for retention in retentions:
        paths = run_benchmark(
            submission_manifest_path,
            dataset_directory,
            model_path,
            output,
            retention=float(retention),
            batch_size=batch_size,
            num_runs=num_runs,
            seed=seed,
        )
        results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        first = results[0]
        retained = int(first["run"]["retained_characters"])
        if retained in seen_k:
            continue
        seen_k.add(retained)
        q_values = [float(result["quality"]["q"]) for result in results]
        rows.append(
            {
                "retention_requested": float(retention),
                "retained_characters": retained,
                "full_length": int(first["run"]["full_length"]),
                "retention_realized": float(first["run"]["retention_realized"]),
                "q_median": statistics.median(q_values),
                "latency_seconds_median": statistics.median(
                    float(result["performance"]["online_latency_seconds"])
                    for result in results
                ),
                "storage_bytes_median": statistics.median(
                    int(result["performance"]["storage_bytes"]) for result in results
                ),
                "communication_bytes_median": statistics.median(
                    int(result["performance"]["communication_bytes"])
                    for result in results
                ),
            }
        )
    rows.sort(key=lambda row: int(row["retained_characters"]))
    summary = {
        "schema_version": "0.1.0",
        "retention_policy": "canonical" if tuple(retentions) == CANONICAL_RETENTIONS else "custom",
        "batch_size": batch_size,
        "num_runs": num_runs,
        "points": rows,
    }
    summary_path = output / "sweep-summary.json"
    plot_path = output / "q-vs-retention.svg"
    output.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_q_svg(rows, plot_path)
    return summary_path, plot_path
