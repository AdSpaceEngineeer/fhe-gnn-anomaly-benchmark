"""Optional BERT-style server timings and a readable companion to report.json."""
import json
import math
from pathlib import Path
from statistics import mean

SERVER_TIMINGS = "server_reported_steps.json"


def read_server_timings(directory):
    """Untrusted, optional observations; never replace harness measurements."""
    path = Path(directory) / SERVER_TIMINGS
    if not path.exists():
        return None, []
    try:
        if not path.is_file() or path.stat().st_size > 65536:
            raise ValueError("expected a JSON file no larger than 64 KiB")
        def no_duplicates(pairs):
            result = {}
            for name, value in pairs:
                if name in result:
                    raise ValueError("duplicate timing name")
                result[name] = value
            return result
        values = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)
        if not isinstance(values, dict) or not 1 <= len(values) <= 64:
            raise ValueError("expected 1 to 64 named timings")
        for name, value in values.items():
            if not isinstance(name, str) or not name.strip() or len(name) > 120 or any(ord(c) < 32 for c in name):
                raise ValueError("invalid timing name")
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("timings must be finite nonnegative seconds, not strings or booleans")
        return values, []
    except (ValueError, OSError, UnicodeError) as exc:
        return None, ["Ignored optional server timings: " + str(exc)]


def _number(value):
    return "—" if value is None else f"{value:.6g}"


def _escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def comparison_markdown(report):
    runs = report.get("runs", [])
    name = _escape(report.get("submission", "unknown"))
    lines = ["# Benchmark comparison", "",
             f"Submission: `{name}` · Workload: `{_escape(report.get('artifact_id', 'unknown'))}`",
             f"Status: **{_escape(report.get('status', 'unknown'))}** · "
             f"Security: `{_escape(report.get('security', {}).get('status', 'not checked'))}` · "
             f"Eligible FHE comparison: **{bool(report.get('eligible_for_comparison', False))}**", ""]
    if not runs:
        lines += ["No completed inference runs; no performance or quality comparison is available."]
        return "\n".join(lines) + "\n"
    lines += [f"Completed runs: {len(runs)}. Quality uses the fixed test split and threshold.", "",
              "| Metric | Frozen plaintext | Submission |", "|---|---:|---:|"]
    def row(label, reference, submitted):
        lines.append(f"| {label} | {_number(reference)} | {_number(submitted)} |")
    for label, key in (("Recall", "recall"), ("F1", "f1"), ("Accuracy", "accuracy")):
        row(label, runs[0]["verification"]["plaintext_quality"][key],
            mean(r["verification"]["quality"][key] for r in runs))
    row("Maximum score error (worst run)", 0, max(r["verification"]["max_absolute_error"] for r in runs))
    row("Prediction agreement", 1, mean(r["verification"]["prediction_agreement"] for r in runs))
    row("Key generation (s, once)", None, report["keygen"]["wall_seconds"])
    for stage in ("encrypt", "evaluate", "decrypt"):
        row(f"{stage.capitalize()} wall time (s)", None, mean(r["stages"][stage]["wall_seconds"] for r in runs))
    row("Inference wall time (s)", None, mean(r["inference_wall_seconds"] for r in runs))
    row("Evaluator throughput (nodes/s)", None, mean(r["throughput_nodes_per_second"] for r in runs))
    measurements = [report.get("keygen", {}), report.get("security_check", {})]
    measurements += [stage for r in runs for stage in r["stages"].values()]
    peak = max(max(m.get("sampled_process_tree_peak_rss_bytes", 0), m.get("process_lifetime_peak_rss_bytes", 0)) for m in measurements)
    row("Peak stage RAM (MiB, maximum)", None, peak / 2**20)
    for label, key in (("Public/evaluation keys (MiB)", "public_and_evaluation_keys"),
                       ("Input payload (MiB)", "encrypted_input"), ("Output payload (MiB)", "encrypted_output"),
                       ("Persisted intermediates (MiB)", "persisted_intermediates")):
        row(label, None, mean(r["storage_bytes"][key] for r in runs) / 2**20)
    communications = [sum(r["communication_bytes"][key] for key in
                      ("client_to_server_input", "server_to_client_result", "public_workload_upload_once", "key_upload_once"))
                      for r in runs]
    row("Communication (MiB/run, amortized)", None, mean(communications) / 2**20)
    timing_names = sorted({name for r in runs for name in (r.get("server_reported_steps") or {})})
    if timing_names:
        lines += ["", "Optional server-reported seconds (not independently verified):", "",
                  "| Timing | Mean seconds | Runs reporting |", "|---|---:|---:|"]
        for name in timing_names:
            values = [r["server_reported_steps"][name] for r in runs if name in (r.get("server_reported_steps") or {})]
            lines.append(f"| {_escape(name)} | {_number(mean(values))} | {len(values)}/{len(runs)} |")
    lines += ["", "Values are means across completed runs unless labelled otherwise. MiB = 2^20 bytes.",
              "A dash means not measured, not zero. The frozen reference supplies quality, not matched plaintext timing.",
              "Server-reported timings are additional detail; they never replace or subtract from harness wall times.",
              "Key/public-workload uploads are counted once and amortized; no network-transfer duration is measured."]
    if not report.get("description", {}).get("is_fhe", False):
        lines += ["", "**Plaintext debug run: stage/payload names do not imply encryption. These are not FHE overhead results.**"]
    warnings = [w for r in runs for w in r.get("server_timing_warnings", [])]
    if warnings:
        lines += ["", "Warnings: " + "; ".join(_escape(w) for w in warnings)]
    return "\n".join(lines) + "\n"


def write_comparison(path, report):
    Path(path).write_text(comparison_markdown(report), encoding="utf-8", newline="\n")
