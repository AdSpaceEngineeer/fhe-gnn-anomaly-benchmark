#!/usr/bin/env python
"""One-command inference benchmark. Run from any working directory."""
import argparse
import os
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.params import ROOT, DEFAULT_THREADS, DEFAULT_ARTIFACTS, KEY_POLICY
from harness.utils import read_json, write_json, sha256, directory_bytes, run_measured, set_threads
from harness.reporting import SERVER_TIMINGS, read_server_timings, write_comparison


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--submission", default="toy_ckks", help="Name in submissions/ or path to adapter.py")
    p.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS), help="Maintainer override; only the published frozen workload is registered")
    p.add_argument("--num-runs", type=int, default=1)
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    p.add_argument("--timeout-seconds", type=float, default=86400, help="Maximum wall seconds per stage; default 24 hours")
    p.add_argument("--out", help="New output directory; existing directories are never overwritten")
    p.add_argument("--debug-plaintext", action="store_true", help="Allow non-FHE adapter for harness testing")
    p.add_argument("--include-hardware", action="store_true", help="Opt in to OS/CPU summary in report")
    args = p.parse_args(argv)
    if args.num_runs < 1 or args.threads < 1 or args.timeout_seconds <= 0:
        p.error("Run count, threads and timeout must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    set_threads(args.threads)
    import numpy as np
    from harness.generate_input import load_bundle
    from harness.security import validate_description
    from harness.verify_result import verify
    adapter = Path(args.submission)
    if not adapter.is_file():
        adapter = ROOT / "submissions" / args.submission / "adapter.py"
    adapter = adapter.resolve()
    if not adapter.is_file():
        raise ValueError("Submission not found: " + str(adapter))
    bundle = load_bundle(args.artifacts)
    print("[harness] Workload: %s (%d events); weights and graph supplied automatically" %
          (bundle["manifest"]["id"], bundle["public"]["node_count"]), flush=True)
    out = Path(args.out) if args.out else ROOT / "measurements" / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    io = out / "io"
    io.mkdir()
    report = {"format_version": 1, "status": "running", "submission": adapter.parent.name,
              "adapter_sha256": sha256(adapter), "artifact_id": bundle["manifest"]["id"],
              "artifact_manifest_sha256": bundle["manifest_sha256"],
              "registered_artifacts": bundle["registered"], "purpose": bundle["manifest"]["purpose"],
              "threads_requested": args.threads, "key_policy": KEY_POLICY, "runs": [],
              "memory_method": "stage process lifetime high-water RSS plus 10ms sampled process-tree RSS",
              "communication_method": "serialized payload bytes; no network transport timing"}
    report["harness_sha256"] = {p.name: sha256(p) for p in sorted((ROOT / "harness").glob("*.py"))}
    report["submission_source_sha256"] = {p.relative_to(adapter.parent).as_posix(): sha256(p)
                                          for p in sorted(adapter.parent.rglob("*"))
                                          if p.is_file() and p.suffix in (".py", ".md", ".txt", ".cpp", ".h", ".json")
                                          and "__pycache__" not in p.parts}
    if args.include_hardware:
        import platform
        report["hardware"] = {"os": platform.system(), "machine": platform.machine(),
                              "processor": platform.processor(), "logical_cpus": os.cpu_count()}
    # Absolute import path allows stage workers to execute outside the repository.
    os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
    def stage(name, location):
        print("[harness] " + name, flush=True)
        measured = run_measured([sys.executable, "-m", "harness.worker", name,
                                 "--adapter", str(adapter), "--io", str(location),
                                 "--threads", str(args.threads)],
                                location / (name + ".log"), args.timeout_seconds)
        measured.update(read_json(location / (name + "_measurement.json")))
        return measured
    try:
        stage("describe", io)
        description = read_json(io / "description.json")
        report["description"] = description
        report["security"] = validate_description(description, args.debug_plaintext)
        report["keygen"] = stage("keygen", io)
        report["security_check"] = stage("check_context", io)
        context_check = read_json(io / "context_check.json")
        if context_check is not None:
            report["security"] = context_check
        public_key_bytes = directory_bytes(io / "server" / "keys")
        private_key_bytes = directory_bytes(io / "client" / "keys")
        import shutil
        for index in range(args.num_runs):
            trial = io / ("run-%03d" % index)
            trial.mkdir()
            # Physical copies are local bookkeeping, not additional network uploads.
            shutil.copytree(io / "client" / "keys", trial / "client" / "keys")
            shutil.copytree(io / "server" / "keys", trial / "server" / "keys")
            write_json(trial / "client" / "input.json", bundle["sensitive"])
            write_json(trial / "server" / "public.json", bundle["public"])
            (trial / "server" / "intermediate").mkdir()
            times = {name: stage(name, trial) for name in ("encrypt", "evaluate", "decrypt")}
            server_timings, timing_warnings = read_server_timings(trial / "server" / "intermediate")
            for warning in timing_warnings:
                print("[harness] WARNING: " + warning, flush=True)
            scores = read_json(trial / "client" / "scores.json")
            m, d, r = bundle["manifest"], bundle["data"], bundle["reference"]
            checked = verify(scores, r["scores"], d["labels"], d["splits"]["test"],
                             r["threshold"], m["atol"], m["rtol"])
            encrypted_in = directory_bytes(trial / "server" / "input")
            encrypted_out = directory_bytes(trial / "server" / "output")
            public_bytes = (trial / "server" / "public.json").stat().st_size
            evaluation_seconds = times["evaluate"]["wall_seconds"]
            total_seconds = sum(v["wall_seconds"] for v in times.values())
            run = {"index": index, "stages": times, "verification": checked,
                   "server_reported_steps": server_timings,
                   "server_timing_warnings": timing_warnings,
                   "num_nodes_computed": d["features"].__len__(), "num_test_nodes": len(d["splits"]["test"]),
                   "inference_wall_seconds": total_seconds,
                   "throughput_nodes_per_second": len(scores) / evaluation_seconds,
                   "end_to_end_nodes_per_second": len(scores) / total_seconds,
                   "storage_bytes": {"public_and_evaluation_keys": public_key_bytes,
                                     "client_private_keys": private_key_bytes,
                                     "encrypted_input": encrypted_in, "encrypted_output": encrypted_out,
                                     "persisted_intermediates": sum(p.stat().st_size for p in (trial / "server" / "intermediate").rglob("*")
                                                                    if p.is_file() and p != trial / "server" / "intermediate" / SERVER_TIMINGS)},
                   "communication_bytes": {"client_to_server_input": encrypted_in,
                                           "server_to_client_result": encrypted_out,
                                           "public_workload_upload_once": public_bytes if index == 0 else 0,
                                           "key_upload_once": public_key_bytes if index == 0 else 0,
                                           "key_upload_amortized": public_key_bytes / args.num_runs},
                   "amortized_keygen_seconds": report["keygen"]["wall_seconds"] / args.num_runs}
            report["runs"].append(run)
            write_json(out / "report.json", report)
        report["status"] = "passed" if all(r["verification"]["passed"] for r in report["runs"]) else "failed_verification"
        report["eligible_for_comparison"] = bool(report["security"]["eligible"] and bundle["registered"] and report["status"] == "passed")
        report["timing_summary"] = {"mean_inference_wall_seconds": float(np.mean([r["inference_wall_seconds"] for r in report["runs"]])),
                                  "repeat_count": args.num_runs}
        report["network_transfer_seconds"] = None
        report["key_rotation_seconds"] = None
        report["comparison_scope"] = "Only the identical registered artifact ID and manifest; internal fixtures are not benchmark results"
        write_json(out / "report.json", report)
        write_comparison(out / "comparison.md", report)
        print("[harness] %s: %s" % (report["status"], out / "report.json"))
        print("[harness] Comparison table: " + str(out / "comparison.md"))
        print("[harness] Output contains client secret keys and plaintext test inputs; share report.json and comparison.md only.")
        return 0 if report["status"] == "passed" else 2
    except Exception as exc:
        report.update(status="error", error=str(exc), eligible_for_comparison=False)
        write_json(out / "report.json", report)
        write_comparison(out / "comparison.md", report)
        raise


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("[harness] ERROR: " + str(exc), file=sys.stderr)
        sys.exit(1)
