"""Internal stage process. Submitters implement Adapter, not this module."""
import argparse
import sys
import time
from pathlib import Path
import psutil
from harness.utils import load_adapter, read_json, write_json, read_blobs, write_blobs, set_threads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("describe", "keygen", "check_context", "encrypt", "evaluate", "decrypt"))
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--io", required=True)
    parser.add_argument("--threads", type=int, required=True)
    args = parser.parse_args()
    set_threads(args.threads)
    root = Path(args.io)
    adapter = load_adapter(args.adapter)
    start = time.perf_counter()
    if args.stage == "describe":
        write_json(root / "description.json", adapter.describe())
    elif args.stage == "keygen":
        private, public = adapter.keygen(args.threads)
        write_blobs(root / "client" / "keys", private)
        write_blobs(root / "server" / "keys", public)
    elif args.stage == "check_context":
        from harness.security import inspect_public_context
        check = inspect_public_context(adapter.describe(), read_blobs(root / "server" / "keys"), args.threads)
        write_json(root / "context_check.json", check)
    elif args.stage == "encrypt":
        encrypted = adapter.encrypt(read_json(root / "client" / "input.json"),
                                    read_blobs(root / "client" / "keys"), args.threads)
        write_blobs(root / "server" / "input", encrypted)
    elif args.stage == "evaluate":
        result = adapter.evaluate(read_blobs(root / "server" / "input"),
                                  read_json(root / "server" / "public.json"),
                                  read_blobs(root / "server" / "keys"), args.threads,
                                  root / "server" / "intermediate")
        write_blobs(root / "server" / "output", result)
    else:
        scores = adapter.decrypt(read_blobs(root / "server" / "output"),
                                 read_blobs(root / "client" / "keys"), args.threads)
        write_json(root / "client" / "scores.json", [float(s) for s in scores])
    seconds = time.perf_counter() - start
    if sys.platform == "win32":
        peak = psutil.Process().memory_info().peak_wset
    else:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            peak *= 1024
    write_json(root / (args.stage + "_measurement.json"),
               {"operation_and_io_seconds": seconds, "process_lifetime_peak_rss_bytes": peak})


if __name__ == "__main__":
    main()
