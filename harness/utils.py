"""Serialization, hashing and process measurement helpers."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_adapter(path):
    path = Path(path).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("submission_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    adapter = module.Adapter()
    for name in ("describe", "keygen", "encrypt", "evaluate", "decrypt"):
        if not callable(getattr(adapter, name, None)):
            raise ValueError("Adapter missing method: " + name)
    return adapter


def write_blobs(directory, blobs):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not isinstance(blobs, dict):
        raise TypeError("Adapter payload must be a dict of filename: bytes")
    for name, value in blobs.items():
        if not isinstance(name, str) or not name or name in (".", "..") or Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("Payload names must be simple filenames")
        if not isinstance(value, bytes):
            raise TypeError("Payload values must be bytes")
        (directory / name).write_bytes(value)


def read_blobs(directory):
    return {p.name: p.read_bytes() for p in sorted(Path(directory).iterdir()) if p.is_file()}


def directory_bytes(directory):
    return sum(p.stat().st_size for p in Path(directory).rglob("*") if p.is_file())


def set_threads(threads):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)


def run_measured(command, log, timeout, interval=0.01):
    """Sample aggregate process-tree RSS, including child native-library workers."""
    peak = 0
    start = time.perf_counter()
    with Path(log).open("w", encoding="utf-8") as output:
        proc = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        process = psutil.Process(proc.pid)
        try:
            while proc.poll() is None:
                try:
                    family = [process] + process.children(recursive=True)
                    rss = 0
                    for child in family:
                        try:
                            rss += child.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                    peak = max(peak, rss)
                except psutil.NoSuchProcess:
                    pass
                if time.perf_counter() - start > timeout:
                    raise TimeoutError("Stage exceeded --timeout-seconds; see " + str(log))
                time.sleep(interval)
        finally:
            if proc.poll() is None:
                try:
                    for child in process.children(recursive=True):
                        child.kill()
                except psutil.NoSuchProcess:
                    pass
                proc.kill()
                proc.wait()
    elapsed = time.perf_counter() - start
    if proc.returncode:
        tail = Path(log).read_text(encoding="utf-8")[-3000:]
        raise RuntimeError("Stage failed (exit %s):\n%s" % (proc.returncode, tail))
    return {"wall_seconds": elapsed, "sampled_process_tree_peak_rss_bytes": peak,
            "memory_sample_interval_seconds": interval}
