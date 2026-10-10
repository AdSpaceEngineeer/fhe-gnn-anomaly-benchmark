"""Serialization, hashing and process measurement helpers."""
import gzip
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


def open_artifact(path):
    """Read exact logical bytes from a plain file or its lossless .gz version.

    Manifests identify uncompressed content, independent of storage encoding.
    Reject ambiguous copies rather than silently choosing one of them.
    """
    path = Path(path)
    compressed = path.with_name(path.name + ".gz")
    if path.exists() and compressed.exists():
        raise ValueError("Ambiguous plain and compressed artifact: " + path.name)
    if path.exists():
        return path.open("rb")
    if compressed.exists():
        return gzip.open(compressed, "rb")
    raise FileNotFoundError("Missing artifact: " + str(path) + " (or .gz)")


def artifact_sha256(path):
    digest = hashlib.sha256()
    with open_artifact(path) as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_artifact_json(path):
    with open_artifact(path) as handle:
        return json.load(handle)


def load_adapter(path):
    path = Path(path).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("submission_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    adapter = module.Adapter()
    for name in ("configure", "describe", "thread_report", "keygen", "encrypt", "evaluate", "decrypt"):
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


def sample_process_tree(process):
    """Observed RSS/threads and whether the RSS snapshot was complete."""
    complete = True
    try:
        family = [process] + process.children(recursive=True)
    except psutil.Error:
        family, complete = [process], False
    rss, threads = [], []
    for child in family:
        try:
            value = child.memory_info().rss
            if value > 0:
                rss.append(value)
            else:
                complete = False
        except psutil.Error:
            complete = False
        try:
            threads.append(child.num_threads())
        except psutil.Error:
            pass
    return (sum(rss) if rss else None, sum(threads) if threads else None, complete)


def memory_sampling_summary(peak, attempts, observed, incomplete):
    return dict(sampled_process_tree_peak_rss_bytes=peak if observed else None,
                memory_sampling_status=('unavailable' if not observed else
                                        'partial' if incomplete else 'available'),
                memory_sample_attempts=attempts, memory_samples_observed=observed,
                memory_samples_incomplete=incomplete,
                memory_sampling_provenance='measured')


def run_measured(command, log, timeout, interval=0.01):
    """Sample aggregate process-tree RSS, including child native-library workers."""
    peak = None
    peak_threads = 0
    attempts = observed = incomplete = 0
    start = time.perf_counter()
    with Path(log).open("w", encoding="utf-8") as output:
        proc = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        process = psutil.Process(proc.pid)
        try:
            while proc.poll() is None:
                rss, threads, complete = sample_process_tree(process)
                attempts += 1
                if rss is not None:
                    observed += 1
                    peak = rss if peak is None else max(peak, rss)
                if not complete or rss is None:
                    incomplete += 1
                if threads is not None:
                    peak_threads = max(peak_threads, threads)
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
    return {"wall_seconds": elapsed, **memory_sampling_summary(peak, attempts, observed, incomplete),
            "sampled_process_tree_peak_os_threads": peak_threads,
            "memory_sample_interval_seconds": interval}
