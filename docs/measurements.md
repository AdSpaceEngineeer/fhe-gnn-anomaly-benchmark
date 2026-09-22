# Measurement definitions

| Quantity | Harness measurement |
|---|---|
| Stage latency | `perf_counter` wall time of a fresh worker, including imports, serialization and local file I/O |
| Operation + I/O latency | Worker timer after adapter import; context loading, encoding/decoding and file writes included |
| Inference latency | Sum of encryption, evaluator and decryption worker times; excludes key generation and validation |
| Throughput | All graph nodes scored / evaluator wall time; end-to-end throughput also reported |
| Peak memory | OS process-lifetime high-water RSS and 10ms samples of aggregate worker/descendant RSS, bytes |
| Key storage | Exact serialized public/evaluation bundle and separate client-private bundle sizes |
| Ciphertext storage | Exact input/output serialized bytes |
| Intermediate storage | Files retained in `intermediate_dir`; in-memory temporaries are covered by RAM measurements |
| Communication | Client ciphertext upload, result download, one-time key/public-workload bytes and amortized key bytes |
| Quality | Recall/F1 (primary), Accuracy, Precision, ROC-AUC/AP; fixed test split and validation threshold |
| Numerical fidelity | Absolute/relative tolerance check over all scores, mean/max error, prediction agreement |

No actual network is used; network latency and key upload duration are `null`,
not fabricated from local reads. Rotation timing is `null` under the fixed key
policy. Public/evaluation key sizes are combined when the backend serializes them
as one context. OS high-water values cover the whole stage process; sampled
process-tree values can miss very short peaks and double-count shared pages.
GPU/device memory is not included in RSS and needs separate adapter reporting.
Thread settings are requests; a custom/native backend must honor them. Coordinate
large jobs on shared systems and follow local allocation rules.

The first run is cold and repeats include context loads; no warmup is silently
discarded. This is a correctness-stage, file-based baseline, not an optimized
transport or kernel-only performance measurement. The tiny default does not
establish full dataset scaling. A single graph run scores a batch of event nodes;
per-node latency is not separately measured.

Exit codes: `0` numerical verification passed; `2` valid scores exceeded the
tolerance; `1` installation, security, artifact, timeout or execution error.
Security status and artifact registration are separate from numerical PASS.
`eligible_for_comparison` is scoped to identical artifact manifests and requires
all three. The plaintext debug adapter is never an FHE comparison.

`report.json` is the shareable result. `io/` contains client secret keys and test
plaintext; never submit it. Default output uses a fresh directory, and an existing
`--out` is rejected. Reports do not collect hardware identifiers by default;
`--include-hardware` adds an OS/CPU summary. Without comparable hardware/runtime
conditions, do not interpret timing differences as scheme superiority.
