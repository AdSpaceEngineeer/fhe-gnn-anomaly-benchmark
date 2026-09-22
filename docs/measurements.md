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
transport or kernel-only performance measurement. The default is the fixed trained
100,000-event workload; internal fixtures do not establish its FHE performance.
A single graph run scores a batch of event nodes;
per-node latency is not separately measured.

Exit codes: `0` numerical verification passed; `2` valid scores exceeded the
tolerance; `1` installation, security, artifact, timeout or execution error.
Security status and artifact registration are separate from numerical PASS.
`eligible_for_comparison` is scoped to identical artifact manifests and requires
all three. The plaintext debug adapter is never an FHE comparison.

`report.json` and `comparison.md` are the shareable results. `io/` contains client secret keys and test
plaintext; never submit it. Default output uses a fresh directory, and an existing
`--out` is rejected. Reports do not collect hardware identifiers by default;
`--include-hardware` adds an OS/CPU summary. Without comparable hardware/runtime
conditions, do not interpret timing differences as scheme superiority.

## Optional server-reported timings

Following [BERT's server reporting convention](https://github.com/fhe-benchmarking/BERT/blob/170bfe567545d74d0fad785a052518351d37bc93/submissions/server_encrypted_compute.py),
an adapter may write `intermediate_dir/server_reported_steps.json`, a flat JSON
object mapping step names to finite nonnegative seconds, for example:

```json
{"Encrypted computation": 12.3, "I/O": 1.4, "Total": 13.7}
```

The harness stores these separately in each run's `server_reported_steps`; it
never substitutes or subtracts them from independently measured stage times.
Names can describe other useful components. Document whether they overlap and
what `Total` includes. Missing files are optional; malformed, oversized (>64 KiB),
duplicate-key or nonnumeric/negative/nonfinite reports are ignored with warnings.
This does not change numerical verification or the main measurements.

The evaluator worker also records `harness_file_io_seconds` for reading its inputs
and writing returned outputs, and `adapter_call_seconds` for the adapter call.
Adapter-reported I/O covers only the adapter's declared scope; these fields are
not assumed to equal the worker wall time. The timing file is excluded from
persisted intermediate-value bytes, because it is reporting metadata.

## Compact companion table

`comparison.md` compares fixed-reference Recall/F1/Accuracy with the submission,
then displays main overhead and optional timing detail. It averages completed
repeats, labels worst-case score error/maximum RAM, and counts each one-time
key/public-workload upload once when amortizing communication. Missing timings
appear as a dash, not zero. No matched plaintext timing is invented. Debug runs
are labelled non-FHE; failed runs without completed results get no fabricated table.
