# Benchmark comparison

Submission: `plaintext_debug` · Workload: `scam-list-gcn-100k-v1`
Status: **passed** · Security: `not_fhe` · Eligible FHE comparison: **False**

Completed runs: 1. Quality uses the fixed test split and threshold.

| Metric | Frozen plaintext | Submission |
|---|---:|---:|
| Recall | 0.897783 | 0.897783 |
| F1 | 0.915254 | 0.915254 |
| Accuracy | 0.99325 | 0.99325 |
| Maximum score error (worst run) | 0 | 0 |
| Prediction agreement | 1 | 1 |
| Key generation (s, once) | — | 0.788915 |
| Encrypt wall time (s) | — | 1.13328 |
| Evaluate wall time (s) | — | 2.52215 |
| Decrypt wall time (s) | — | 0.961982 |
| Inference wall time (s) | — | 4.61741 |
| Evaluator throughput (nodes/s) | — | 39648.8 |
| Peak stage RAM (MiB, maximum) | — | 394.254 |
| Public/evaluation keys (MiB) | — | 0 |
| Input payload (MiB) | — | 6.22645 |
| Output payload (MiB) | — | 1.96502 |
| Persisted intermediates (MiB) | — | 0 |
| Communication (MiB/run, amortized) | — | 58.2632 |

Values are means across completed runs unless labelled otherwise. MiB = 2^20 bytes.
A dash means not measured, not zero. The frozen reference supplies quality, not matched plaintext timing.
Server-reported timings are additional detail; they never replace or subtract from harness wall times.
Key/public-workload uploads are counted once and amortized; no network-transfer duration is measured.

**Plaintext debug run: stage/payload names do not imply encryption. These are not FHE overhead results.**
