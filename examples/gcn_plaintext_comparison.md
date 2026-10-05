# Benchmark comparison

Submission: `plaintext_debug` · Workload: `scam-list-gcn-relu-100k-v2`
Status: **passed** · Security: `not_fhe` · Eligible FHE comparison: **False**

Completed runs: 1. Quality uses the fixed test split and threshold.

| Metric | Frozen plaintext | Submission |
|---|---:|---:|
| Accuracy (primary) | 0.9934 | 0.9934 |
| ROC-AUC (primary) | 0.998725 | 0.998725 |
| Recall | 0.902709 | 0.902709 |
| F1 | 0.917397 | 0.917397 |
| Precision | 0.93257 | 0.93257 |
| Average precision | 0.971232 | 0.971232 |
| Maximum score error (worst run) | 0 | 0 |
| Prediction agreement | 1 | 1 |
| Requested threads | — | 2 |
| Reported evaluator compute threads | — | 2 |
| Sampled evaluator OS threads (peak) | — | 12 |
| Key generation (s, once) | — | 0.589744 |
| Encrypt wall time (s) | — | 0.898057 |
| Evaluate wall time (s) | — | 2.02272 |
| Decrypt wall time (s) | — | 0.781474 |
| Inference wall time (s) | — | 3.70225 |
| Evaluator throughput (nodes/s) | — | 49438.5 |
| Peak stage RAM (MiB, maximum) | — | 369.723 |
| Public/evaluation keys (MiB) | — | 0 |
| Input payload (MiB) | — | 6.22645 |
| Output payload (MiB) | — | 1.97091 |
| Persisted intermediates (MiB) | — | 0 |
| Communication (MiB/run, amortized) | — | 58.2711 |

Values are means across completed runs unless labelled otherwise. MiB = 2^20 bytes.
A dash means not measured, not zero. The frozen reference supplies quality, not matched plaintext timing.
Server-reported timings are additional detail; they never replace or subtract from harness wall times.
Key/public-workload uploads are counted once and amortized; no network-transfer duration is measured.

**Plaintext debug run: stage/payload names do not imply encryption. These are not FHE overhead results.**
