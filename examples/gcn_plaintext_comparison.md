# Benchmark comparison

Submission: `plaintext_debug` · Workload: `scam-list-gcn-relu-100k-v2`
Status: **passed** · Security: `not_fhe` · Eligible FHE comparison: **False**

Optional hardware information:

| Field | Reported value |
|---|---|
| Operating system | Not reported |
| Architecture | Not reported |
| CPU description | Not reported |
| System logical CPUs | Not reported |

Hardware disclosure is optional. System logical CPUs are not a process allocation or configured compute-thread count.

Missing values are shown as Not reported; genuine zero measurements remain 0.

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
| Requested threads | Not reported | 2 |
| Reported evaluator compute threads | Not reported | 2 |
| Sampled evaluator OS threads (peak) | Not reported | 12 |
| Key generation (s, once) | Not reported | 0.589744 |
| Encrypt wall time (s) | Not reported | 0.898057 |
| Evaluate wall time (s) | Not reported | 2.02272 |
| Decrypt wall time (s) | Not reported | 0.781474 |
| Inference wall time (s) | Not reported | 3.70225 |
| Evaluator throughput (nodes/s) | Not reported | 49438.5 |
| Peak stage RAM (MiB, maximum) | Not reported | 369.723 |
| Public/evaluation keys (MiB) | Not reported | 0 |
| Input payload (MiB) | Not reported | 6.22645 |
| Output payload (MiB) | Not reported | 1.97091 |
| Persisted intermediates (MiB) | Not reported | 0 |
| Communication (MiB/run, amortized) | Not reported | 58.2711 |

Memory sampling by stage:

| Stage | Sampling | RSS observations | Sampled peak MiB | Process high-water MiB |
|---|---|---:|---:|---:|
| description_stage | available (legacy; coverage: Not reported) | Not reported | 59.2188 | 55.043 |
| keygen | available (legacy; coverage: Not reported) | Not reported | 59.2148 | 55.1055 |
| security_check | available (legacy; coverage: Not reported) | Not reported | 59.0312 | 54.9297 |
| run 1: encrypt | available (legacy; coverage: Not reported) | Not reported | 86.0469 | 88.3438 |
| run 1: evaluate | available (legacy; coverage: Not reported) | Not reported | 353.465 | 369.723 |
| run 1: decrypt | available (legacy; coverage: Not reported) | Not reported | 71.1016 | 69.2969 |

Unavailable sampling is not zero memory use. Partial sampling missed some processes or snapshots.
Available samples can still miss short peaks. Process high-water RSS is reported separately.

Values are means across completed runs unless labelled otherwise. MiB = 2^20 bytes.
The frozen reference supplies quality, not matched plaintext timing.
Server-reported timings are additional detail; they never replace or subtract from harness wall times.
Key/public-workload uploads are counted once and amortized; no network-transfer duration is measured.

**Plaintext debug run: stage/payload names do not imply encryption. These are not FHE overhead results.**
