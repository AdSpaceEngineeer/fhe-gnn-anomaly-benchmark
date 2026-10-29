# Benchmark comparison

Submission: `plaintext_debug` · Workload: `scam-list-tam-synthetic-v1`
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
| Accuracy (primary) | 0.963288 | 0.963288 |
| ROC-AUC (primary) | 0.927265 | 0.927265 |
| Recall | 0.598338 | 0.598338 |
| F1 | 0.599168 | 0.599168 |
| Precision | 0.6 | 0.6 |
| Average precision | 0.640382 | 0.640382 |
| Maximum score error (worst run) | 0 | 0 |
| Prediction agreement | 1 | 1 |
| Requested threads | Not reported | 2 |
| Reported evaluator compute threads | Not reported | 2 |
| Sampled evaluator OS threads (peak) | Not reported | 12 |
| Key generation (s, once) | Not reported | 0.568501 |
| Encrypt wall time (s) | Not reported | 0.755828 |
| Evaluate wall time (s) | Not reported | 1.835 |
| Decrypt wall time (s) | Not reported | 0.63266 |
| Inference wall time (s) | Not reported | 3.22349 |
| Evaluator throughput (nodes/s) | Not reported | 21447.9 |
| Peak stage RAM (MiB, maximum) | Not reported | 319.273 |
| Public/evaluation keys (MiB) | Not reported | 0 |
| Input payload (MiB) | Not reported | 2.40144 |
| Output payload (MiB) | Not reported | 0.784155 |
| Persisted intermediates (MiB) | Not reported | 0 |
| Communication (MiB/run, amortized) | Not reported | 69.6164 |

Memory sampling by stage:

| Stage | Sampling | RSS observations | Sampled peak MiB | Process high-water MiB |
|---|---|---:|---:|---:|
| description_stage | available (legacy; coverage: Not reported) | Not reported | 58.7109 | 54.7891 |
| keygen | available (legacy; coverage: Not reported) | Not reported | 59.0703 | 54.8984 |
| security_check | available (legacy; coverage: Not reported) | Not reported | 58.8945 | 54.8281 |
| run 1: encrypt | available (legacy; coverage: Not reported) | Not reported | 69.8672 | 67.5508 |
| run 1: evaluate | available (legacy; coverage: Not reported) | Not reported | 308.43 | 319.273 |
| run 1: decrypt | available (legacy; coverage: Not reported) | Not reported | 63.168 | 61.2891 |

Unavailable sampling is not zero memory use. Partial sampling missed some processes or snapshots.
Available samples can still miss short peaks. Process high-water RSS is reported separately.

Values are means across completed runs unless labelled otherwise. MiB = 2^20 bytes.
The frozen reference supplies quality, not matched plaintext timing.
Server-reported timings are additional detail; they never replace or subtract from harness wall times.
Key/public-workload uploads are counted once and amortized; no network-transfer duration is measured.

**Plaintext debug run: stage/payload names do not imply encryption. These are not FHE overhead results.**
