# Measurements

The runner writes `report.json` and `comparison.md` for each invocation. Compare
submissions only on the same artifact ID and manifest checksum. Record thread
configuration alongside results. Hardware reporting is optional via
`--include-hardware`; meaningful runtime comparisons require comparable hardware.

| Measurement | Definition |
|---|---|
| Accuracy and ROC-AUC | Primary test-set quality metrics using decrypted scores |
| Recall, F1, Precision, Average Precision | Additional test-set quality metrics |
| Score error | Maximum and mean absolute difference from frozen plaintext scores |
| Prediction agreement | Fraction of all nodes with matching threshold decisions |
| Stage latency | Wall time of each keygen/encrypt/evaluate/decrypt process, including startup and file I/O |
| Inference latency | Encryption + evaluation + decryption wall times; key generation excluded |
| Throughput | Nodes computed divided by evaluation time; end-to-end throughput also reported |
| Peak RAM | Maximum observed stage/process memory, including sampled child processes |
| Storage | Serialized keys, input/output ciphertext payloads and persisted intermediates |
| Communication | Serialized input/result bytes; public workload and key uploads counted once |
| Threads | Requested limit, required submitter-reported configuration and sampled OS thread count |

GCN throughput is events/second; TAM throughput is accounts/second. Each
inference computes the complete fixed graph. Repetitions run that same workload
with the invocation's key set; they are not independently sized subgraphs.

Memory combines the stage process lifetime high-water RSS with 10 ms samples
of aggregate process-tree RSS. Sampling can miss short peaks; summing RSS can
count shared pages more than once. Key, ciphertext and persisted intermediate
sizes exclude transient in-memory objects. OS thread sampling is not a count of
active CPU cores or proof of a submission's thread declaration.

Each stage in `report.json` includes `memory_sampling_status`: `available`
(usable complete snapshots), `partial` (some processes/snapshots missed), or
`unavailable` (no usable positive RSS observation). The sampled peak is `null`
when unavailable, never a placeholder zero. `memory_sample_attempts`,
`memory_samples_observed` and `memory_samples_incomplete` describe coverage;
incomplete attempts include missing and partially observed snapshots.
Availability does not guarantee that brief peaks were captured.
`comparison.md` shows sampling status, observation count, sampled peak and
process high-water RSS separately, including completed stages of failed runs.
Historical examples retain their original measurements and mark sample coverage
as unknown; unavailable sample counts are `null`, not invented retrospectively.

Communication measures bytes, not actual network transport. Key upload time,
rotation time and network latency remain null, not zero. Key generation cost is
reported once and amortized across repetitions. No plaintext runtime baseline is
implied by the frozen plaintext quality metrics.

## Optional server timings

Following the BERT harness convention, submissions may supply named timings in
`server_reported_steps.json`. Report arithmetic separately from context loading,
serialization and file I/O where possible. The runner also measures its own
evaluate-stage file I/O and the adapter call; the latter can include work other
than arithmetic. Invalid optional timing files produce a warning, not a fabricated
measurement. Main stage wall times remain authoritative elapsed measurements.

## Verification

A score passes numerical verification when
`abs(submitted - reference) <= atol + rtol * abs(reference)` for every node.
The registered manifests use `atol=0.001`, `rtol=0.001`. Quality metrics are
reported even when this numerical gate fails. The threshold is never refitted
on a submission. ROC-AUC uses continuous scores; Accuracy uses the published
validation-selected threshold.

`eligible_for_comparison` requires a registered unchanged bundle, successful
numerical verification and accepted security evidence. It is an automated
eligibility flag, not a cryptographic audit or certification. Plaintext debug runs
are never eligible FHE results.

The metric categories follow the [FHE Benchmarking Suite](https://fhe-benchmarking.github.io/).
Optional server timings follow its [BERT harness](https://github.com/fhe-benchmarking/BERT).
