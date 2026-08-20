# Submission contract

This contract mirrors the conceptual client/server separation of the
HomomorphicEncryption.org ML-inference harness. Exact executable names may be
adapted when the upstream harness is forked.

## Required stages

| Stage | Responsibility | Required outputs |
|---|---|---|
| `client_key_generation` | Build cryptographic context and keys | elapsed time; public/evaluation-key bytes |
| `client_preprocess_input` | Truncate and encode identifiers | realized `k`, `L`, encoding metadata |
| `client_encrypt_input` | Encrypt encoded identifier features | elapsed time; ciphertext bytes |
| `server_preprocess_model` | Prepare frozen weights and public graph | elapsed time; cached artifact bytes |
| `server_encrypted_compute` | Produce encrypted anomaly scores | elapsed time; peak RAM; output bytes |
| `client_decrypt_output` | Decrypt and decode scores | elapsed time |
| `client_postprocess` | Apply fixed decision rule | labels/scores in harness format |
| `quality_check` | Compare to ground truth and plaintext baselines | recall, F1, accuracy, `Q(k)` |

## Conformance rules

1. The secret key never leaves the client boundary.
2. The graph, weights, splits, identifier seed, truncation rule, and decision
   threshold are fixed by the harness.
3. Every run records a machine-readable result conforming to
   `schemas/result.schema.json`.
4. A submission reports warm and cold runs separately and must not silently
   exclude setup, key upload, I/O, or failed runs.
5. A submission identifies any deviation from the frozen model or security
   target. Deviations are not ranked with conforming submissions.
6. No submission may claim that synthetic identifiers are private real-world
   records or that benchmark performance demonstrates production readiness.

## Planned instance sizes

Instance sizes will be frozen after profiling the plaintext graph workload.
They should cover single-node inference and reproducible small, medium, and
large induced subgraphs or fixed batches without requiring identical hardware.
The harness must publish node/edge counts and batch construction seeds.
