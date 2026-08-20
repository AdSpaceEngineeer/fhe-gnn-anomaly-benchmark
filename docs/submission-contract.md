# Submission contract

The harness is implementation-agnostic. A submission provides a JSON manifest
whose `commands` object maps stages to non-empty argument arrays. Commands are
executed directly, without a shell. `$PYTHON` expands to the harness Python
interpreter and `$SUBMISSION_DIR` expands to the manifest's directory.

## Required stages

| Stage | Responsibility | Required output |
|---|---|---|
| `client_key_generation` | Create context, public key, secret key, and evaluation material | `setup/public_keys/`, `setup/evaluation_keys/` |
| `server_preprocess_model` | Prepare declared model weights and public graph-independent state | `setup/server_intermediates/` |
| `client_preprocess_input` | Apply exact `k`, balanced prefix/suffix split, and canonical encoding | `run/client_preprocessed/` |
| `client_encrypt_input` | Encrypt identifier-derived logical inputs | `run/ciphertexts_upload/` |
| `server_encrypted_compute` | Evaluate the declared GNN and produce encrypted target logits | `run/ciphertexts_download/` or `run/encrypted_outputs/` |
| `client_decrypt_output` | Decrypt and decode target logits | `run/decrypted_outputs/` |
| `client_postprocess` | Apply fixed sigmoid/threshold and write scores | `run/client_outputs/scores.npy` |

`scores.npy` must be a one-dimensional float array aligned with the harness
target order, with every score in `[0,1]`.

## Optional and conditional stages

- `server_upload_evaluation_key` measures remote initial key upload.
- `client_key_rotation` and `server_upload_rotated_evaluation_key` are both
  mandatory when `key_policy` is `ephemeral_per_batch`. Rotated key artifacts
  go under `run/rotated_evaluation_keys/`.

The harness runs setup once. For `ephemeral_per_batch`, it runs the two rotation
stages before every measured inference after the first.

## Environment

Every stage receives:

| Variable | Meaning |
|---|---|
| `FHE_BENCH_STAGE` | Current stage name |
| `FHE_BENCH_WORK_DIR` | Current setup or per-run artifact root |
| `FHE_BENCH_STAGE_REPORT` | Optional JSON self-report path |
| `FHE_BENCH_SETUP_DIR` | Persistent setup artifact root |
| `FHE_BENCH_INSTANCE_DIR` | Fixed public graph, target indices, and client input |
| `FHE_BENCH_MODEL_PATH` | Declared plaintext model artifact used as oracle |
| `FHE_BENCH_MODEL_SHA256` | Required SHA-256 of that model artifact |
| `FHE_BENCH_RETENTION` | Requested retention fraction |
| `FHE_BENCH_RETAINED_CHARACTERS` | Realized integer `k` |
| `FHE_BENCH_FULL_LENGTH` | Full identifier length `L` |

The client identifier file is logically client-side. FHE server code must not
read it, client-preprocessed plaintext, harness labels, decrypted outputs, or
the secret key. Public topology/features, target indices, declared model
semantics, batch size, and `k` may be visible to the server.

## Stage self-report

A stage may write:

```json
{
  "peak_ram_bytes": 123456,
  "detail_seconds": {
    "ciphertext_io": 0.12,
    "homomorphic_operations": 4.56
  }
}
```

Peak RAM is required for encryption, encrypted compute, and decryption in a
real submission. Internal timing is supplementary. The harness independently
records process wall time and measures artifact bytes from fixed directories.

## Manifest example

```json
{
  "name": "example-ckks-backend",
  "implementation": "Example Library 1.2",
  "scheme": "CKKS",
  "claimed_security_bits": 128,
  "key_policy": "reused",
  "is_fhe": true,
  "parameters": {
    "polynomial_modulus_degree": 32768
  },
  "additional_leakage": [],
  "commands": {
    "client_key_generation": ["$SUBMISSION_DIR/bin/keygen"],
    "server_preprocess_model": ["$SUBMISSION_DIR/bin/model-setup"],
    "client_preprocess_input": ["$SUBMISSION_DIR/bin/preprocess"],
    "client_encrypt_input": ["$SUBMISSION_DIR/bin/encrypt"],
    "server_encrypted_compute": ["$SUBMISSION_DIR/bin/evaluate"],
    "client_decrypt_output": ["$SUBMISSION_DIR/bin/decrypt"],
    "client_postprocess": ["$SUBMISSION_DIR/bin/postprocess"]
  }
}
```

## Conformance rules

1. The secret key never leaves the client boundary.
2. Protected server computation uses FHE, not plaintext identifiers or a
   trusted execution environment presented as FHE.
3. The supplied model artifact is immutable during a run and is identified by
   `FHE_BENCH_MODEL_SHA256`. Using the repository's published bundle unchanged
   is recommended when reporting results against that reference. A compatible
   alternative model bundle is permitted and receives its own recorded name,
   adapter, checksum, and automatically computed quality metrics.
4. Comparable FHE results claim at least 128-bit security and document scheme
   parameters supporting that claim.
5. Every measured configuration produces three result files by default and
   records all runs, including failures separately rather than silently
   excluding them.
6. Warm/cold behavior, setup, remote transfer, caching, hardware, software,
   selected model bundle, and additional leakage are disclosed.
7. The plaintext dry run is never reported as encrypted performance.
8. Synthetic identifiers are not represented as private real-world records,
   and benchmark performance is not represented as production readiness.

Before results from different implementations are placed in the same table,
run `fhe-gnn-benchmark validate-comparison` on them. It rejects differences in
the model hash, dataset, batch size, retained character count, identifier
length, or seed.
