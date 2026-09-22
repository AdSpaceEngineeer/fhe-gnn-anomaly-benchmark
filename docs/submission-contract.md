# Submission contract, version 1

## Ownership

The benchmark owns features, normalized graph, model, activation target, sensitive
columns, scoring rule, threshold, split and numerical tolerances. Submissions own
encryption algorithms, parameters, encoding, packing and encrypted implementation.
Use the published frozen weights and checksums unchanged; do not retrain during
a benchmark. Arithmetic rearrangements must preserve the inference semantics.
There is one registered workload, `scam-list-gcn-100k-v1`, selected by default.
Internal software fixtures are not alternative benchmark sizes. The harness
supplies weights automatically; there is no user-facing model-preparation stage.
Any backend-specific weight encoding/packing is internal to evaluation and timed.
Polynomial/LUT approximations and fixed-point quantization are allowed if declared
and checked against the same frozen reference. Report quality even when numerical
verification fails; changing the tolerance/threshold creates another workload.

## Client and evaluator

The client receives the three sensitive normalized numeric columns. The evaluator
receives their ciphertexts, the other five public features, public normalized
adjacency, frozen weights and public/evaluation keys. It must not receive the
secret key, plaintext sensitive columns, labels, reference scores or raw log.
The client decrypts one anomaly score per event. The harness applies the fixed
threshold and calculates metrics on the fixed test nodes.

Stages run in separate processes on one machine, communicating through files.
This reproduces a logical client/server interface; it is not an OS sandbox.
All published synthetic artifacts are available locally, so submissions are
trusted research code and require source review. A malicious implementation can
read files or hardcode scores; automated output checks cannot prove FHE use.

## Python interface

| Method | Input | Output |
|---|---|---|
| `describe()` | None | JSON metadata including security evidence, parameters, encoding, packing, activation |
| `keygen(threads)` | Requested thread budget | `(private_files, public_files)` |
| `encrypt(sensitive, private_files, threads)` | `N x 3` floats in fixed sensitive-column order | Encrypted input files |
| `evaluate(encrypted, public, public_files, threads, intermediate_dir)` | Ciphertexts and public workload | Encrypted score files |
| `decrypt(encrypted_scores, private_files, threads)` | Result ciphertexts and client keys | List of `N` finite scores |

File payloads are dictionaries `{simple_filename: bytes}`. These are actual
serialized payloads, not estimated ciphertext sizes. Names cannot contain paths.
The adapter may call C++ binaries or other runtimes; document their installation.
Respect `threads` in the backend, not only BLAS environment variables.

An evaluator may write `intermediate_dir/server_reported_steps.json` using BERT's
optional flat `{name: seconds}` format, such as `{"Encrypted computation": 1.2,
"I/O": 0.3, "Total": 1.5}`. Document the scope and any overlapping timers.
The harness labels these as self-reported, adds them to JSON and the compact
comparison, and retains all independently measured main metrics unchanged.

`public` contains `x_public`, `public_indices`, `sensitive_indices`, `feature_count`,
`node_count`, `adjacency` (COO `rows`, `cols`, `values`), `weights`, and `activation`.
Weights use `encoder_1.weight`, `encoder_1.bias`, etc. Matrix shape is input width
by output width. Bias is added AFTER graph aggregation. Sensitive columns are
`transfer_amount_z`, `source_daily_total_amount_z`, `prior_report_count_z`.
Scores are the MEAN of their three squared reconstruction errors, not the sum.

## Security admission

At least 128-bit classical security is mandatory for every encryption/key-switch/
bootstrapping component. Declarations below 128 or missing parameter/evidence
fields are rejected. `seal_tc128` additionally checks the actual serialized
TenSEAL context against the declared chain and SEAL's standard parameter bounds.
`seal_tc128_native` performs equivalent parameter/public-key checks for the native
SEAL file format, constructing its context with explicit `TC128` security.
These validate the standard parameter choice, not arbitrary modifications to
SEAL's secret/error sampling or implementation.

Novel schemes use `validator: external_review`: include estimator/version,
inputs, outputs, assumptions and security references. Runs can be measured, but
reports remain `evidence_requires_review` and ineligible for comparison until a
maintainer adds/reviews the corresponding validator. Self-asserting 128 is not
approval. No security-none override is offered for FHE submissions.

One fresh key set is generated per invocation, reused for its repeat runs, and
replaced on the next invocation. Secret keys are never part of upload bytes.
Reports amortize key generation/upload over repeats; rotation is not exercised.
The client gets only final scores; intermediate decryption is outside v1.

## Artifacts and comparability

`manifest.json` hashes data, weights and reference files. The registry pins the
manifest digest for published instances. Unknown/changed instances can be tested
but are not registered comparisons. The loader recomputes plaintext predictions
and validates finite inputs, model dimensions and disjoint splits before running.
Data preparation/reference checking are outside measured FHE stage times.

Large files may be stored as `data.json.gz`, `reference.json.gz` or
`transactions.csv.gz`. The loader decompresses them automatically and checks
the SHA256 of the original uncompressed bytes against the unchanged manifest.
Both plain and compressed copies of the same logical file are rejected as
ambiguous. Compression is storage-only: it changes neither features nor scores.

The whole frozen graph is one workload instance. Repeats do not retrain or alter
it. Do not slice node rows and renormalize the graph to manufacture a smaller
query: this changes GCN predictions. Internal miniature regression fixtures are
not benchmark workloads and must never be compared with the trained workload.
