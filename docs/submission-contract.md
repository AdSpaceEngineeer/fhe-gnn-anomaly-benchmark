# Submission interface

Create `submissions/<method>/adapter.py`, `README.md` and a submission-specific
`requirements.txt`. Implement the methods in `submissions/template/adapter.py`.
The same adapter may support one or both workloads. The runner requires an
explicit `--workload gcn` or `--workload tam` selection.

## Methods

| Method | Responsibility |
|---|---|
| `configure(workload, threads)` | Configure the workload and backend in each fresh stage process |
| `describe()` | Declare supported workloads, scheme, parameters, encoding, packing, nonlinear implementation and security evidence |
| `thread_report(stage, requested_threads)` | Report compute parallelism actually configured for this stage |
| `keygen(threads)` | Return separate client-private and evaluator-public key files |
| `encrypt(sensitive, private_files, threads)` | Encode and encrypt the supplied N x 3 feature matrix |
| `evaluate(encrypted, public, public_files, threads, intermediate_dir)` | Evaluate the selected frozen workload and return encrypted scores |
| `decrypt(encrypted_scores, private_files, threads)` | Return N numeric scores in the published node order |

File bundles are dictionaries mapping simple filenames to bytes. Key generation
returns `(private_files, public_files)`; encryption and evaluation return one
file bundle each. Objects in Python memory do not persist across stage processes.

An incomplete submission may declare nonempty `unsupported_operations` in
`describe()`. The runner then stops before key generation and records an error,
without presenting incomplete work as a measured inference. The GCN-only CKKS
example uses this declaration for its required encrypted-ReLU placeholder.

## Evaluator inputs

`public` contains `workload`, `weights`, `adjacency`, `x_public`,
`public_indices`, `sensitive_indices`, `feature_count`, `node_count` and
`activation`. Adjacency uses COO arrays `rows`, `cols`, `values`, with an
implicit N x N shape. TAM additionally supplies `scoring_adjacency` and
`norm_epsilon`. Its encoder adjacency is the frozen normalized truncated graph;
its scoring adjacency is the original binary graph with self-loops.

The evaluator receives no client secret key, sensitive plaintext columns, raw
account strings, labels or reference scores through this interface. This is a
logical separation on one machine, not an operating-system sandbox against a
malicious submission. Source review and deployment isolation remain necessary.

Use the frozen data, weights, graph and feature order as supplied. Do not train,
refit preprocessing, change graph connectivity or tune the decision threshold
during a benchmark run. Reference inference uses standard ReLU or PReLU and
cosine normalization. Submitters choose and document their encrypted realization;
the reference workload does not provide approximation coefficients or LUTs.

## Threads

`--threads` is a request passed to the submission and common numerical-library
environment settings. It is not an enforced CPU allocation. Every successful
stage must return a thread report:

```python
{"compute_threads": 8, "worker_processes": 1,
 "threading_model": "Eight backend compute threads; no nested worker pools"}
```

`compute_threads` is the maximum configured concurrent compute threads across
all worker processes, not merely the requested value. Report single-threaded
implementations as 1. Explain multiple pools, nested parallelism and process
counts. These values are self-reported. The harness also samples process-tree
OS thread counts; idle, runtime and I/O threads make that a different measure.

## Security and keys

Submissions require at least 128-bit classical security. Parameters and evidence
belong in the submission. A claimed bit count alone is insufficient. Built-in
SEAL validators check declared bounds and the serialized context; other schemes
are marked `evidence_requires_review` rather than automatically approved.

Generate one fresh key set per invocation and reuse it across repetitions.
Public, secret and evaluation keys are distinct members of that set. Server key
files must exclude secret keys. Key rotation and network upload time are not
measured by the current single-machine runner.

## Results

Return one score per node. The harness checks numerical agreement with the
published reference, then applies the fixed threshold and test split. Accuracy
and ROC-AUC are primary; Recall, F1, Precision and Average Precision are also
reported. Numerical pass/fail is separate from task quality. An inaccurate
submission still receives quality/error results but fails the fidelity gate.

Optionally write `server_reported_steps.json` inside `intermediate_dir`, for example
`{"Encrypted computation": 1.2, "I/O": 0.3, "Total": 1.5}`. Values are finite,
nonnegative seconds. These self-reported details supplement measured stage wall
times and never replace or reduce them.
