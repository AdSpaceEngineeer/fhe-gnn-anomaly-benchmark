# FHE Benchmark — Scam_List_GCN Inference

This benchmark asks whether a graph anomaly detector can score transactions while
three sensitive financial-risk features remain encrypted, and measures the
detection quality and FHE overhead together. The intended industry use is scam,
crime and fraud prevention.

It uses a simplified four-layer GCN attribute autoencoder inspired by
[Ding's DOMINANT implementation](https://github.com/kaize0409/GCN_AnomalyDetection_pytorch).
The architecture provides graph aggregation, matrix multiplication, additions,
polynomial activations and squared reconstruction error. The measurement
categories follow [the FHE Benchmarking Suite](https://fhe-benchmarking.github.io/);
the submission workflow is inspired by [its BERT harness](https://github.com/fhe-benchmarking/BERT).

**Every FHE submission must provide evidence of at least 128-bit classical security.**
The supplied CKKS example uses SEAL's standard tc128 parameter validation, with
additional checks of the actual serialized context.

## Current release

| Workload | Available | Meaning |
|---|---|---|
| `toy-v1` | Frozen inputs/weights/reference and real CKKS example included | Three-node, four-layer arithmetic smoke test; initialized weights, not trained |
| `scam-list-gcn-100k-v1` | Frozen 100,000-event data, trained weights, checkpoint, threshold and verified plaintext scores included | Trained synthetic scam/anomaly baseline; full-graph plaintext check available |

The toy is immediately runnable. Its Recall/F1 are illustrative and must not be
reported as evidence of fraud-detection quality. See [artifact status](artifacts/README.md).
The frozen trained baseline's test Recall is **0.897783**, F1 **0.915254** and
Accuracy **0.993250**. This is the newly frozen September 2026 retraining run,
not a recovered checkpoint from the earlier reported run. Full dataset definitions,
its head, model layers and baseline
provenance are in [the model/dataset notes](docs/scam-list-gcn.md).

## Dataset and sensitive fields

Each synthetic transaction is an event node. The GCN uses eight numeric features;
three must be encrypted: `transfer_amount_z`, `source_daily_total_amount_z` and
`prior_report_count_z`. These represent the transferred amount, the source
account's running daily transferred total and its prior complaint/report count.
They are preprocessed into numeric features before the adapter encodes/encrypts
them. Payment-channel indicators, normalized daily transaction count and graph
topology remain public under the v1 benchmark assumptions.

Read the [dataset guide](docs/dataset.md) for a transaction-log preview, every
field's definition, the exact model column order, sensitivity rationale and what
the evaluator receives. The guide distinguishes the original training dataset
from the three-row arithmetic toy included in this release. Both bundles are
included when cloning; large trained-data files use automatic lossless gzip loading.

## Execution modes

All stages currently execute on one machine in separate processes:

1. Generate a fresh key set and check security parameters.
2. Client encodes/encrypts sensitive model features.
3. Evaluator computes encrypted anomaly scores using public graph and weights.
4. Client decrypts/decodes scores.
5. Harness verifies score agreement and computes quality/overhead metrics.

The secret key stays in client files. No decryption is performed by the evaluator.
This is a logical client/server split, not a sandbox for untrusted code.
Keys are reused across repeat runs within one invocation; the next invocation
generates new keys. Training is excluded from benchmark execution.

## Dependencies

- Python **3.10–3.12**, Linux or Windows.
- Core dependencies in `requirements.txt`; no FHE library is forced on every user.
- Each submission supplies its own dependencies. The example uses TenSEAL 0.3.16.
- PyTorch/pandas are needed only for maintainer training/import tasks.

## Run the example

```bash
git clone https://github.com/AdSpaceEngineeer/fhe-gnn-anomaly-benchmark.git
cd fhe-gnn-anomaly-benchmark
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r submissions/toy_ckks/requirements.txt
python harness/run_submission.py --submission toy_ckks --threads 2
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`.
No model download, training or separate input generation is required for the toy.

Repeat the same graph inference three times with one key set:

```bash
python harness/run_submission.py --submission toy_ckks --num-runs 3 --threads 2
python harness/run_submission.py --help
```

Expected stage output:

```text
[harness] describe
[harness] keygen
[harness] check_context
[harness] encrypt
[harness] evaluate
[harness] decrypt
[harness] passed: .../report.json
```

The example defaults to two threads, is deliberately unoptimized, and supports
at most eight event nodes. Follow your machine's allocation rules. Environment
limits are also passed to numerical libraries; custom backends must honor the
thread argument. Increase neither threads nor workload size blindly.

## Run the trained plaintext baseline

No PyTorch, retraining, external dataset download or manual decompression is needed:

```bash
python harness/run_submission.py --submission plaintext_debug --debug-plaintext --artifacts artifacts/scam-list-gcn-100k-v1 --threads 2
```

The loader checks the registered manifest, data and model hashes, then recomputes
the frozen plaintext reference. The harness executes inference on all 100,000
nodes and computes quality metrics on the 20,000 fixed test nodes. The threshold
is fixed at `4.344182877864071`.

This is a **plaintext validation**, not an FHE measurement. The supplied toy CKKS
adapter cannot run this large bundle. An engineer's adapter can select it using
the same `--artifacts` option; no full-graph encrypted result is claimed here.
See [the frozen bundle](artifacts/scam-list-gcn-100k-v1/README.md) for provenance.

## Submit your implementation

1. Copy `submissions/template/` to `submissions/my_method/`.
2. Implement `Adapter` in `adapter.py`. Put your parameters, encoding and packing
   choices in your implementation. Add other source/build files as necessary.
3. Complete the submission README, including the technical method and security evidence.
4. Install your dependencies and run:

```bash
python -m pip install -r submissions/my_method/requirements.txt
python harness/run_submission.py --submission my_method --threads 2
```

That command defaults to the toy. When your adapter supports the full workload,
add `--artifacts artifacts/scam-list-gcn-100k-v1` to benchmark the trained model.

Keep the harness and published workload unchanged. We recommend using the
published frozen weights and model checksum as-is. The
[submission contract](docs/submission-contract.md) defines every method, file
format, security requirement and permitted approximation. Novel security
parameterizations remain marked for review even if numerical checks pass.

Optional supporting implementation files and native runtimes are allowed.
Backend/hardware identification is optional; parameters needed to assess security
are required. Use `--include-hardware` only if you want an OS/CPU summary recorded.

## Results

Each run writes a new `measurements/<run-id>/report.json` containing:

- Recall and F1 (primary), Accuracy, Precision, ROC-AUC and average precision.
- Maximum/mean score error and prediction agreement against the fixed plaintext reference.
- Per-stage wall time, throughput and repeated-run measurements.
- Peak process memory, serialized key/input/output sizes and persisted intermediates.
- Upload/download payload bytes and amortized key overhead.
- Artifact checksums, security status, submission description and verification status.

Numerical PASS is separate from security approval and workload registration.
All three are needed for an eligible comparison, and only within the same
artifact manifest. Network transfer time and unexercised key rotation are
`null`; this local harness measures communication bytes. See
[measurement definitions](docs/measurements.md) for scope and limitations.

Share `report.json`, not the `io/` directory: it contains client secrets and
plaintext test inputs. Hardware is not collected unless requested.

## Directory structure

```text
README.md
requirements.txt                  Core inference dependencies
requirements-training.txt         Maintainer-only training/import dependencies
harness/
  run_submission.py               Main command
  worker.py                       Executes one client/server stage
  params.py, utils.py             Fixed settings, serialization and timing
  model.py, generate_input.py      Frozen inference and validated input loading
  verify_result.py, metrics.py     Numerical checks and detection metrics
  security.py                     Security evidence/context checks
artifacts/                        Frozen instances, checksums and registry
submissions/
  template/                       Starting point for a new submission
  toy_ckks/                       Real encrypted arithmetic example
  plaintext_debug/                Pipeline check; no FHE
scripts/
  scam_list_gcn.py                 Original maintainer training/generation script
  export_artifacts.py             Import an existing run without retraining
  build_toy_artifacts.py           Maintainer toy fixture provenance
docs/                             Technical specification and usage contracts
tests/                            Integrity, failure cases and integration tests
examples/                         Measured toy report, without keys
```

## Tests and troubleshooting

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
RUN_FHE_TESTS=1 python -m pytest -q tests/test_ckks.py
```

PowerShell: set `$env:RUN_FHE_TESTS="1"` before running the second test command.

A missing backend should be installed from the submission's requirements.
An invalid security parameter set is rejected before encrypted inference.
A scale/depth error means the chosen parameters cannot execute the circuit;
it is not permission to lower the security requirement.
An existing `--out` directory is rejected to preserve previous runs.
Never fix a checksum failure by silently updating the published manifest.

To check the harness without installing an FHE backend:

```bash
python harness/run_submission.py --submission plaintext_debug --debug-plaintext
```

This mode performs no encryption and is never eligible as an FHE result.
