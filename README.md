# FHE Benchmark — Scam_List_GCN Inference

## Overview

This repository provides an inference benchmark for fully homomorphic encryption
(FHE) in graph-based scam, crime and fraud prevention. It measures the accuracy
and computational overhead of anomaly scoring when selected financial-risk
features remain encrypted during evaluation.

The workload is **Scam_List_GCN**, a simplified four-layer graph convolutional
autoencoder inspired by [Ding's DOMINANT implementation](https://github.com/kaize0409/GCN_AnomalyDetection_pytorch).
Its operations include graph aggregation, matrix multiplication, polynomial
activations and squared reconstruction error. The benchmark uses one frozen
model and a synthetic dataset of 100,000 transaction events. Model weights,
preprocessing, graph structure and reference scores are included in the repository;
training is outside the benchmark.

Submitters should clone this repository and implement their method in
`submissions/<submission>/`. Each submission contains its implementation,
dependencies and technical README, including cryptographic parameters, encoding
and packing strategy. The harness supplies the fixed workload and evaluates the
submission against the published plaintext reference. The model, dataset,
threshold and harness remain unchanged.

## Execution model

All stages run on one machine in separate processes, with files representing
client/server communication.

| Component | Responsibility |
|---|---|
| Client | Generate keys; encode and encrypt sensitive inputs; decrypt the resulting anomaly scores |
| Evaluator | Compute encrypted scores using ciphertext inputs, public features, normalized graph, frozen weights and public/evaluation keys |
| Harness | Validate workload integrity and security parameters, orchestrate stages, verify scores and record measurements |

The three encrypted model features are `transfer_amount_z`,
`source_daily_total_amount_z` and `prior_report_count_z`: normalized transaction
amount, running daily transferred total and prior report count. Payment-channel
indicators, normalized daily transaction count, graph topology and model weights
are public under the benchmark's v1 confidentiality boundary.

The evaluator interface excludes secret keys, plaintext sensitive features,
labels and reference scores. This separation is a logical research interface,
not an operating-system sandbox. One fresh key set is generated per invocation
and reused across its repeat runs.

## Running the benchmark

### Dependencies

- Python 3.10–3.12.
- Core packages in `requirements.txt`.
- Submission-specific packages in `submissions/<submission>/requirements.txt`.
  The supplied CKKS example uses TenSEAL 0.3.16 and its Microsoft SEAL bindings.

### Installation

Clone the repository, create an environment and copy the CKKS example:

```console
git clone https://github.com/AdSpaceEngineeer/fhe-gnn-anomaly-benchmark.git
cd fhe-gnn-anomaly-benchmark

python -m venv .venv
source .venv/bin/activate

python -c "import shutil; shutil.copytree('submissions/toy_ckks', 'submissions/my_method')"
python -m pip install -r requirements.txt -r submissions/my_method/requirements.txt
```

Copy the entire submission directory, including its helper files. The example
is a starting implementation that submitters can modify; `submissions/template/`
provides a scheme-independent alternative.

### Execution

Run the submission with one command:

```console
python harness/run_submission.py --submission my_method --threads 2
```

The harness selects `scam-list-gcn-100k-v1` and supplies its frozen model and data
automatically. No training, model download or manual weight preparation is
required. Use `--num-runs` for repeated measurements and `--help` for all options.

The CKKS example is unoptimized and targets the full workload. Its interface and
packing algebra have been tested, but a completed encrypted run of this revision
has not been validated. Resource requirements and implementation details are
documented in the [submission README](submissions/toy_ckks/README.md).

## Metrics and security

| Category | Measurements |
|---|---|
| Model performance | Anomaly Recall and F1 (primary); Accuracy, Precision, ROC-AUC and average precision |
| Numerical agreement | Score error and prediction agreement against the frozen plaintext reference |
| Latency and throughput | Stage wall times, total inference time and event throughput |
| Memory consumption | Peak process memory and sampled process-tree memory |
| Storage requirements | Serialized keys, input/output ciphertexts and retained intermediate files |
| Communication complexity | Serialized client/server payload sizes, including one-time key and public-workload uploads |

Each run produces `report.json` and a compact `comparison.md` in
`measurements/<run-id>/`. Stage wall times include file I/O and process overhead.
Network-transfer and key-rotation durations are not measured.

Following the BERT harness convention, submissions may additionally write
`intermediate_dir/server_reported_steps.json`, a dictionary of named durations
in seconds. Arithmetic and I/O timings are recorded as optional server-reported
detail, separate from the harness's independent measurements.

**FHE submissions must provide evidence of at least 128-bit classical security.**
The harness checks supported SEAL contexts against declared parameters. Other
schemes require security-evidence review before their results are eligible for
comparison; numerical agreement alone does not establish security.

Share the JSON report and comparison table, not the run's `io/` directory, which
contains client secrets and plaintext inputs. Hardware reporting is opt-in.
See the [measurement definitions](docs/measurements.md) and
[submission contract](docs/submission-contract.md) for reporting and admission rules.

## Example output

The following results come from a verified **plaintext-only** run of the complete
100,000-event workload. Detection metrics use its fixed 20,000-event test split.

| Metric | Frozen reference | Plaintext verification run |
|---|---:|---:|
| Anomaly Recall | 0.897783 | 0.897783 |
| Anomaly F1 | 0.915254 | 0.915254 |
| Accuracy | 0.993250 | 0.993250 |
| Maximum absolute score error | — | 0 |
| Prediction agreement | — | 1.000000 |

To run this verification:

```console
python harness/run_submission.py --submission plaintext_debug --debug-plaintext --threads 2
```

The corresponding [JSON report](examples/frozen_plaintext_report.json) and
[comparison table](examples/frozen_plaintext_comparison.md) demonstrate the output
format. These are synthetic-data plaintext results, not FHE performance measurements
or evidence of real-world fraud-detection accuracy.

## Directory structure

```text
├── README.md
├── requirements.txt          # Scheme-independent harness dependencies
├── harness/                  # Fixed workload execution, verification and metrics
│   ├── run_submission.py     # Main entry point
│   ├── model.py              # Plaintext inference reference
│   ├── verify_result.py      # Score and prediction verification
│   └── reporting.py          # Optional timings and comparison tables
├── artifacts/                # Frozen dataset, weights, scores and checksum registry
├── submissions/              # User implementations and their dependencies
│   ├── toy_ckks/             # Copyable CKKS implementation
│   ├── template/             # Scheme-independent adapter template
│   └── plaintext_debug/      # Plaintext pipeline verification
├── measurements/             # Generated run reports and client/server files
├── examples/                 # Example reports with their validation status
├── docs/                     # Dataset, architecture and benchmark specifications
├── scripts/                  # Maintainer training and artifact export
└── tests/                    # Integrity, interface and arithmetic tests
```

## Stage descriptions

The harness invokes the following methods on a submission's `Adapter` class:

| Interface method | Role | Inputs and outputs |
|---|---|---|
| `describe()` | Declare the implementation | Return parameters, security evidence, encoding, packing and activation details |
| `keygen(threads)` | Client key generation | Return separate client-private and public/evaluation key bundles |
| `encrypt(sensitive, private_files, threads)` | Client encoding and encryption | Transform the three sensitive feature columns into serialized ciphertext inputs |
| `evaluate(encrypted, public, public_files, threads, intermediate_dir)` | Server inference | Use ciphertexts and the public workload to return encrypted anomaly scores |
| `decrypt(encrypted_scores, private_files, threads)` | Client decryption and decoding | Return one numeric anomaly score per event |

The harness checks the cryptographic context after key generation and verifies
scores after decryption. Scheme-specific encoding and packing remain inside the
submission and its measured stages. The [submission contract](docs/submission-contract.md)
defines payload formats and the exact public inputs.

## Technical references

- [Dataset specification](docs/dataset.md): transaction-log head, field definitions,
  feature encoding, graph construction and sensitive columns.
- [GCN architecture and plaintext baseline](docs/scam-list-gcn.md): layers, scoring
  rule, training provenance and detection results.
- [Benchmark methodology](docs/benchmark-spec.md): industry motivation, model
  selection, fixed-workload policy and ciphertext operations.
- [Frozen workload artifacts](artifacts/scam-list-gcn-100k-v1/README.md): published
  model, dataset and integrity checks.
- [DOMINANT implementation](https://github.com/kaize0409/GCN_AnomalyDetection_pytorch):
  starting architectural reference for the simplified GCN.
- [FHE Benchmarking Suite](https://fhe-benchmarking.github.io/): measurement categories
  and minimum-security requirement.
- [BERT inference benchmark](https://github.com/fhe-benchmarking/BERT):
  submission workflow and optional server-timing convention.
