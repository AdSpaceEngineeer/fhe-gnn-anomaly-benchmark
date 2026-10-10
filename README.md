# FHE-GNN Anomaly Detection Benchmark

## Overview

This benchmark evaluates encrypted graph-neural-network inference for scam,
crime and fraud prevention. It measures whether an evaluator can compute anomaly
scores from protected financial-behaviour features while preserving model
quality, and records the computational and data-transfer costs of doing so.

Two frozen workloads are provided:

| Selection | Workload | Inference | Dataset |
|---|---|---|---|
| `gcn` | Scam_List_GCN | Four-layer ReLU feature-reconstruction GCN | 100,000 synthetic transaction events; 8 features |
| `tam` | Scam_List_TAM | Two-layer PReLU GCN with normalized neighbourhood affinity | 39,357 synthetic accounts; 10 features |

TAM is the **higher computational challenge**, adding norm-dependent
normalization and encrypted affinity products. It is not presented as better
fraud detection: the workloads have separate datasets, weights and baselines.
Both are synthetic research workloads, not evidence of real-world detection
performance.

Submissions implement encryption, encrypted evaluation and decryption using
their chosen scheme and backend. Three designated feature columns are protected
in each workload; graph structure, remaining features and model weights are
public. Standard ReLU/PReLU and normalization remain in the reference models.
Their encrypted implementation is the submitter's responsibility.

## Execution model

The client generates one fresh key set, encrypts the sensitive inputs, and
decrypts the returned anomaly scores. The evaluator receives public/evaluation
keys, ciphertexts and the public workload inputs. It must not receive secret
keys or sensitive plaintext features.

The runner executes each stage in a separate process, verifies outputs against
the frozen reference, and applies the published threshold and test split.
Repeated runs reuse the key set. This single-machine interface separates client
and evaluator files logically; it is not a sandbox for untrusted code.

Training and data preparation are outside benchmark execution. Datasets,
weights, preprocessing and thresholds are already included; no model preparation
or retraining is required from submitters.

## Running the benchmark

Use Python 3.10–3.12. Install the common dependencies and create a submission:

```bash
git clone https://github.com/AdSpaceEngineeer/fhe-gnn-anomaly-benchmark.git
cd fhe-gnn-anomaly-benchmark
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp -r submissions/template submissions/my_method
```

Implement `submissions/my_method/adapter.py`, document the method in its README,
and list its dependencies in its own `requirements.txt`. The common requirements
do not select an FHE library. Install the submission's dependencies, then run:

```bash
python -m pip install -r submissions/my_method/requirements.txt
python harness/run_submission.py --workload gcn --submission my_method --threads 2 --out measurements/my-gcn-run
```

Select `--workload tam` when the submission supports TAM. The workload selection
is mandatory. Use a new output directory for every invocation; `--num-runs`
controls repetitions. `--threads` requests parallelism; submitters must also
report the compute threads actually configured.

To check installation without performing any FHE operations:

```bash
python harness/run_submission.py --workload gcn --submission plaintext_debug --debug-plaintext --out measurements/plaintext-check
```

The plaintext check is not an FHE submission. See the
[submission interface](docs/submission-contract.md) for the complete contract.

An optional [incomplete CKKS example for GCN](submissions/toy_ckks/README.md)
demonstrates key setup, encryption and decryption. Its encrypted-ReLU placeholder
must be implemented by the submitter. The unchanged example is rejected before
expensive benchmark execution; no activation approximation is supplied.

### Guidance for first-time FHE engineers

Review the [model internals and observed ranges](docs/model-internals.md).
Measure sample ciphertext/key sizes and estimate storage before a full run;
check RAM and personal disk quotas separately. First exercise stage interfaces
and reporting with a small, clearly labelled non-benchmark diagnostic of your
submission. Plan depth across GCN's four graph convolutions, three ReLUs and
squared-error scoring; an isolated activation test does not establish full-model
feasibility. TAM additionally requires norm-dependent normalization. Running this
benchmark does not itself guarantee cryptographic security.

## Metrics and security

The runner writes `report.json` and a compact `comparison.md` table.

| Category | Reported measurements |
|---|---|
| Model quality | **Accuracy and ROC-AUC**; Recall, F1, Precision and Average Precision |
| Fidelity | Score errors, prediction agreement and numerical verification |
| Latency and throughput | Key generation; encryption, evaluation and decryption wall times; total inference; nodes/second |
| Memory | Per-stage process high-water RAM, sampled process-tree peak RAM and sampling availability |
| Storage | Serialized keys, input/output ciphertexts and persisted intermediates |
| Communication | Serialized client/server payload bytes; one-time and amortized key uploads |
| Parallelism | Requested threads, submitter-reported compute threads/processes and sampled OS threads |

Optional server timings distinguish encrypted arithmetic and file handling;
they supplement, not replace, measured wall time. Real network transfer duration
and key rotation are not measured. See [measurement definitions](docs/measurements.md).

FHE submissions require **at least 128-bit classical security**. Declare the
parameters and supporting evidence in the submission. SEAL-based contexts have
built-in parameter/context checks; other schemes require evidence review.
An automated check is not a cryptographic security certification. Hardware
details are optional (`--include-hardware`); thread reporting is required.

Only the same registered workload/artifact version should be compared.
Submission quality is reported even when numerical verification fails.
Publish only `report.json` and `comparison.md`: the run directory also contains
client secret keys and sensitive plaintext inputs.

## Example output

**Verified plaintext reference results — not FHE results.** The values below
use the frozen float64 reference inference, fixed test splits and thresholds.

| Workload | Test nodes | Accuracy | ROC-AUC | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Scam_List_GCN | 20,000 | 0.993400 | 0.998725 | 0.902709 | 0.917397 |
| Scam_List_TAM | 7,872 | 0.963288 | 0.927265 | 0.598338 | 0.599168 |

Full plaintext integration reports and comparison tables are in
[examples/](examples/README.md). Their timings describe the verification runs,
not encrypted performance or portable performance guarantees.

## Directory structure

```text
artifacts/          Frozen datasets, weights, references and checksum registry
docs/               Dataset, model, operation and submission specifications
examples/           Verified plaintext reports and comparison tables
harness/            Shared runner, stage execution, security and measurements
  workloads/        Workload-specific reference inference
scripts/            Maintainer-only training and export utilities
submissions/        Submission template and implementations
tests/              Interface, model, security and reporting tests
requirements.txt    Scheme-independent inference dependencies
```

## Stage descriptions

| Stage / method | Responsibility |
|---|---|
| `configure(workload, threads)` | Configure each stage's backend and workload |
| `describe()` | Declare scheme, parameters, encoding, packing and security evidence |
| `thread_report(stage, requested_threads)` | Report configured compute parallelism |
| `keygen(threads)` | Generate and separate private and evaluator-public keys |
| Context check | Harness checks applicable serialized security parameters |
| `encrypt(...)` | Encode and encrypt the designated feature columns |
| `evaluate(...)` | Compute encrypted anomaly scores using the frozen workload |
| `decrypt(...)` | Return decoded scores in the published node order |
| Verification and reporting | Harness computes fidelity, quality and overhead measurements |

## Technical references

- [Benchmark specification](docs/benchmark-spec.md) and [operation comparison](docs/operations.md).
- GCN: [dataset](docs/dataset.md), [model](docs/scam-list-gcn.md), [frozen artifacts](artifacts/README.md).
- TAM: [dataset](docs/dataset-tam.md), [model](docs/scam-list-tam.md).
- [Ma et al., A Comprehensive Survey on Graph Anomaly Detection with Deep Learning](https://arxiv.org/abs/2106.07178), journal publication 2023.
- [DOMINANT implementation](https://github.com/kaize0409/GCN_AnomalyDetection_pytorch), the starting architectural reference for the simplified GCN.
- [TAM paper](https://arxiv.org/abs/2306.00006) and [implementation](https://github.com/mala-lab/TAM-master), the affinity-workload references.
- [T-Finance dataset paper](https://proceedings.mlr.press/v162/tang22b/tang22b.pdf), inspiration for the synthetic account schema, not the data source.
- [FHE Benchmarking Suite](https://fhe-benchmarking.github.io/) and [BERT benchmark](https://github.com/fhe-benchmarking/BERT), references for measurement categories and staged execution.
