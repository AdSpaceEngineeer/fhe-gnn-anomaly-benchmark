# FHE Benchmark — Scam_List_GCN Inference

Measure whether a graph anomaly detector can score transactions while three
sensitive financial-risk features remain encrypted. The intended industry use is
scam, crime and fraud prevention; the released data are entirely synthetic.

The simplified four-layer GCN autoencoder is inspired by
[Ding's DOMINANT implementation](https://github.com/kaize0409/GCN_AnomalyDetection_pytorch).
It provides graph aggregation, matrix multiplication, additions, polynomial
activations and squared reconstruction error. The measurement categories follow
[the FHE Benchmarking Suite](https://fhe-benchmarking.github.io/), and the optional
server-timing format follows [its BERT harness](https://github.com/fhe-benchmarking/BERT).

**Every FHE submission must provide evidence of at least 128-bit classical security.**

## One fixed workload

The benchmark runs `scam-list-gcn-100k-v1`: one published, frozen GCN and its
100,000-event dataset, normalized graph, preprocessing, split and scoring rule.
There are no small/medium/large benchmark variants. Submissions do not train,
retrain, select weights or prepare the model manually. The harness supplies its
frozen inputs and weights automatically.

| Test metric | Frozen plaintext baseline |
|---|---:|
| Recall (primary) | 0.897783 |
| F1 (primary) | 0.915254 |
| Accuracy (secondary) | 0.993250 |

These are the verified September 2026 retraining results. The exact matching
[checkpoint and data](artifacts/scam-list-gcn-100k-v1/README.md) are included when
cloning. Large data files use lossless gzip; no manual decompression is needed.
The older miniature GCN is retained only under `tests/fixtures/` for regression
testing and is not a registered benchmark workload.

## Dataset and sensitive fields

Each synthetic transfer is an event node. The GCN receives eight numeric features.
Three must be encrypted: `transfer_amount_z`, `source_daily_total_amount_z` and
`prior_report_count_z`. These represent the transferred amount, the source
account's running daily transferred total and its prior complaint/report count.
Public inputs are payment-channel indicators, normalized daily transaction count,
normalized graph and weights. This is the chosen v1 boundary, not a claim that
other fields are non-sensitive in real deployments.

See [the dataset guide](docs/dataset.md) for the actual transaction-log head,
field definitions, column order, preprocessing and encryption boundary, and
[the model notes](docs/scam-list-gcn.md) for layers and baseline provenance.

## Copy the CKKS example and run

The name “toy CKKS” refers to a simple FHE implementation, **not a different GCN**.
Copy the entire example directory, including its helper and dependency file:

```bash
git clone https://github.com/AdSpaceEngineeer/fhe-gnn-anomaly-benchmark.git
cd fhe-gnn-anomaly-benchmark
python -m venv .venv
source .venv/bin/activate
python -c "import shutil; shutil.copytree('submissions/toy_ckks', 'submissions/my_method')"
python -m pip install -r requirements.txt -r submissions/my_method/requirements.txt
python harness/run_submission.py --submission my_method --threads 2
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`.
The copy command refuses to overwrite an existing `my_method` directory.
Supported environment: Python 3.10–3.12 on Linux or Windows.

The example uses TenSEAL's native Microsoft SEAL bindings, explicit TC128
validation, packed ciphertexts and the unchanged trained GCN. Parameters, encoding
and packing live inside the submission; no harness or weight edits are needed.

**Resource/validation note:** the 100,000-event encrypted workload is substantial.
Key generation can take minutes and consume several GiB; full inference can take
much longer. This implementation is deliberately not performance-tuned. Code,
packing/algebra and interface checks do not establish a completed full-workload
encrypted run: none is claimed for this revision. Live CKKS tests are manual-only.
Coordinate resource use on shared machines and do not treat the example as a
quick laptop demo.

`--threads` is an upper budget; this simple native implementation is largely
single-threaded. The default timeout is 86,400 seconds **per stage** and can be
changed with `--timeout-seconds`. Raising it does not reserve memory or disk.

## Check the frozen baseline without encryption

```bash
python harness/run_submission.py --submission plaintext_debug --debug-plaintext --threads 2
```

This runs the same full workload, verifies checksums/reference scores, and
calculates quality on the 20,000 fixed test nodes. It needs only core dependencies,
not PyTorch or a retraining step. It is not an FHE result.

## Execution and measurements

All stages run locally in separate processes:

1. Generate a fresh key set and check security parameters.
2. Client encodes/encrypts the three sensitive feature columns.
3. Evaluator computes encrypted scores using the public graph and frozen weights.
4. Client decrypts/decodes the scores.
5. Harness verifies scores, applies the fixed threshold and records metrics.

There is no model-preparation stage. Any FHE-specific weight encoding/packing
occurs inside the adapter and remains included in its measured inference time.
Keys are reused across repeats within one invocation and replaced on the next.
The evaluator is not given the secret key. This is a logical client/server
boundary, not an OS sandbox for malicious code.

Each new `measurements/<run-id>/` directory contains:

- `report.json`: machine-readable measurements and verification.
- `comparison.md`: compact plaintext-versus-submission quality table, FHE overhead
  and optional timing detail. Unmeasured values appear as a dash, not zero.

The main metrics include Recall/F1, Accuracy, numerical score error, prediction
agreement, stage wall time, throughput, peak RAM, key/ciphertext/intermediate
storage, and communication bytes. Network-transfer and key-rotation durations
are not measured. See [measurement definitions](docs/measurements.md).

Submissions may additionally write
`intermediate_dir/server_reported_steps.json`:

```json
{"Encrypted computation": 12.3, "I/O": 1.4, "Total": 13.7}
```

Like BERT, this is a flat dictionary of named durations in seconds. These are
clearly labelled **server-reported** observations, never replacements for the
harness's measurements. The example reports its native serialization/file I/O
separately; the harness also measures its own input/output file access.

Share `report.json` and `comparison.md`, **not `io/`**: that directory contains
client secrets and plaintext test inputs. Backend/hardware disclosure is optional;
security-critical parameters/evidence are mandatory. Hardware is collected only
with `--include-hardware`.

## Implement your own method

Either modify your copied CKKS example or start from `submissions/template/`.
Implement `describe`, `keygen`, `encrypt`, `evaluate` and `decrypt`; put your
dependencies, security parameters, encoding and packing in your submission.
Complete its README. Do not change the harness, frozen model, graph or threshold.

The [submission contract](docs/submission-contract.md) specifies all inputs,
outputs and security evidence. Numerical PASS, security approval and the registered
workload are separate checks. Novel schemes need evidence review even if their
outputs pass. We recommend using the published weights/checksum as-is.

## Repository structure

```text
harness/        Fixed runner, validation, measurements and comparison tables
artifacts/      One frozen trained workload and checksum registry
submissions/
  toy_ckks/     Copyable CKKS example plus packed-arithmetic helper
  template/     Scheme-independent starting point
  plaintext_debug/  No-encryption pipeline check
docs/           Dataset, model, interface and measurement explanations
scripts/        Maintainer-only training/export and internal fixture generation
tests/          Integrity, interfaces, algebra and optional encrypted tests
examples/       Clearly labelled historical/example reports
```

## Tests and troubleshooting

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Ordinary tests do not run encrypted inference. The GitHub live-CKKS job is opt-in
through a manual workflow input; it is not run on every push. See the
[example README](submissions/toy_ckks/README.md) for optional integration checks.

Missing backend: install your submission's requirements. Invalid security
parameters: correct the submission without lowering the 128-bit requirement.
Existing output directory: choose a new `--out`; runs are never overwritten.
Checksum failure: do not silently edit the manifest. Timeout: inspect stage logs
before increasing the limit. A failed or untested encrypted run is not a benchmark result.
