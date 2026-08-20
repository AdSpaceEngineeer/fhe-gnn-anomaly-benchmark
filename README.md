# FHE GNN Anomaly Benchmark

An industry-oriented benchmark for graph anomaly detection with FHE-protected
identifiers such as IBANs, phone numbers, and wallet IDs.

> **Status:** executable research prototype. YelpChi preparation, deterministic
> identifier injection, a trainable plaintext GNN, staged submission execution,
> result validation, published YelpChi weights, a measured plaintext baseline,
> a non-FHE protocol dry run, and a real TenSEAL/CKKS test-drive backend are
> implemented.

## Research question

Can encrypted identifier-derived features support useful graph-based fraud,
scam, and anti-money-laundering inference without exposing identifiers to the
evaluating server, and what systems cost does that privacy impose?

The benchmark reuses the *measurement philosophy* of the
[HomomorphicEncryption.org FHE Benchmarking Suite](https://github.com/fhe-benchmarking/fhe-benchmarking.github.io):
separate key generation, encryption, encrypted server compute, decryption,
postprocessing, and quality checks. It does not copy or depend on that suite's
ML-inference harness.

## Workload

- **Task:** inference-only binary node anomaly scoring on YelpChi.
- **Sensitive input:** balanced prefix/suffix encodings of deterministic,
  synthetic, mod-97-valid GB IBAN-like node identifiers.
- **Public input:** one-hop graph topology, non-identifier features, published
  weights, batch shape, and retained identifier length.
- **GNN:** one-hop mean-aggregation message passing with separate self and
  neighbour weights, a degree-2 activation `h = z + 0.125 z²`, and a linear
  anomaly-logit head. Sigmoid and thresholding occur after decryption.
- **Primary quality:** anomaly Recall and F1. Accuracy is secondary.
- **Systems measures:** stage and online latency, throughput, peak RAM, artifact
  storage, directional communication, and key lifecycle overhead.
- **FHE policy:** implementation- and scheme-agnostic, with a minimum claimed
  security target of 128 bits for comparable FHE submissions.

Training is plaintext and excluded from benchmark timing. For directly
comparable published results, the repository recommends using the supplied
model bundle, weights, checksum, graph instance, split, identifier seed, and
decision threshold as-is. A differently trained compatible model can instead
be registered as a bundle; the harness automatically evaluates it and records
its checksum so the result is clearly identified.

## Published baseline

The recommended bundle is [the YelpChi model manifest](baselines/yelpchi/model.json),
which resolves to `model.npz` with
SHA-256 `78f0bfa62d4bf88c49b82667596ed8e41ca70ac70ae77a17c1dd9d29954a4337`.
On the 18,384-node test split, its full-identifier plaintext baseline is Recall
`0.597305`, F1 `0.522422`, and Accuracy `0.841275`. See the
[baseline report and truncation sweep](baselines/yelpchi/README.md).

Every result records the model name, adapter, artifact hash, quality metrics,
and systems metrics.

## Identifier Truncation Robustness

For full identifier length `L` and total retained prefix/suffix characters `k`:

```text
Q(k) = min(Recall_protected(k) / Recall_plaintext(L),
           F1_protected(k)     / F1_plaintext(L))
```

Requested retention levels are `{0.2, 0.4, 0.6, 0.8, 1.0}`. Because `k` is an
integer, every result records both the requested fraction and realized `k/L`.
The `sweep` command runs all five by default and writes summary JSON plus an SVG
plot. User-supplied points are permitted as supplemental experiments. The
benchmark reports Q(k) with latency, storage, and communication rather than
declaring a universal quality-loss threshold.

The YelpChi profile fixes `L=22`. The general harness accepts equal-length
alphanumeric identifiers from 2 through 64 characters; a new profile is needed
outside that bound so resource growth remains explicit.

## Installation and commands

```bash
python -m pip install -e ".[yelpchi]"

# YelpChi.zip is external. It can be read directly without extraction.
# Canonical source: https://github.com/safe-graph/DGFraud/tree/master/dataset
fhe-gnn-benchmark prepare-yelpchi /path/to/YelpChi.zip ./prepared/yelpchi

# Reproducing training is optional verification, not part of FHE comparison.
fhe-gnn-benchmark evaluate-baseline ./prepared/yelpchi \
  ./baselines/yelpchi/model.json ./reproduced-baseline.json

# Register separately trained weights that use the built-in GNN adapter.
fhe-gnn-benchmark bundle-model ./my-model/model.npz ./my-model/model.json \
  --name my-compatible-gnn

# This bundled submission is only a protocol test and provides no privacy.
fhe-gnn-benchmark run \
  submissions/plaintext_reference/submission.json \
  ./prepared/yelpchi ./baselines/yelpchi/model.json ./results \
  --retention 0.4 --batch-size 100 --num-runs 3

fhe-gnn-benchmark validate ./results/measurements/batch-100/k-9/results-1.json

# Run the default five retention points and generate Q(k) JSON/SVG.
fhe-gnn-benchmark sweep \
  submissions/plaintext_reference/submission.json \
  ./prepared/yelpchi ./baselines/yelpchi/model.json ./sweep-results \
  --batch-size 100 --num-runs 3

# Verify that result files use the same declared workload.
fhe-gnn-benchmark validate-comparison result-from-scheme-a.json result-from-scheme-b.json
```

For the optional real CKKS backend, install `.[tenseal]` and replace the
submission manifest above with `submissions/tenseal_ckks/submission.json`.
Its checked-in batch-1 smoke results are under `results/tenseal_ckks/smoke/`.

The standard proposed target-batch variants are `1`, `100`, `1000`, and
`10000`, subject to final YelpChi profiling. A batch instance contains its
targets plus their complete incoming one-hop context, so inference preserves
the frozen one-layer GNN semantics. Primary Recall/F1/Accuracy comparisons use
the complete 18,384-node test split (`--batch-size 18384`); quality numbers from
the single-target latency case are diagnostic only.

## Submission boundary

A `submission.json` maps language-neutral stage names to argument-array
commands. The harness executes commands without a shell and independently
measures wall time and fixed artifact paths. A stage may write the JSON file at
`FHE_BENCH_STAGE_REPORT` to report peak RAM and internal timing.

Key policy is explicit:

- `reused`: initial key setup is amortised across repeated runs.
- `ephemeral_per_batch`: key rotation and rotated evaluation-key upload stages
  are mandatory and measured before every run after the first.

See [the specification](docs/benchmark-spec.md),
[submission contract](docs/submission-contract.md), and
[result schema](schemas/result.schema.json).

## Repository layout

```text
docs/          Benchmark specification and executable contract
baselines/     Published model bundle and measured plaintext quality
schemas/       Result schema v0.4
src/           Dataset, identifier, model, harness, metrics, and CLI code
submissions/   Plaintext protocol exerciser and TenSEAL/CKKS backend
results/       Small checked-in test-drive reports (not ciphertext artifacts)
tests/         Unit and end-to-end subprocess tests
```

## Evidence base

- [IMDA–Mastercard FHE cross-border financial-crime POC](https://www.imda.gov.sg/-/media/imda/files/programme/pet-sandbox/imda-pet-sandbox--case-study--mastercard.pdf)
- [IBM–Intesa Sanpaolo encrypted transaction-validation prototype](https://www.ibm.com/case-studies/blog/intesa-sanpaolo-ibm-secure-digital-transactions-fhe)
- [UK government PET use-case repository: AUSTRAC, Duality, and Enveil](https://www.gov.uk/guidance/repository-of-privacy-enhancing-technologies-pets-use-cases/finance-and-insurance)
- [NVIDIA card–merchant GNN fraud workflow](https://developer.nvidia.com/blog/optimizing-fraud-detection-in-financial-services-with-graph-neural-networks-and-nvidia-gpus/)
- [CARE-GNN paper and YelpChi reference implementation](https://github.com/YingtongDou/CARE-GNN)
- [Systematic review of GNNs for financial fraud detection](https://doi.org/10.1016/j.eswa.2023.122156)

These sources establish industry experimentation and a relational modeling
rationale. They do not establish routine production deployment or prove that a
GNN is universally superior to every non-graph detector.

## Development

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

## License

Apache-2.0. See [LICENSE](LICENSE).
