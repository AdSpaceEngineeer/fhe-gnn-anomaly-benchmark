# FHE GNN Anomaly Benchmark

An industry-oriented benchmark design for graph anomaly detection with
FHE-protected identifiers such as IBANs, phone numbers, and wallet IDs.

> **Status:** research scaffold. The specification, identifier generator,
> metrics, result schema, tests, and submission contract are present. No FHE
> backend or measured performance claims are included yet.

## Research question

Can encrypted identifier-derived features support useful graph-based fraud,
scam, and anti-money-laundering inference without exposing the identifiers to
the evaluating server, and what systems cost does that privacy impose?

The benchmark follows the stage separation used by the
[HomomorphicEncryption.org ML-inference harness](https://github.com/fhe-benchmarking/ml-inference):
key generation, client preprocessing and encryption, server-side homomorphic
evaluation, decryption, postprocessing, and quality checking.

## Scope

- **Primary task:** inference-only binary node anomaly scoring on YelpChi.
- **Sensitive input:** balanced prefix/suffix character encodings of synthetic
  IBAN-form node identifiers.
- **Public inputs:** graph topology, non-identifier features, frozen model
  weights, batch shape, and retained identifier length.
- **Primary quality metrics:** anomaly recall and anomaly F1.
- **Systems metrics:** latency, throughput, peak RAM, storage, communication,
  and ephemeral key-management overhead.
- **Implementation policy:** scheme-agnostic; CKKS, BFV, TFHE, or another FHE
  approach may be used if security parameters and model deviations are
  disclosed.

See [the benchmark specification](docs/benchmark-spec.md) and
[submission contract](docs/submission-contract.md).

## Identifier Truncation Robustness

For full identifier length `L` and retained characters `k`, define:

```text
Q(k) = min(recall_encrypted(k) / recall_plaintext(L),
           f1_encrypted(k)     / f1_plaintext(L))
```

The benchmark reports the Pareto frontier that maximizes `Q(k)` while
minimizing `k/L`, latency, storage, and communication. It does not declare a
unique acceptable truncation point unless a user supplies a quality-loss
tolerance.

## Repository layout

```text
docs/        Benchmark specification and submission contract
schemas/     Machine-readable result schema
src/         Identifier and metric reference utilities
tests/       Unit tests
```

## Local development

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

The Python utilities do not implement cryptography. They define shared,
testable preprocessing and evaluation semantics for future FHE submissions.

## Evidence base

- [IMDA–Mastercard FHE financial-crime case study](https://www.imda.gov.sg/-/media/imda/files/programme/pet-sandbox/imda-pet-sandbox--case-study--mastercard.pdf)
- [Intesa Sanpaolo and IBM FHE transaction validation](https://www.ibm.com/case-studies/blog/intesa-sanpaolo-ibm-secure-digital-transactions-fhe)
- [NVIDIA GNN fraud-detection workflow](https://developer.nvidia.com/blog/optimizing-fraud-detection-in-financial-services-with-graph-neural-networks-and-nvidia-gpus/)
- [Systematic review of GNNs for financial fraud detection](https://doi.org/10.1016/j.eswa.2023.122156)

## License

Apache-2.0. See [LICENSE](LICENSE).
