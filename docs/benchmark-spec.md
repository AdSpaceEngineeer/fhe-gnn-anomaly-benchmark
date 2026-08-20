# Benchmark specification

## 1. Objective

Design an implementation-agnostic FHE benchmark for inference-time fraud,
scam, and AML detection using encrypted alphanumeric identifiers in a graph
neural network (GNN). The benchmark tests whether identifier-derived features
can preserve anomaly-event detection utility while remaining hidden from the
evaluator.

## 2. Motivation and industry grounding

Existing FHE neural-network workloads largely benchmark standalone encrypted
classification. This benchmark instead evaluates a recognizable client/server
workflow and follows the stage separation in the
[FHE ML-inference harness](https://github.com/fhe-benchmarking/ml-inference).

The use case is grounded in:

- Mastercard's IMDA PET Sandbox proof of concept for checking encrypted IBANs
  against cross-border financial-crime intelligence.
- Intesa Sanpaolo and IBM's prototype for validating encrypted cryptocurrency
  counterparty details against a blacklist.
- Google's homomorphic-encryption-based Private Set Membership deployment for
  privacy-preserving Chrome OS device enrolment.
- UK government records of HE-based financial-crime initiatives involving
  AUSTRAC, Duality, and Enveil.

These are evidence of demand and experimentation, not evidence that FHE is
already routine production infrastructure.

## 3. Why a GNN workload

Fraud and financial crime are relational. Accounts, devices, merchants,
beneficiaries, phone numbers, and transactions form networks in which
coordinated behaviour may be more informative than isolated records. NVIDIA
documents an end-to-end card–merchant fraud workflow using relational GNN
embeddings. A systematic review of 33 studies reports GNN applications across
banking, payments, insurance, cryptocurrency, and money laundering.

The benchmark therefore treats GNNs as a strong fit for relational risk. It
does not claim that GNNs are universally the best anomaly detector.

## 4. Research questions

1. How do encrypted identifier length and representation affect recall, F1,
   and FHE cost?
2. What overhead arises from encryption, evaluation, decryption, and ephemeral
   keys?
3. Does encrypted inference preserve the ranking and decisions of a fixed
   plaintext GNN?

## 5. Data and identifier injection

YelpChi is the primary dataset and is evaluated as binary node anomaly scoring.
A second industry graph dataset remains to be selected. IBM AMLSim is a leading
candidate because it creates scalable synthetic account-transaction graphs
with known laundering patterns.

Synthetic IBAN-form identifiers are assigned deterministically to YelpChi
nodes. The generator must:

- preserve an IBAN-like alphanumeric structure;
- support controlled prefix and suffix reuse within predefined relational
  groups;
- avoid direct label tokens and access to validation/test labels; and
- record its seed and configuration for reproducibility.

Random independent identifiers are invalid because they contain no signal and
make the truncation experiment meaningless. Any relationship between group
membership and anomaly prevalence must be generated from training-split
information or a predeclared label-free process.

## 6. Model and computation

Training is plaintext and excluded from timing. The benchmark fixes a trained,
FHE-compatible message-passing GNN, dataset splits, weights, activation
polynomials, and decision rule. Encrypted identifier features are processed by
fixed-neighbourhood message passing, ciphertext addition,
ciphertext–plaintext multiplication, rotations/packing, and specified
polynomial activations.

Submissions may change the FHE implementation, parameters, packing, and
evaluation plan. Changes to model semantics must be declared and reported as a
separate, non-comparable result.

## 7. Threat and encryption boundary

- The client holds the secret key.
- The client encrypts balanced prefix/suffix identifier encodings.
- The server receives evaluation keys and returns encrypted anomaly scores.
- The server may see graph topology, non-identifier features, frozen weights,
  batch shape, and retained identifier length.
- Network observers may see message sizes and timing; hiding those channels is
  out of scope.

Private graph construction, private set intersection, identifier linkage, and
topology hiding are excluded. The server is assumed honest-but-curious for the
initial benchmark. Submissions must disclose their FHE scheme, parameters,
claimed security level, and any additional leakage.

## 8. Protocol and quality metric

For full length `L`, test total retained characters `k` at nominal retention
fractions `{0.2, 0.4, 0.6, 0.8, 1.0}`. Retain `ceil(k/2)` prefix characters and
`floor(k/2)` suffix characters. Implementations must report the realized `k/L`
after rounding.

Let `R_E(k)` and `F1_E(k)` be encrypted-workload recall and F1, and let
`R_P(L)` and `F1_P(L)` be the full-identifier plaintext values:

```text
Q(k) = min(R_E(k) / R_P(L), F1_E(k) / F1_P(L)).
```

Report the Pareto frontier that maximizes `Q(k)` while minimizing retention,
latency, storage, and communication. With no prescribed quality-loss tolerance,
the curve's knee is descriptive and is not a unique acceptable truncation.
Also report plaintext-truncated results to separate truncation loss from FHE
approximation loss.

## 9. Required measurements

- Anomaly recall and anomaly F1 (primary); accuracy (secondary).
- Plaintext full, plaintext truncated, and FHE truncated quality.
- Per-stage and end-to-end wall-clock latency.
- Batch throughput and run-to-run dispersion.
- Peak RAM for client preprocessing/encryption, server evaluation, and client
  decryption.
- Public/evaluation key, ciphertext, intermediate, and output sizes.
- Client-to-server and server-to-client bytes.
- Key generation, evaluation-key upload, rotation, and amortized per-batch key
  overhead.
- Hardware, operating system, software versions, run counts, random seed, FHE
  scheme, parameters, and claimed security level.

## 10. Outputs and non-goals

Outputs are a forkable harness, synthetic-identifier generator, frozen model
interface, correctness tests, reporting schema, and plaintext/FHE baselines.
Training under FHE, encrypted topology, private linkage, full graph
construction, and prescribing CKKS/BFV/TFHE are non-goals.

## References

- FHE ML inference: https://github.com/fhe-benchmarking/ml-inference
- IMDA PET Sandbox: https://www.imda.gov.sg/how-we-can-help/data-innovation/privacy-enhancing-technology-sandboxes
- IBM/Intesa Sanpaolo: https://www.ibm.com/case-studies/blog/intesa-sanpaolo-ibm-secure-digital-transactions-fhe
- Google Private Set Membership: https://security.googleblog.com/2021/10/protecting-your-device-information-with.html
- NVIDIA GNN fraud workflow: https://developer.nvidia.com/blog/optimizing-fraud-detection-in-financial-services-with-graph-neural-networks-and-nvidia-gpus/
- Motie and Raahemi, 2024: https://doi.org/10.1016/j.eswa.2023.122156
- IBM AMLSim: https://github.com/IBM/AMLSim
