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
workflow. It reuses the metrics and stage-separation philosophy documented by
the [FHE Benchmarking Suite](https://github.com/fhe-benchmarking/fhe-benchmarking.github.io)
and its [ML-inference workload](https://github.com/fhe-benchmarking/ml-inference),
not their implementation or repository structure.

The use case is grounded in:

- Mastercard's IMDA PET Sandbox proof of concept for checking encrypted IBANs
  against cross-border financial-crime intelligence.
- Intesa Sanpaolo and IBM's prototype for validating encrypted cryptocurrency
  counterparty details against a blacklist.
- Google's homomorphic-encryption-based Private Set Membership deployment for
  privacy-preserving Chrome OS device enrolment.
- The UK government's PET use-case repository records HE-based financial-crime
  work by AUSTRAC, Duality, and Enveil.

These are evidence of demand and experimentation, not evidence that FHE is
already routine production infrastructure.

## 3. Why a GNN workload

Fraud and financial crime are relational. Accounts, devices, merchants,
beneficiaries, phone numbers, and transactions form networks in which
coordinated behaviour may be more informative than isolated records. NVIDIA
documents an end-to-end card–merchant fraud workflow using relational GNN
embeddings. CARE-GNN applies relation-aware message passing to YelpChi fraud
detection, and a systematic review of 33 studies reports GNN applications
across banking, payments, insurance, cryptocurrency, and money laundering.

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
A canonical CARE-GNN-format `YelpChi.mat` must contain `features`, `label`, and
the homogeneous adjacency matrix `homo`. Preparation exports three separated
artifacts: public topology/features, harness-only labels/splits, and client-only
synthetic identifiers. The raw dataset is not redistributed by this repository.
A canonical archive is available from the
[DGFraud dataset directory](https://github.com/safe-graph/DGFraud/tree/master/dataset);
the benchmark records and verifies its SHA-256 in the prepared metadata.
A second industry graph dataset remains to be selected. IBM AMLSim is a leading
candidate because it creates scalable synthetic account-transaction graphs
with known laundering patterns.

Synthetic IBAN-form identifiers are assigned deterministically to YelpChi
nodes. Each node's group is the minimum node index in its closed one-hop public
neighbourhood; labels are never consulted. The generator:

- preserve an IBAN-like alphanumeric structure;
- support controlled prefix and suffix reuse within predefined relational
  groups;
- avoid direct label tokens and access to validation/test labels; and
- record its seed and configuration for reproducibility.

The canonical YelpChi identifier length is `L=22`. General benchmark profiles
accept one equal length per run in the inclusive range 2 through 64 characters.
The bound prevents an undeclared change in workload scale; longer identifiers
require a new named profile and reporting rationale.

Random independent identifiers are invalid because they contain no relational
signal and make the truncation experiment meaningless. The implemented grouping
is a predeclared label-free process.

Splits are deterministic, class-stratified 40/20/40 train/validation/test. A
single instance selects one test anomaly for latency measurement. Batch
instances sample the test class ratio and include every incoming one-hop
neighbour of each target. Proposed target-batch variants are 1, 100, 1000, and
10000, pending final FHE profiling on the canonical artifact. The complete
18,384-node test split is a separate mandatory quality configuration; a backend
may internally chunk it, but it must return one score per fixed test node.

## 6. Model and computation

Training is plaintext and excluded from timing. The reference is a one-hop
mean-aggregation message-passing GNN. For node `v`:

```text
z_v = W_self x_v + W_neighbour mean(x_u : u -> v) + b
h_v = z_v + 0.125 z_v^2
logit_v = w_out^T h_v + b_out
```

The published bundle fixes weights, feature normalization, degree-2 activation,
output head, and a validation-selected decision threshold. Every result records
the model adapter and artifact SHA-256. The server returns encrypted logits;
sigmoid and thresholding occur after client decryption. This is a deliberately
small, FHE-oriented GNN, not GCN and not a claim of state-of-the-art YelpChi
quality.

Using the published bundle unchanged is recommended for direct comparisons.
The harness also accepts a separately trained compatible bundle through the
`polynomial_message_passing_v1` adapter, evaluates its Recall/F1/Accuracy and
Q(k) automatically, and records its distinct checksum. Additional architectures
can be supported by adding explicit versioned adapters rather than silently
changing model semantics. Submissions may change the FHE implementation,
parameters, packing, and evaluation plan.

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
claimed security level, key policy, and any additional leakage. Comparable FHE
submissions must claim at least 128-bit security.

## 8. Protocol and quality metric

For full length `L`, the canonical sweep tests total retained characters `k` at
nominal retention fractions `{0.2, 0.4, 0.6, 0.8, 1.0}`. The harness runs all
five by default and produces a machine-readable summary plus a Q(k) SVG. Custom
retention points are supplemental. Retain `ceil(k/2)` prefix characters and
`floor(k/2)` suffix characters. Implementations report the realized `k/L` after
rounding.

Characters use the alphabet `0-9A-Z`. The canonical logical representation is
one encrypted 36-way one-hot vector per retained character plus a public
prefix/suffix position class. Homomorphic summation yields a fixed 72-feature
prefix/suffix histogram normalized by `L`. Packing may differ, but logical
values and resulting model inputs must match this definition.

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

Primary quality tables use all 18,384 fixed YelpChi test nodes. Metrics from the
single-target latency configuration are diagnostic and must not replace the
full-test Recall, F1, or Accuracy. Systems tables use the declared batch-size
variants and report them separately.

The harness executes three measured runs by default and stores one schema-valid
JSON file per run. It measures stage wall time independently. Submission-reported
peak RAM and fine-grained internal timings are supplementary, never substitutes
for harness timing. Fixed artifact directories determine exact byte counts.

Key policy is either `reused` or `ephemeral_per_batch`. Ephemeral submissions
must implement key rotation and rotated evaluation-key upload before every run
after the first; those costs and bytes are reported separately. Reused-key setup
is amortised across the measured runs.

## 10. Outputs and non-goals

Outputs are an executable harness, synthetic-identifier generator, published
model bundle and weights, correctness tests, reporting schema, measured
plaintext baseline, a real TenSEAL/CKKS test drive, and standardized result
files for future FHE submissions.
Training under FHE, encrypted topology, private linkage, full graph
construction, and prescribing CKKS/BFV/TFHE are non-goals.

The bundled plaintext submission is only a protocol exerciser and is marked
non-FHE. The published YelpChi bundle and plaintext quality report under
`baselines/yelpchi/` are the recommended reference. The TenSEAL submission is a
real open-source CKKS implementation; its checked-in batch-1 measurements are
a functional and systems smoke test, not the mandatory full-test quality run.

## References

- FHE ML inference: https://github.com/fhe-benchmarking/ml-inference
- IMDA PET Sandbox: https://www.imda.gov.sg/how-we-can-help/data-innovation/privacy-enhancing-technology-sandboxes
- IBM/Intesa Sanpaolo: https://www.ibm.com/case-studies/blog/intesa-sanpaolo-ibm-secure-digital-transactions-fhe
- Google Private Set Membership: https://security.googleblog.com/2021/10/protecting-your-device-information-with.html
- UK government PET finance use cases: https://www.gov.uk/guidance/repository-of-privacy-enhancing-technologies-pets-use-cases/finance-and-insurance
- NVIDIA GNN fraud workflow: https://developer.nvidia.com/blog/optimizing-fraud-detection-in-financial-services-with-graph-neural-networks-and-nvidia-gpus/
- CARE-GNN: https://arxiv.org/abs/2008.08692
- CARE-GNN code and YelpChi format: https://github.com/YingtongDou/CARE-GNN
- Motie and Raahemi, 2024: https://doi.org/10.1016/j.eswa.2023.122156
- IBM AMLSim: https://github.com/IBM/AMLSim
- TenSEAL: https://github.com/OpenMined/TenSEAL
- Microsoft SEAL: https://github.com/microsoft/SEAL
