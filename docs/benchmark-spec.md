# Benchmark specification, version 1

## Objective and industry grounding

Measure encrypted GCN inference for scam, crime and fraud prevention using
sensitive numeric transaction features. Graph relationships represent repeated
source/beneficiary activity. This is a synthetic industry analogy, not a claim
of deployment performance or regulatory certification.

Ma et al.'s 2023 GNN fraud-detection survey motivated the relational model choice.
Ding's [DOMINANT implementation](https://github.com/kaize0409/GCN_AnomalyDetection_pytorch)
motivated the attributed graph autoencoder. This benchmark simplifies that design
to attribute reconstruction only: there is no structure decoder or adjacency
reconstruction loss. Synthetic transaction events replace the source repository's
citation/social-network data.

## Workload

A fixed, transductive graph with eight numeric features and four GCN layers:

```text
A_norm = D^(-1/2) (A + I) D^(-1/2)
H1 = p(A_norm X W1 + b1)
H2 = p(A_norm H1 W2 + b2)
H3 = p(A_norm H2 W3 + b3)
X_hat = A_norm H3 W4 + b4
p(z) = z + z^2/8
s_i = (1/3) sum_{m in S} (X_hat[i,m] - X[i,m])^2
predicted_scam_i = (s_i >= fixed_validation_threshold)
```

The input is the full frozen graph, even when metrics use only test nodes.
Training excludes test labels from the loss but sees their node features/graph
through the transductive forward pass. This is not a chronological held-out
deployment evaluation. The synthetic generator deliberately makes anomalies
distinguishable; high scores do not establish real-world fraud generalization.

The original widths are 8/64/32/64/8. The independent toy uses 8/2/2/2/8 and
initialized weights. A toy result is not a result on the trained baseline.

## Data and encryption boundary

The normalized sensitive fields are transferred amount, running daily transferred
total, and prior report count. Amounts receive log1p before standardization;
report count is standardized directly. Scaler statistics are frozen from the
training split. Payment channel (four indicators), daily transaction count,
graph topology and weights are public under the v1 threat model.

The graph links events sharing the same source account or the same destination
account, connecting up to three preceding/following events within each sorted
account sequence. This is a count window, not a fixed elapsed-time window;
cross-role matches are not added by the original generator. Raw account strings
are graph-construction inputs, not encrypted numeric features.

Sensitivity is a chosen confidentiality boundary. Real deployments may also need
to protect graph links, public behavioral features, preprocessing statistics or
model weights; v1 does not promise that protection. All released data are synthetic.

## Fixed comparison policy

Use the same dataset, normalized graph, weights, preprocessing, activation target,
sensitive columns, threshold and splits. Numerical approximations/packing
optimizations must be declared; they may not change the target model.
At least 128-bit classical security is required. The submission contract explains
evidence review, key handling and the actual-context check for the example.

The original 100,000-event run's matching frozen bundle is awaiting import.
Historical reported metrics remain in the model notes. The runnable toy bundle is
versioned and independently labelled; it is not a replacement training run.

## Measurements and execution

See [the contract](submission-contract.md), [measurement definitions](measurements.md)
and the root quickstart. Recall/F1 and numerical agreement are measured separately
from latency, throughput, memory, storage and communication. Training is never
included. The measurement categories follow
[the FHE Benchmarking Suite](https://fhe-benchmarking.github.io/).

## Ciphertext operations

| Operation | Workload role |
|---|---|
| Ciphertext + plaintext | Bias and public feature contribution |
| Ciphertext + ciphertext | Graph/feature sums and score reduction |
| Ciphertext x plaintext | Frozen graph coefficients and model weights |
| Matrix multiplication | Linear GCN projections, possibly decomposed into scalar products |
| Ciphertext - ciphertext | Sensitive feature reconstruction difference |
| Ciphertext square | Squared reconstruction error |
| Polynomial activation | Three hidden nonlinear transformations |

These operations do not have a universal overhead ranking: packing, rotations,
multiplicative depth, parameter sizes and backend determine cost.
