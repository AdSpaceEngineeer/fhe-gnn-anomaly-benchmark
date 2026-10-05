# Scam_List_TAM

Scam_List_TAM is a single-model, TAM-inspired account-anomaly workload. Its basis
is [Truncated Affinity Maximization for Graph Anomaly Detection](https://arxiv.org/abs/2306.00006)
and the authors' [implementation](https://github.com/mala-lab/TAM-master).
It is not a reproduction of the original TAM experiments or real T-Finance data.

## Model

| Layer | Equation | Width |
|---|---|---|
| GCN 1 | H1 = PReLU(Abar_cut X W1 + b1) | 10 -> 64 |
| GCN 2 | H2 = PReLU(Abar_cut H1 W2 + b2) | 64 -> 32 |
| Normalization | U_i = H2_i / max(norm2(H2_i), 1e-12) | 32 |
| Anomaly score | s_i = 1 - U_i dot (sum_j B_ij U_j) / sum_j B_ij | One score/account |

PReLU(z) is z for z>=0 and a*z otherwise. Each layer has one learned scalar a,
published with the weights. Abar_cut is the normalized, once-truncated encoder
graph with self-loops. B is the untruncated binary scoring graph with self-loops.
Do not substitute one graph for the other.

Normalization uses a standard zero-norm guard, not an FHE approximation. The
original frozen run had no guarded nodes (minimum embedding norm 0.091486).
The standard function, including the guard, remains part of reference inference.
The scoring implementation avoids a dense N x N similarity matrix using the
algebraically equivalent sparse neighbor sum. No global min-max normalization
is required; the fixed anomaly threshold is `0.39311713209104254`.

## Training and preparation

One seeded NSGT-style truncation is computed from standardized features before
training and frozen. An edge survives if either endpoint keeps it. Training uses
one model, 80 epochs, seed 42 and sampled nonedge-affinity regularization. Loss
is mean train-anchor `(1 - neighbor_affinity)` plus mean sampled nonedge cosine;
labels are not supplied to the model or loss. The final epoch is frozen.

Differences from TAM include a synthetic sparse graph, smaller embedding width,
one truncation/model rather than an ensemble, sampled regularization, and a
train/validation/test protocol. These affect detection results and preclude a
claim of reproducing or improving the original paper.

## Plaintext baseline

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 0.961410 | 0.963288 |
| ROC-AUC | 0.900347 | 0.927265 |
| Recall | 0.569444 | 0.598338 |
| F1 | 0.574431 | 0.599168 |
| Precision | 0.579505 | 0.600000 |
| Average Precision | 0.589466 | 0.640382 |

These are float64 frozen-reference metrics; small score-order differences from
the archived float32 training report can affect ROC-AUC/average precision in the
last decimals. Classification metrics match. Accuracy is influenced by the low
anomaly prevalence; the additional quality metrics should be considered with it.

TAM is labelled the higher computational challenge because encrypted evaluation
requires norm-dependent normalization in addition to piecewise activation.
This does not assert superior fraud detection or measured FHE runtime. See
[Dataset](dataset-tam.md) and [Operations](operations.md).
