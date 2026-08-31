# Technical scope: FHE for GNN scam/fraud anomaly benchmark

## 1. Objective and industry motivation

The objective is to define an industry-use benchmark for fully homomorphic
encryption (FHE) in graph neural-network inference for scam, crime, and fraud
prevention.

The benchmark should answer:

```text
Can a graph anomaly detector run over encrypted sensitive transaction-risk
features while preserving fraud/scam detection quality, and what latency,
throughput, memory, storage, communication, and key-management overhead does
that privacy boundary impose?
```

The workload is industry-facing because fraud and scam risk is not usually
isolated to one transaction row. Risk often appears through repeated
counterparties, payment channel choice, transaction velocity, amount patterns,
complaint history, and graph neighbourhood structure.

The benchmark borrows the measurement philosophy of
[fhe-benchmarking.github.io](https://fhe-benchmarking.github.io/): report both
model utility and FHE systems cost, with clear separation between encryption,
encrypted compute, decryption, postprocessing, and quality checks.

## 2. Choice of GCN

The starting implementation pattern is Ding's
[GCN_AnomalyDetection_pytorch](https://github.com/kaize0409/GCN_AnomalyDetection_pytorch).
That repository is useful because its data abstraction is simple and close to
what this benchmark needs:

| Ding-style artifact | Meaning for this benchmark |
|---|---|
| `Network` | Event-event graph adjacency |
| `Attributes` | Event-node feature matrix |
| `Label` | Event anomaly/scam label for evaluation |

Ma et al. 2023 survey graph neural networks for fraud detection and support the
broader claim that fraud detection often benefits from graph-structured models.
For v1, the selected architecture is not a full state-of-the-art fraud system.
It is a simplified DOMINANT-style GCN attribute autoencoder selected because it
creates a useful FHE operation mix:

- normalized graph aggregation;
- plaintext-weighted matrix multiplication;
- ciphertext/plaintext additions;
- ciphertext subtraction;
- ciphertext squaring for reconstruction error;
- a polynomial activation.

`Scam_List_GCN`:

| Layer / stage | Input | Operation | Output |
|---|---|---|---|
| Input feature matrix | Event-node features `X` | Encode raw transaction log into numeric public/sensitive features | `X in R^(N x 8)` |
| Normalized adjacency | Event graph `A` | `A_norm = D^(-1/2)(A+I)D^(-1/2)` | `A_norm` |
| Encoder GCN layer 1 | `X`, `A_norm` | `activation(A_norm X W1 + b1)` | `H1` |
| Encoder GCN layer 2 | `H1`, `A_norm` | `activation(A_norm H1 W2 + b2)` | `Z` |
| Attribute decoder GCN layer 1 | `Z`, `A_norm` | `activation(A_norm Z W3 + b3)` | `Hd` |
| Attribute decoder GCN layer 2 | `Hd`, `A_norm` | `A_norm Hd W4 + b4` | `X_hat` |
| Anomaly scoring | `X`, `X_hat` | `s_i = mean_m((x_im - xhat_im)^2), m in S` | Event anomaly score `s_i` |

The default activation is:

```text
activation(z) = z + 0.125z^2
```

## 3. Choice of dataset

The v1 dataset is synthetic but transaction-log-shaped. Each row is a
transaction or interaction event. Each event becomes a graph node. Edges are
created between events that share a source or destination account within a
small temporal window.

This preserves the useful abstraction from Ding's GCN implementation:

```text
transaction log -> Network, Attributes, Label
```

Example `df.head()`:

| event_id | timestamp | source_account | destination_account | payment_channel | transfer_amount | source_daily_txn_count | source_daily_total_amount | prior_report_count | scam_label |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 100001 | 2026-01-01 00:00:20+00:00 | ACC-007069 | ACC-008124 | bank_transfer | 9.82 | 1 | 9.82 | 0 | 0 |
| 100002 | 2026-01-01 00:00:38+00:00 | ACC-019945 | ACC-006841 | wallet | 67.82 | 1 | 67.82 | 0 | 0 |
| 100003 | 2026-01-01 00:00:40+00:00 | ACC-015803 | ACC-009010 | wallet | 18.14 | 1 | 18.14 | 0 | 0 |
| 100004 | 2026-01-01 00:00:51+00:00 | ACC-003754 | ACC-001379 | wallet | 79.94 | 1 | 79.94 | 0 | 0 |
| 100005 | 2026-01-01 00:01:06+00:00 | ACC-015121 | ACC-012577 | bank_transfer | 24.76 | 1 | 24.76 | 0 | 0 |

Sensitive fields selected for future encryption:

| Raw field | Encoded feature | Regulatory / confidentiality rationale | Used in GCN computation |
|---|---|---|---|
| `transfer_amount` | `transfer_amount_z` | Transaction value can reveal financial behaviour and commercial risk | Yes: feature projection and reconstruction score |
| `source_daily_total_amount` | `source_daily_total_amount_z` | Aggregated account behaviour can reveal transaction velocity and customer activity | Yes: feature projection and reconstruction score |
| `prior_report_count` | `prior_report_count_z` | Complaint or investigation history is sensitive risk intelligence | Yes: feature projection and reconstruction score |

Public v1 fields:

- `payment_channel`, after one-hot encoding;
- `source_daily_txn_count`, after normalization;
- event graph topology;
- frozen model weights and threshold.

Labels are used only for validation/test metrics.

## 4. GCN performance baseline

Training is plaintext and out of benchmark scope. The plaintext run establishes
the fixed model and the baseline metrics that future FHE submissions compare
against.

Baseline run summary:

```text
events=100000
edges=740632
scam_rate=0.0406
features=8
sensitive=['transfer_amount_z', 'source_daily_total_amount_z', 'prior_report_count_z']
validation_threshold=4.344190838049405
```

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 0.992625 | 0.916928 | 0.900000 | 0.908385 | 0.997928 | 0.957634 |
| Test | 0.993250 | 0.933419 | 0.897783 | 0.915254 | 0.998697 | 0.970149 |

The script writes the dataset, feature matrix, graph, split, weights,
threshold, checksum, metrics, and anomaly scores. Future FHE comparisons should
use these artifacts unchanged.

## 5. Benchmark metrics

Model performance metrics:

| Metric | Role |
|---|---|
| Anomaly Recall | Primary metric; missing scams/fraud is costly |
| Anomaly F1 | Primary metric; balances recall with false positives |
| Accuracy | Secondary metric because class imbalance can make it misleading |
| ROC-AUC | Diagnostic ranking metric |
| Average precision | Diagnostic metric for rare-event detection |
| Score error vs plaintext | Measures approximation/encryption-induced deviation |

FHE overhead metrics:

| Metric | What to report |
|---|---|
| Latency | Wall-clock time for single and batch inference |
| Throughput | Events scored per second |
| Memory consumption | Peak RAM during encryption, encrypted compute, and decryption |
| Storage requirements | Public keys, evaluation keys, ciphertexts, intermediates, encrypted outputs |
| Communication complexity | Client-to-server and server-to-client bytes |
| Key overhead | Key generation, evaluation-key size/upload, rotation cost, amortised key cost |

Ciphertext operation table for the future FHE path:

| Increasing Overhead | Ciphertext Operation | Role in GCN | Equation | Operation to Benchmark | Adapter method |
|---:|---|---|---|---|---|
| 1 | Ciphertext-plaintext addition | Merge encrypted and public feature paths | `Enc(X_s W_s) + X_p W_p` | Add plaintext tensor to ciphertext tensor | `add_plain(ct, pt)` |
| 2 | Ciphertext-ciphertext addition | Neighbor aggregation and score reduction | `sum_j c_ij Enc(h_j)`, `sum_m Enc(e_im^2)` | Add ciphertext tensors | `add(ct1, ct2)` |
| 3 | Ciphertext-plaintext scalar multiplication | Apply normalized graph weights | `c_ij Enc(h_j)` | Multiply ciphertext by plaintext scalar | `mul_plain(ct, pt)` |
| 4 | Ciphertext-plaintext matrix multiplication | Encoder and decoder projection | `Enc(X_s) W_s`, `Enc(H) W` | Multiply ciphertext tensor by plaintext weight matrix | `matmul_plain(ct, W)` |
| 5 | Ciphertext-ciphertext subtraction | Reconstruction difference | `Enc(X_s) - Enc(X_hat_s)` | Subtract ciphertext tensors | `sub(ct1, ct2)` |
| 6 | Ciphertext-ciphertext multiplication | Squared reconstruction error | `Enc(e)^2` | Multiply ciphertext by ciphertext | `square(ct)` or `mul(ct1, ct2)` |
| 7 | Polynomial or LUT nonlinear approximation | GCN activation under FHE | `p_sigma(Enc(Z))` | Apply encrypted activation approximation | `activation(ct, kind="poly_relu")` |

## 6. Schemes, methods, and harnesses

Reserved for future implementation.

This section will be developed later. It should eventually define how FHE
engineers plug in:

- CKKS, BFV, TFHE, or other schemes;
- polynomial or lookup-table activation methods;
- ciphertext packing strategies;
- key policies;
- stage-level harness commands;
- result schemas and validation rules.

For now, the only fixed rule is that direct comparisons should preserve the
same dataset, graph, model weights, checksum, threshold, and inference
semantics.
