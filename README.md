# FHE GNN Anomaly Benchmark

## 1. Objective and industry motivation

This repository scopes a benchmark for fully homomorphic encryption (FHE) in
graph neural-network inference for scam, crime, and fraud prevention.

The benchmark asks a practical industry question: can a third-party evaluator
run a graph-based anomaly detector while sensitive transaction-risk fields stay
encrypted, and can we measure both detection quality and FHE system overhead in
a repeatable way?

The motivation follows the benchmarking style of
[fhe-benchmarking.github.io](https://fhe-benchmarking.github.io/): report model
utility together with latency, throughput, memory, storage, communication, and
key-management cost. The use case is industry-facing because scam and fraud
detection often depends on linked transaction events, repeated counterparties,
payment channels, velocity patterns, complaint history, and suspicious local
neighbourhoods.

## 2. Choice of GCN

The starting point is the attributed-graph anomaly-detection pattern in Ding's
[GCN_AnomalyDetection_pytorch](https://github.com/kaize0409/GCN_AnomalyDetection_pytorch):
a graph `Network`, node `Attributes`, and node `Label`, with anomaly detection
based on reconstructing node attributes.

Ma et al. 2023 survey GNN-based fraud detection and motivate graph models for
fraud settings where relational structure matters. For the benchmark, we choose
a simplified DOMINANT-style GCN attribute autoencoder because it exposes a
useful variety of FHE-relevant operations without making v1 too broad.

`Scam_List_GCN` layers:

| Layer / stage | Input | Operation | Output |
|---|---|---|---|
| Input feature matrix | Event-node features `X` | Encode raw transaction log into numeric features | `X in R^(N x 8)` |
| Normalized adjacency | Event graph `A` | Add self-loops and normalize: `D^(-1/2)(A+I)D^(-1/2)` | `A_norm` |
| Encoder GCN layer 1 | `X`, `A_norm` | `activation(A_norm X W1 + b1)` | `H1` |
| Encoder GCN layer 2 | `H1`, `A_norm` | `activation(A_norm H1 W2 + b2)` | `Z` |
| Attribute decoder GCN layer 1 | `Z`, `A_norm` | `activation(A_norm Z W3 + b3)` | `Hd` |
| Attribute decoder GCN layer 2 | `Hd`, `A_norm` | `A_norm Hd W4 + b4` | `X_hat` |
| Anomaly scoring | `X`, `X_hat` | Reconstruction error on selected sensitive columns | Event anomaly scores |

Default activation:

```text
activation(z) = z + 0.125 z^2
```

## 3. Choice of dataset

The v1 dataset is synthetic, transaction-log-shaped, and intentionally close to
the original GCN PyTorch repository's `Network`, `Attributes`, `Label` format.
Each transaction is an event node. Edges connect events that share a source or
destination account within a small temporal window.

Example `df.head()`:

| event_id | timestamp | source_account | destination_account | payment_channel | transfer_amount | source_daily_txn_count | source_daily_total_amount | prior_report_count | scam_label |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 100001 | 2026-01-01 00:00:20+00:00 | ACC-007069 | ACC-008124 | bank_transfer | 9.82 | 1 | 9.82 | 0 | 0 |
| 100002 | 2026-01-01 00:00:38+00:00 | ACC-019945 | ACC-006841 | wallet | 67.82 | 1 | 67.82 | 0 | 0 |
| 100003 | 2026-01-01 00:00:40+00:00 | ACC-015803 | ACC-009010 | wallet | 18.14 | 1 | 18.14 | 0 | 0 |
| 100004 | 2026-01-01 00:00:51+00:00 | ACC-003754 | ACC-001379 | wallet | 79.94 | 1 | 79.94 | 0 | 0 |
| 100005 | 2026-01-01 00:01:06+00:00 | ACC-015121 | ACC-012577 | bank_transfer | 24.76 | 1 | 24.76 | 0 | 0 |

Fields selected for future encryption:

| Raw field | Encoded feature used by GCN | Why selected |
|---|---|---|
| `transfer_amount` | `transfer_amount_z` | Sensitive transaction value; used directly in feature projection and reconstruction scoring |
| `source_daily_total_amount` | `source_daily_total_amount_z` | Sensitive velocity/behaviour aggregate; used directly in GCN inference |
| `prior_report_count` | `prior_report_count_z` | Sensitive complaint/investigative history; used directly in GCN inference |

Public fields in v1 include payment channel, source daily transaction count,
and graph topology. Labels are used only for evaluation.

## 4. GCN performance baseline

Training is plaintext and excluded from future FHE timing. The benchmark should
publish the generated dataset, graph, split, frozen weights, model checksum,
threshold, and plaintext baseline metrics. FHE submissions should run the same
inference path with the same frozen model for direct comparison.

Recent plaintext run:

```text
events=100000
edges=740632
scam_rate=0.0406
features=8
sensitive=['transfer_amount_z', 'source_daily_total_amount_z', 'prior_report_count_z']
```

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 0.992625 | 0.916928 | 0.900000 | 0.908385 | 0.997928 | 0.957634 |
| Test | 0.993250 | 0.933419 | 0.897783 | 0.915254 | 0.998697 | 0.970149 |

Validation-selected threshold: `4.344190838049405`.

To reproduce the plaintext baseline:

```bash
python -m pip install -r requirements-scam-list-gcn.txt
python scripts/scam_list_gcn.py --outdir runs/scam_list_gcn --num-events 100000 --num-accounts 20000 --epochs 80
```

## 5. Benchmark metrics

Model performance metrics:

- Anomaly Recall and Anomaly F1 as primary metrics.
- Accuracy as a secondary metric because scam/fraud data is imbalanced.
- ROC-AUC and average precision as diagnostics.
- Maximum and mean score error versus plaintext frozen-model inference.

FHE overhead metrics:

- Latency and throughput for single and batch inference.
- Peak RAM during encryption, encrypted compute, and decryption.
- Storage for public keys, evaluation keys, ciphertexts, intermediate values,
  and encrypted outputs.
- Communication complexity for client-to-server uploads and server-to-client
  result payloads.
- Key generation time, evaluation-key size, evaluation-key upload time,
  rotation cost, and amortised key overhead per batch.

Draft operation table for the future FHE path:

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

The eventual benchmark should allow FHE engineers to plug in their own scheme,
parameters, polynomial or LUT activation strategy, packing method, and execution
harness while keeping the dataset, graph, model weights, checksum, and inference
semantics fixed.

## Current repository contents

```text
README.md                         Six-section project overview
docs/benchmark-spec.md            Technical scope matching this README
docs/scam-list-gcn.md             Supporting detail for dataset/model baseline
docs/schemes-methods-harnesses.md Reserved placeholder for FHE implementation work
scripts/scam_list_gcn.py          Synthetic data + Scam_List_GCN baseline
requirements-scam-list-gcn.txt    Dependencies for the baseline script
LICENSE                           Apache-2.0 license
```