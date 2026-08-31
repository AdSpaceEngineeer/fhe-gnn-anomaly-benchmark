# Scam_List_GCN Plaintext Baseline

`Scam_List_GCN` is the agreed plaintext baseline for the next benchmark
direction: event-level scam detection on a synthetic transaction graph. It is
not an FHE implementation. It generates the dataset, trains the simplified GCN,
and saves frozen weights and baseline metrics for later FHE evaluation.

## Dataset shape

The raw log is intentionally closer to a transaction/interactions table than
to already-encoded GCN features:

| Field | Definition | Role |
|---|---|---|
| `event_id` | Unique transaction or interaction event | Row identifier |
| `timestamp` | Event time | Used to derive daily velocity fields |
| `source_account` | Account initiating the event | Builds graph edges |
| `destination_account` | Account receiving the event | Builds graph edges |
| `payment_channel` | Channel such as wallet, bank transfer, card, or instant pay | Public feature after encoding |
| `transfer_amount` | Value transferred in the event | Sensitive numeric feature |
| `source_daily_txn_count` | Count of source-account events so far that day | Public numeric feature |
| `source_daily_total_amount` | Running daily amount for the source account | Sensitive numeric feature |
| `prior_report_count` | Prior complaint/report count for the source account | Sensitive numeric feature |
| `scam_label` | Ground-truth event label, `1` for scam/anomaly | Evaluation only |

Example `df.head()`:

| event_id | timestamp | source_account | destination_account | payment_channel | transfer_amount | source_daily_txn_count | source_daily_total_amount | prior_report_count | scam_label |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 100001 | 2026-01-01 00:27:44+00:00 | ACC-000309 | ACC-000056 | bank_transfer | 16.57 | 1 | 16.57 | 1 | 0 |
| 100002 | 2026-01-01 00:35:37+00:00 | ACC-000124 | ACC-000329 | wallet | 32.16 | 1 | 32.16 | 0 | 0 |
| 100003 | 2026-01-01 00:42:46+00:00 | ACC-000209 | ACC-000181 | bank_transfer | 39.92 | 1 | 39.92 | 0 | 0 |
| 100004 | 2026-01-01 00:42:52+00:00 | ACC-000326 | ACC-000373 | bank_transfer | 18.03 | 1 | 18.03 | 0 | 0 |
| 100005 | 2026-01-01 00:58:19+00:00 | ACC-000469 | ACC-000261 | card | 37.98 | 1 | 37.98 | 0 | 0 |

## Mapping to GCN inputs

| Raw artifact | Simplified GCN artifact | Definition |
|---|---|---|
| `source_account`, `destination_account` | `Network` | Event-event graph: two event nodes connect when they share a source or destination account within a small temporal window |
| `payment_channel`, `source_daily_txn_count` | Public `Attributes` | Encoded public feature columns |
| `transfer_amount`, `source_daily_total_amount`, `prior_report_count` | Sensitive `Attributes` | Normalized numeric columns later intended for encryption in the FHE benchmark |
| `scam_label` | `Label` | Held for validation/test scoring only |

The first FHE benchmark should encrypt only these normalized feature columns:

| Raw field | Encoded feature |
|---|---|
| `transfer_amount` | `transfer_amount_z` |
| `source_daily_total_amount` | `source_daily_total_amount_z` |
| `prior_report_count` | `prior_report_count_z` |

Account identifiers are sensitive in a real system, but in this baseline they
are used to construct the plaintext graph and are not model features. Encrypting
graph topology is a future benchmark variant.

## Simplified GCN

The model is a DOMINANT-inspired attribute autoencoder, simplified for a future
FHE workload by dropping full adjacency reconstruction.

| Layer / stage | Input | Operation | Output |
|---|---|---|---|
| Input feature matrix | Node features `X` | Public features plus normalized sensitive features | `X` |
| Normalized adjacency | Event graph `A` | Add self-loops and compute `D^-1/2 (A + I) D^-1/2` | `A_norm` |
| Encoder GCN layer 1 | `X`, `A_norm` | `activation(A_norm X W1 + b1)` | `H1` |
| Encoder GCN layer 2 | `H1`, `A_norm` | `activation(A_norm H1 W2 + b2)` | `Z` |
| Attribute decoder GCN layer 1 | `Z`, `A_norm` | `activation(A_norm Z W3 + b3)` | `Hd` |
| Attribute decoder GCN layer 2 | `Hd`, `A_norm` | `A_norm Hd W4 + b4` | `X_hat` |
| Anomaly scoring | `X`, `X_hat` | Reconstruction error over the sensitive feature columns | Event anomaly score |

Default activation is `poly2`, defined as `h = z + 0.125 z^2`, so the
plaintext baseline already follows an FHE-compatible activation shape.

## Future FHE operation mapping

| Increasing Overhead | Ciphertext Operation | Role in GCN | Equation | Operation to Benchmark | Adapter method |
|---:|---|---|---|---|---|
| 1 | Ciphertext-plaintext addition | Merge encrypted and public feature paths | `Enc(X_s W_s) + X_p W_p` | Add plaintext tensor to ciphertext tensor | `add_plain(ct, pt)` |
| 2 | Ciphertext-ciphertext addition | Neighbor aggregation and score reduction | `sum_j c_ij Enc(h_j)`, `sum_m Enc(e_im^2)` | Add ciphertext tensors | `add(ct1, ct2)` |
| 3 | Ciphertext-plaintext scalar multiplication | Apply normalized graph weights | `c_ij Enc(h_j)` | Multiply ciphertext by plaintext scalar | `mul_plain(ct, pt)` |
| 4 | Ciphertext-plaintext matrix multiplication | Encoder and decoder projection | `Enc(X_s) W_s`, `Enc(H1) W_d` | Multiply ciphertext tensor by plaintext weight matrix | `matmul_plain(ct, W)` |
| 5 | Ciphertext-ciphertext subtraction | Reconstruction difference | `Enc(X_s) - Enc(X_hat_s)` | Subtract ciphertext tensors | `sub(ct1, ct2)` |
| 6 | Ciphertext-ciphertext multiplication | Squared reconstruction error | `Enc(e)^2` | Multiply ciphertext by ciphertext | `square(ct)` or `mul(ct1, ct2)` |
| 7 | Polynomial or LUT nonlinear approximation | Replace GCN activation under FHE | `p_sigma(Enc(Z))` | Apply encrypted activation approximation | `activation(ct, kind="poly_relu")` |

## Running on JupyterLab

Use a virtual environment. The script supports older server stacks, including
Python 3.6 and Torch 1.4.

```bash
python -m pip install -r requirements-scam-list-gcn.txt
python scripts/scam_list_gcn.py --outdir runs/scam_list_gcn --num-events 100000 --num-accounts 20000 --epochs 80
```

If the script is uploaded directly into a JupyterLab home directory rather than
inside the repository, run:

```bash
python scam_list_gcn.py --outdir runs/scam_list_gcn --num-events 100000 --num-accounts 20000 --epochs 80
```

## Outputs

| File | Meaning |
|---|---|
| `transactions.csv` | Raw synthetic transaction log |
| `features.npy` | Encoded GCN feature matrix |
| `labels.npy` | Event scam labels |
| `network_adjacency.npz` | Sparse event graph |
| `network_adjacency_normalized.npz` | GCN-normalized graph |
| `splits.json` | Train/validation/test node indices |
| `scam_list_gcn.pt` | Frozen trained model weights and metadata |
| `baseline_metrics.json` | Validation/test Recall, F1, Accuracy, ROC-AUC, and model checksum |
| `anomaly_scores.csv` | Event-level anomaly scores |

For the eventual benchmark, training remains out of scope. Train once, publish
the dataset artifact, frozen weights, checksum, and baseline metrics, then let
FHE submissions run the same inference path.
