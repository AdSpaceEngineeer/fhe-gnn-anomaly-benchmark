# Scam_List_GCN plaintext baseline

`Scam_List_GCN` is the v1 plaintext baseline for the FHE/GNN benchmark. It
generates a synthetic scam transaction log, converts events into graph nodes,
trains a simplified GCN attribute autoencoder, and saves frozen model artifacts
for later FHE inference comparison.

Training is plaintext and is not part of future FHE timing.

For a submitter-focused explanation of raw fields, preprocessing, model column
order and the encryption boundary, see [the dataset guide](dataset.md).

## Dataset head

| event_id | timestamp | source_account | destination_account | payment_channel | transfer_amount | source_daily_txn_count | source_daily_total_amount | prior_report_count | scam_label |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 100001 | 2026-01-01 00:00:20+00:00 | ACC-007069 | ACC-008124 | bank_transfer | 9.82 | 1 | 9.82 | 0 | 0 |
| 100002 | 2026-01-01 00:00:38+00:00 | ACC-019945 | ACC-006841 | wallet | 67.82 | 1 | 67.82 | 0 | 0 |
| 100003 | 2026-01-01 00:00:40+00:00 | ACC-015803 | ACC-009010 | wallet | 18.14 | 1 | 18.14 | 0 | 0 |
| 100004 | 2026-01-01 00:00:51+00:00 | ACC-003754 | ACC-001379 | wallet | 79.94 | 1 | 79.94 | 0 | 0 |
| 100005 | 2026-01-01 00:01:06+00:00 | ACC-015121 | ACC-012577 | bank_transfer | 24.76 | 1 | 24.76 | 0 | 0 |

## Data fields

| Field | Definition | Role |
|---|---|---|
| `event_id` | Unique transaction or interaction event key | Row key |
| `timestamp` | Event time | Used to derive daily behaviour fields |
| `source_account` | Account initiating the event | Used to build event graph |
| `destination_account` | Account receiving the event | Used to build event graph |
| `payment_channel` | Channel such as wallet, bank transfer, card, or instant pay | Public feature after encoding |
| `transfer_amount` | Value transferred in the event | Sensitive feature |
| `source_daily_txn_count` | Count of source-account events so far that day | Public feature |
| `source_daily_total_amount` | Running daily amount for the source account | Sensitive feature |
| `prior_report_count` | Prior complaint/report count for the source account | Sensitive feature |
| `scam_label` | Ground-truth event label, `1` for scam/anomaly | Evaluation only |

The graph connects up to three neighboring events in each time-sorted source
account or destination account sequence. This is an event-count window, not a
fixed number of hours. This creates the `Network` artifact. Encoded public and
sensitive columns create `Attributes`. `scam_label` creates `Label`.

## Sensitive fields for FHE inference

| Raw field | Encoded feature | Why selected |
|---|---|---|
| `transfer_amount` | `transfer_amount_z` | Sensitive transaction value |
| `source_daily_total_amount` | `source_daily_total_amount_z` | Sensitive transaction-behaviour aggregate |
| `prior_report_count` | `prior_report_count_z` | Sensitive complaint/risk-intelligence history |

These fields are encoded into numeric model features first, then encrypted for
FHE inference. The first benchmark version keeps graph topology public.

## Simplified GCN layers

| Layer / stage | Input | Operation | Output |
|---|---|---|---|
| Input feature matrix | Event-node features `X` | Encode raw log into public and sensitive numeric features | `X in R^(N x 8)` |
| Normalized adjacency | Event graph `A` | Add self-loops and compute `D^(-1/2)(A+I)D^(-1/2)` | `A_norm` |
| Encoder GCN layer 1 | `X`, `A_norm` | `activation(A_norm X W1 + b1)` | `H1` |
| Encoder GCN layer 2 | `H1`, `A_norm` | `activation(A_norm H1 W2 + b2)` | `Z` |
| Attribute decoder GCN layer 1 | `Z`, `A_norm` | `activation(A_norm Z W3 + b3)` | `Hd` |
| Attribute decoder GCN layer 2 | `Hd`, `A_norm` | `A_norm Hd W4 + b4` | `X_hat` |
| Anomaly scoring | `X`, `X_hat` | Reconstruction error over selected sensitive feature columns | Event anomaly scores |

Default activation:

```text
activation(z) = z + 0.125z^2
```

## Plaintext baseline result

These are the verified results of the newly frozen September 2026 retraining run,
published as [`scam-list-gcn-100k-v1`](../artifacts/scam-list-gcn-100k-v1/README.md).
The older reported run's checkpoint was unavailable; this release does not claim
to recover it. The same generator configuration, seed and architecture were used.
The internal miniature regression fixture does not reproduce these numbers.

```text
events=100000
adjacency_off_diagonal_entries=740632
unique_undirected_edges=370316
scam_rate=0.04062
features=8
sensitive=['transfer_amount_z', 'source_daily_total_amount_z', 'prior_report_count_z']
validation_threshold=4.344182877864071
```

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 0.992625 | 0.916928 | 0.900000 | 0.908385 | 0.997928 | 0.957636 |
| Test | 0.993250 | 0.933419 | 0.897783 | 0.915254 | 0.998697 | 0.970152 |

The original float32 training report is retained as `baseline_metrics.json`.
The benchmark uses the independently evaluated float64 scores in `reference.json.gz`
with exactly the same frozen float32 weights and threshold. Their classification
metrics agree; numerical rounding may differ. Scores are the **mean** squared
reconstruction error across sensitive columns `[4, 6, 7]`.

The 80-epoch final checkpoint is frozen, not the highest validation-F1 checkpoint
observed along the way. Do not select a different epoch or retune the threshold
when comparing submissions. The baseline has been checked in plaintext; no
full-graph FHE performance is reported.

## Maintainer training command (not required for benchmark submissions)

```bash
python -m pip install -r requirements-training.txt
python scripts/scam_list_gcn.py --outdir runs/scam_list_gcn --num-events 100000 --num-accounts 20000 --epochs 80
```

## Training output artifacts

The following are generated by maintainer training. The published inference
bundle combines features, graph, preprocessing and splits into `data.json.gz`;
the original checkpoint and JSON weights are both included. See the
[artifact guide](../artifacts/README.md) for the clone-and-run files.

| File | Meaning |
|---|---|
| `transactions.csv` | Synthetic transaction log |
| `features.npy` | Encoded GCN feature matrix |
| `labels.npy` | Event scam labels |
| `network_adjacency.npz` | Sparse event graph |
| `network_adjacency_normalized.npz` | GCN-normalized graph |
| `splits.json` | Train/validation/test node indices |
| `scam_list_gcn.pt` | Frozen model weights and metadata |
| `baseline_metrics.json` | Recall, F1, Accuracy, ROC-AUC, average precision, checksum |
| `anomaly_scores.csv` | Event-level anomaly scores |
