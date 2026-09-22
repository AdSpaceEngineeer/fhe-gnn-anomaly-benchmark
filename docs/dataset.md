# Dataset guide: transaction log, model inputs and sensitive fields

Each row describes one synthetic transfer event. Each event becomes a GCN node;
edges link events with shared source or destination accounts. The model produces
one anomaly score per event by reconstructing its numeric features.

There are two distinct datasets in the project:

- **Frozen trained baseline `scam-list-gcn-100k-v1`:** 100,000 synthetic events
  from the newly frozen September 2026 retraining run. Its matching log, features,
  graph, weights and reference scores are included; see
  [the bundle](../artifacts/scam-list-gcn-100k-v1/README.md).
- **Runnable `toy-v1`:** three fixed numeric feature rows in
  [data.json](../artifacts/toy-v1/data.json), with initialized, untrained weights.
  It tests encrypted arithmetic. It has no corresponding raw transaction log
  and is not a sample extracted from the trained baseline.

All data are synthetic. “Sensitive” identifies the fields that submissions must
protect under the benchmark's chosen confidentiality boundary. It is not a claim
that the other fields would be non-sensitive in a real financial dataset.

## What the transaction log looks like

First five rows verified from the published `transactions.csv.gz` (lossless gzip):

| event_id | timestamp | source_account | destination_account | payment_channel | transfer_amount | source_daily_txn_count | source_daily_total_amount | prior_report_count | scam_label |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 100001 | 2026-01-01 00:00:20+00:00 | ACC-007069 | ACC-008124 | bank_transfer | 9.82 | 1 | 9.82 | 0 | 0 |
| 100002 | 2026-01-01 00:00:38+00:00 | ACC-019945 | ACC-006841 | wallet | 67.82 | 1 | 67.82 | 0 | 0 |
| 100003 | 2026-01-01 00:00:40+00:00 | ACC-015803 | ACC-009010 | wallet | 18.14 | 1 | 18.14 | 0 | 0 |
| 100004 | 2026-01-01 00:00:51+00:00 | ACC-003754 | ACC-001379 | wallet | 79.94 | 1 | 79.94 | 0 | 0 |
| 100005 | 2026-01-01 00:01:06+00:00 | ACC-015121 | ACC-012577 | bank_transfer | 24.76 | 1 | 24.76 | 0 | 0 |

The log has 100,000 rows and 4,062 synthetic scam labels. The generator configured
a pool of 20,000 accounts; 19,995 appear in the log. Frozen split sizes are 64,000
training, 16,000 validation and 20,000 test nodes. The default training loss uses
only the 61,400 normal training nodes. The full graph remains visible during the
transductive forward pass; this is not a chronological deployment evaluation.

To inspect the log in a notebook with pandas installed:

```python
import pandas as pd
df = pd.read_csv("artifacts/scam-list-gcn-100k-v1/transactions.csv.gz")
print(df.head())
```

Pandas decompresses it automatically. The benchmark itself uses the frozen
numeric features in `data.json.gz`, not newly encoded rows from this preview.

## Raw field definitions

| Field | Definition and plaintext format | Treatment in v1 |
|---|---|---|
| `event_id` | Unique event number; integer | Row identifier; not a numeric GCN feature |
| `timestamp` | Event time; UTC date/time | Used to order events and define daily totals; not a numeric GCN feature |
| `source_account` | Account initiating the transfer; synthetic string such as `ACC-007069` | Used to construct graph edges; raw string is not passed to the evaluator |
| `destination_account` | Beneficiary account; synthetic account string | Used to construct graph edges; raw string is not passed to the evaluator |
| `payment_channel` | `wallet`, `bank_transfer`, `card` or `instant_pay`; text category | Public model input after one-hot encoding |
| `transfer_amount` | Amount of this transfer; positive numeric value, two decimal places, unspecified synthetic currency units | **Sensitive: encrypt its normalized model feature** |
| `source_daily_txn_count` | Number of this source account's transfers so far on the UTC calendar day, including this event; positive integer | Public model input after standardization |
| `source_daily_total_amount` | Sum transferred by this source account so far on that day, including this event; numeric value | **Sensitive: encrypt its normalized model feature** |
| `prior_report_count` | Simulated number of earlier complaints/reports associated with the source account; nonnegative integer | **Sensitive: encrypt its normalized model feature** |
| `scam_label` | Synthetic ground truth: `1` = scam/anomaly, `0` = normal; integer | Used by the harness for quality metrics; not an inference input and not sent to the evaluator |

The generator assigns `prior_report_count` per account; it is not a score produced
by the GCN or a count obtained from a real complaints database. Daily counts and
totals are cumulative, so they do not include transfers later that day.

## Why these three fields are selected for encryption

| Sensitive field | What it could reveal in a real analyst's database |
|---|---|
| `transfer_amount` | An individual customer's transaction value and financial activity |
| `source_daily_total_amount` | Aggregated spending/transfers and patterns of account activity |
| `prior_report_count` | Complaint history and investigative or risk intelligence about an account |

Account identifiers and relationships can also be sensitive in practice. Version
one protects only the three numeric features above: it does not protect the graph
topology, public daily transaction count or frozen preprocessing statistics.
Removing account strings does not conceal the relationships represented by edges.

## What the GCN and the FHE adapter actually receive

The raw log is converted into an `N x 8` numeric matrix `X`. Monetary features
first use `log1p(value) = ln(1 + value)`; the four numeric features are then
standardized with the frozen training-split means and scales:

```text
z = (transformed_value - training_mean) / training_scale
```

Submitters use the published features and preprocessing settings without refitting
the scaler. The toy already supplies its numeric values directly and has no fitted
scaler. Feature order for the existing generator and toy is:

| Column (zero-based) | Model feature | Encoding from the raw field | Evaluator receives |
|---:|---|---|---|
| 0 | `channel_wallet` | Category indicator, 0 or 1 | Plaintext |
| 1 | `channel_bank_transfer` | Category indicator, 0 or 1 | Plaintext |
| 2 | `channel_card` | Category indicator, 0 or 1 | Plaintext |
| 3 | `channel_instant_pay` | Category indicator, 0 or 1 | Plaintext |
| 4 | `transfer_amount_z` | `log1p(transfer_amount)`, then standardize | **Ciphertext** |
| 5 | `source_daily_txn_count_z` | Standardize daily transaction count | Plaintext |
| 6 | `source_daily_total_amount_z` | `log1p(source_daily_total_amount)`, then standardize | **Ciphertext** |
| 7 | `prior_report_count_z` | Standardize prior report count | **Ciphertext** |

The harness reads the column indices from the frozen bundle's schema. In this
schema, `encrypt()` receives `X[:, [4, 6, 7]]` in that order. The adapter performs
its scheme-specific numeric encoding/packing and encryption. The evaluator gets
those ciphertexts, the five public feature columns, normalized graph and weights.
It does not receive plaintext sensitive features, raw account strings or labels
through the adapter interface.

These are real-valued inputs after preprocessing, not just binary flags. They
participate in the GCN's weighted graph aggregation, feature projections and
polynomial activations. The encrypted final output contains one score per event:

```text
s_i = mean((X_hat[i, [4, 6, 7]] - X[i, [4, 6, 7]]) ** 2)
```

The client decrypts the scores. The harness applies the frozen threshold and uses
the test labels to calculate Recall, F1 and the other quality metrics.

## How the graph is constructed

Events are sorted by timestamp. Within each source-account sequence and each
destination-account sequence, an event is linked to up to three earlier/later
events. Edges are undirected and duplicate links are collapsed. Matching an
account appearing as source in one event and destination in another does not,
by itself, create an edge in the original generator.

Self-loops are added and adjacency is normalized as `D^(-1/2)(A+I)D^(-1/2)`.
The normalized graph is public in v1. Keep it unchanged during inference;
slicing rows or rebuilding the graph can change the model's predictions.

See [the submission contract](submission-contract.md) for the full interface and
[model/baseline notes](scam-list-gcn.md) for layers and reported training results.
