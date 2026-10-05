# Scam_List_GCN dataset

Each row is a synthetic transfer event and becomes a graph node. The frozen
dataset contains 100,000 events, 4,062 anomalies and 19,995 observed accounts
from a 20,000-account pool. Splits contain 64,000 train, 16,000 validation and
20,000 test nodes. Reconstruction training uses 61,400 normal training nodes.

## Transaction preview

First five rows from the published log; every row below has daily count 1, daily
total equal to its transfer amount, prior-report count 0 and scam label 0.

| event_id | timestamp UTC | source_account | destination_account | payment_channel | transfer_amount |
|---:|---|---|---|---|---:|
| 100001 | 2026-01-01 00:00:20 | ACC-007069 | ACC-008124 | bank_transfer | 9.82 |
| 100002 | 2026-01-01 00:00:38 | ACC-019945 | ACC-006841 | wallet | 67.82 |
| 100003 | 2026-01-01 00:00:40 | ACC-015803 | ACC-009010 | wallet | 18.14 |
| 100004 | 2026-01-01 00:00:51 | ACC-003754 | ACC-001379 | wallet | 79.94 |
| 100005 | 2026-01-01 00:01:06 | ACC-015121 | ACC-012577 | bank_transfer | 24.76 |

```python
import pandas as pd
df = pd.read_csv('artifacts/scam-list-gcn-relu-100k-v2/transactions.csv.gz')
print(df.head())
```

## Fields

| Field | Definition and format | Evaluator treatment |
|---|---|---|
| event_id | Unique integer event identifier | Not a feature |
| timestamp | UTC date/time | Used for graph ordering and daily totals; not a feature |
| source_account | Sender account string | Used for graph construction; string excluded |
| destination_account | Beneficiary account string | Used for graph construction; string excluded |
| payment_channel | wallet, bank_transfer, card or instant_pay | Public one-hot columns 0..3 |
| transfer_amount | Positive transfer value in synthetic currency units | Encrypted normalized column 4 |
| source_daily_txn_count | Sender's transfers so far that UTC day, including current event | Public normalized column 5 |
| source_daily_total_amount | Sender's cumulative transfer value that day, including current event | Encrypted normalized column 6 |
| prior_report_count | Simulated earlier complaints associated with sender; nonnegative integer | Encrypted normalized column 7 |
| scam_label | Synthetic ground truth, 0/1 | Evaluation only; excluded from model inputs |

The three sensitive numeric fields represent individual financial activity,
aggregated transfer behaviour and complaint history. Their model names are
`transfer_amount_z`, `source_daily_total_amount_z`, `prior_report_count_z`.
The two monetary fields use log1p; all four numeric fields then use training-fitted
z-scores. The means/scales are frozen. Scheme-specific encoding and encryption
are performed by the submission; no raw account string is encrypted as a feature.

## Graph and privacy boundary

Within each timestamp-sorted source-account sequence and destination-account
sequence, events link to up to three preceding/following events. Edges are
undirected and duplicates collapse. Cross-role account equality alone creates
no edge. The graph has 370,316 undirected edges (740,632 symmetric nonzeros)
before self-loops; normalization is D^(-1/2)(A+I)D^(-1/2).

The graph and remaining features are public by benchmark policy. They may still
reveal sensitive relationships or correlated information in a real system.
All data here are synthetic. Evaluation is transductive, not chronological.

The encrypted output is one mean squared reconstruction-error score per event.
The client decrypts it; the harness applies the frozen threshold and test labels.
See [Model](scam-list-gcn.md) and [TAM dataset](dataset-tam.md).
