# Scam_List_TAM dataset

This is a fully synthetic T-Finance-inspired account graph, not a copy or renamed
version of real T-Finance. The original dataset describes anonymized accounts,
transaction relationships and ten features related to registration, login and
interaction activity ([dataset authors](https://proceedings.mlr.press/v162/tang22b/tang22b.pdf)).
The field names below are this benchmark's definitions, not verified original
T-Finance column names.

There are 39,357 accounts, 1,803 anomalies (4.58%) and 353,623 undirected edges.
Splits contain 25,188 train, 6,297 validation and 7,872 test accounts. One
feature-dependent truncation retains 310,328 undirected encoder edges. This is
much sparser than the real T-Finance graph reported in TAM.

## Account preview

Selected columns from the first five published rows; fractions rounded here only.

| account_id | account_age_days | login_count_30d | interaction_count_30d | cross_border_partner_fraction_30d | anomaly_label |
|---|---:|---:|---:|---:|---:|
| ACC-000000 | 1496 | 15 | 45 | 0.045455 | 0 |
| ACC-000001 | 1555 | 94 | 28 | 0.250000 | 0 |
| ACC-000002 | 2001 | 30 | 52 | 0.105263 | 0 |
| ACC-000003 | 531 | 33 | 50 | 0.100000 | 0 |
| ACC-000004 | 1750 | 38 | 46 | 0.157895 | 0 |

```python
import pandas as pd
df = pd.read_csv('artifacts/scam-list-tam-synthetic-v1/accounts.csv.gz')
print(df.head())
```

## Numeric model fields

All 30-day fields cover the preceding 30 days at the synthetic snapshot.

| Column | Raw field | Meaning | Evaluator receives |
|---:|---|---|---|
| 0 | account_age_days | Days since registration; integer >=30 in this generator | Ciphertext |
| 1 | login_count_30d | Successful login count | Plaintext |
| 2 | interaction_count_30d | Transfers sent plus received, including repeats | Ciphertext |
| 3 | failed_login_count_30d | Failed login attempts | Plaintext |
| 4 | active_login_days_30d | Distinct days with successful login | Plaintext |
| 5 | distinct_login_devices_30d | Distinct devices with successful login | Plaintext |
| 6 | days_since_last_login | Days since latest successful login | Plaintext |
| 7 | unique_counterparties_30d | Distinct transfer partners | Plaintext |
| 8 | days_since_last_transfer | Days since most recent transfer | Plaintext |
| 9 | cross_border_partner_fraction_30d | Foreign-country distinct partners / all distinct partners | Ciphertext |

Counts/ages are nonnegative numeric values; the fraction is in [0,1]. Columns
0..8 use log1p; column 9 is unchanged before all ten receive training-fitted
z-score standardization. Model names append `_z`. Encrypt columns [0,2,9] in
that order. The fraction is already computed: submissions do not compute it by
dividing encrypted raw transfer counts.

`account_id`, `country_code` and `anomaly_label` are not numeric model features.
The country code is used only during synthetic preparation. `transfer_edges.csv.gz`
stores sender, recipient, transfer count and recency for each directed pair;
repeated transfers determine interaction counts. Raw counts and country codes
are not supplied to the evaluator through the adapter interface.

## Generation and graph

Three latent usage communities generate correlated activity. Anomalies combine
overlapping behavioural changes and cross-community transfer relationships.
The raw undirected graph links accounts with any transfer record. The encoder
uses one frozen NSGT-style truncation; cosine scoring uses the original graph.
Both inference graphs include self-loops. Public topology and truncation may
reveal correlated information; protecting three columns is not complete account
privacy. Labels are synthetic, not investigator-confirmed real cases.

The output is one encrypted anomaly score/account. See [Model](scam-list-tam.md)
for normalization, scoring, threshold and the frozen plaintext baseline.
