# Frozen model internals and observed ranges

These are plaintext float64 ranges over every node in the published synthetic
datasets, including train, validation and test nodes. They describe the frozen
reference computations, not prescribed FHE approximation intervals. Encrypted
approximation and encoding errors can move later intermediates outside these
ranges. Rounded endpoints below are observations, not certified bounds.

## Scam_List_GCN

100,000 event nodes; feature widths `8 -> 64 -> 32 -> 64 -> 8`.
Each layer computes `Z = Abar (H W) + b`, with bias added **after** aggregation.
`Abar = D^(-1/2) (A + I) D^(-1/2)` is the published normalized graph.
The first three layers apply `ReLU(Z) = max(0, Z)`; the final layer is linear.

| Layer | Input -> output width | Preactivation min | Preactivation max | Output min | Output max |
|---|---|---:|---:|---:|---:|
| encoder_1 | 8 -> 64 | -5.224792 | 3.002181 | 0 | 3.002181 |
| encoder_2 | 64 -> 32 | -1.474737 | 4.408591 | 0 | 4.408591 |
| decoder_1 | 32 -> 64 | -2.905365 | 2.810228 | 0 | 2.810228 |
| decoder_2 | 64 -> 8 | -1.545872 | 9.533169 | -1.545872 | 9.533169 |

Sensitive input columns (zero-based indices) are `transfer_amount_z` (4),
`source_daily_total_amount_z` (6), and `prior_report_count_z` (7).
The score is `s_i = (1/3) sum_{m in {4,6,7}} (X_im - Xhat_im)^2`.
The decoder reconstructs all eight columns; scoring uses only these three.
The fixed classification threshold is `4.11271162092986`.

Artifact: `scam-list-gcn-relu-100k-v2`.

- `weights.json` SHA256: `097ade05a13960d1049033871db26b46a5528f6be29c3d17315194d9964ad0f1`
- Manifest SHA256: `a5ec6d87b828a30450a02bf32c77af8de2b6f20a3cdc03e8a345a9b4c5ebaa06`

See [feature definitions](dataset.md), [architecture](scam-list-gcn.md), and
[reference inference](../harness/workloads/gcn.py).

## Scam_List_TAM

39,357 account nodes; feature widths `10 -> 64 -> 32`. Each layer computes
`Z = Abar_cut (H W) + b`, then `PReLU(Z) = Z` for nonnegative Z and `alpha Z`
otherwise. The learned scalar slopes are `0.24318136274814606` (layer 1) and
`0.28408777713775635` (layer 2).

| Layer | Input -> output width | Preactivation min | Preactivation max | Output min | Output max |
|---|---|---:|---:|---:|---:|
| gcn1 | 10 -> 64 | -2.774235 | 2.553466 | -0.674642 | 2.553466 |
| gcn2 | 64 -> 32 | -2.374445 | 1.181005 | -0.674551 | 1.181005 |

Normalize the final embedding with `q_i = sum_m H_im^2`,
`r_i = max(sqrt(q_i), 1e-12)` and `U_i = H_i / r_i`.

| Quantity | Observed min | Observed max |
|---|---:|---:|
| Squared norm q | 0.008369691 | 5.502063774 |
| Guarded norm r | 0.091486014 | 2.345647837 |

No frozen plaintext node activates the norm guard; the guard remains part of
the workload. This observation does not guarantee nonzero approximate norms.
The score is `s_i = 1 - U_i dot (sum_j B_ij U_j) / sum_j B_ij`.
The encoder uses the truncated normalized graph `Abar_cut`; scoring uses the
distinct untruncated binary graph `B` with self-loops. Do not interchange them.
The fixed classification threshold is `0.39311713209104254`.

Sensitive columns are `account_age_days_z` (0), `interaction_count_30d_z` (2)
and `cross_border_partner_fraction_30d_z` (9).
Artifact: `scam-list-tam-synthetic-v1`.

- `weights.json` SHA256: `ea6320f81f14ecb2907ef07864be71d8620c9cbc9ec0c4638bffa39fd055925c`
- Manifest SHA256: `0936868fabfbe3e1fe13b419ed15f01f047bcfa790126cb2af152d1f45f7eeaf`

See [feature definitions](dataset-tam.md), [architecture](scam-list-tam.md), and
[reference inference](../harness/workloads/tam.py).

## Reading and reproducing the ranges

Load the registered bundle with `harness.generate_input.load_bundle`, use its
full feature matrix and graph, and record min/max immediately before and after
each activation in the linked reference inference. For TAM, also record squared
norms and guarded norms before division. Use the supplied float64 weights without
retraining, clipping, sampling nodes or changing either graph. Small last-digit
differences can arise across numerical libraries. The [operation table](operations.md)
describes the dependent computations; approximation, packing and depth choices
remain the submitter's responsibility.
