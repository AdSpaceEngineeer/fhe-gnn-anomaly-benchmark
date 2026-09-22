# Scam_List_GCN frozen trained baseline v1

This is the newly frozen September 2026 retraining run, supplying a new checkpoint
after the earlier run's artifacts became unavailable. This bundle publishes the
new run's exact data, weights and independently checked scores.
It uses the agreed generator, seed 42, 100,000 events, a 20,000-account pool and
80 training epochs. The architecture has widths 8/64/32/64/8 and activation
`p(z) = z + 0.125*z*z`; the final decoder layer is linear.

## Integrity and provenance

- Manifest SHA256: `0e019cfc4c7c09b207b6f1e1e4c1c259f611a811394b1cc046603202b7dd4860`.
- Original checkpoint SHA256: `21eeec488f8cba7dcd66c49021bd163b4eea926a1cc990c22e2d84f6485ebf51`.
- Training/export source commit: [`6c64bc60861b72652fb223215d48d197e73e58ac`](https://github.com/AdSpaceEngineeer/fhe-gnn-anomaly-benchmark/commit/6c64bc60861b72652fb223215d48d197e73e58ac).
- `environment.txt`: exact package snapshot from training/export; Python 3.12.14, CPU PyTorch 2.6.0. This is provenance, not the recommended core install file.
- `baseline_metrics.json`: unmodified training report and configuration.

The JSON weights were verified exactly against the checkpoint tensors. All
100,000 float64 reference scores were recomputed from the frozen graph, features
and weights. Published gzip files decompress to the exact original manifest-hashed
bytes. No feature, weight, split, threshold or score was changed for packaging.

## Data and results

The log contains 100,000 events and 4,062 synthetic anomalies. Split sizes are
64,000 training, 16,000 validation and 20,000 test nodes. Training loss uses the
61,400 normal training nodes; the forward pass is transductive over the full graph.
There are 370,316 unique undirected edges (740,632 off-diagonal sparse entries),
plus 100,000 self-loops after normalization.

The three sensitive features are `transfer_amount_z`,
`source_daily_total_amount_z` and `prior_report_count_z` (columns 4, 6, 7).
See [dataset definitions and head](../../docs/dataset.md). The score is their mean
squared reconstruction error. Classification uses the fixed validation threshold
`4.344182877864071`, not a threshold chosen by the submission.

| Split | Recall | F1 | Accuracy |
|---|---:|---:|---:|
| Validation | 0.900000 | 0.908385 | 0.992625 |
| Test | 0.897783 | 0.915254 | 0.993250 |

The 80-epoch final checkpoint is retained, even though some earlier epochs had
higher validation F1. The independent float64 reference and original float32
training report are both preserved. High synthetic-data scores do not establish
real-world scam-detection effectiveness.

## Run without retraining

From the repository root, with core requirements installed:

```bash
python harness/run_submission.py --submission plaintext_debug --debug-plaintext --artifacts artifacts/scam-list-gcn-100k-v1 --threads 2
```

No manual decompression or PyTorch installation is necessary for that command.
It validates the bundle and runs plaintext inference, reporting metrics on the
fixed test nodes. It is not an FHE result. The toy CKKS submission is deliberately
too small for this bundle; full-size FHE submissions remain research work.
