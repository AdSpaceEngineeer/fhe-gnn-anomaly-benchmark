# Implementation status

| Task | Status |
|---|---|
| 1. Publish original frozen workload | Awaiting original generated run folder; toy fixture and import utility complete |
| 2. Submission/security contract | Implemented; minimum 128-bit classical security |
| 3. Submission template | Implemented |
| 4. Inference-only workload modules | Implemented and checked against PyTorch equations |
| 5. Shared helper files | Implemented |
| 6. Verification/metrics | Implemented; measurement limitations documented |
| 7. Main runner | Implemented as `harness/run_submission.py` |
| 8. Toy CKKS example | Implemented and tested through all four GCN layers |
| 9. Dependency files | Core, development, training and CKKS dependencies separated |
| 10. README quickstart/tests | Implemented; Windows local validation and Linux/Windows CI configured |

To finish task 1, obtain the original output folder containing
`scam_list_gcn.pt`, `features.npy`, `labels.npy`,
`network_adjacency_normalized.npz`, `feature_schema.json`, `splits.json`,
`baseline_metrics.json` and `transactions.csv`.
The Python training script alone cannot recover the exact frozen model artifact.
No replacement model has been trained or presented as that original baseline.
