# Implementation status

| Task | Status |
|---|---|
| 1. Publish frozen workload | Complete: newly frozen `scam-list-gcn-100k-v1` data, trained checkpoint/weights, threshold, splits and verified reference scores |
| 2. Submission/security contract | Implemented; minimum 128-bit classical security |
| 3. Submission template | Implemented |
| 4. Inference-only workload modules | Implemented and checked against PyTorch equations |
| 5. Shared helper files | Implemented |
| 6. Verification/metrics | Implemented; measurement limitations documented |
| 7. Main runner | Implemented as `harness/run_submission.py` |
| 8. Toy CKKS example | Implemented and tested through all four GCN layers |
| 9. Dependency files | Core, development, training and CKKS dependencies separated |
| 10. README quickstart/tests | Implemented; Windows local validation and Linux/Windows CI configured |

All ten implementation tasks are complete. The previously unavailable checkpoint
was replaced through an explicitly approved retraining run, now frozen as the
published baseline. Its provenance is documented; it is not presented as recovery
of the older checkpoint. No training is needed by benchmark submitters.

Remaining research work includes full-size FHE submissions and measurement of
their overhead. The CKKS example is intentionally a small, unoptimized arithmetic
test, not an implementation validated on the 100,000-node trained graph.
