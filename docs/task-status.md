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
| 8. Copyable toy CKKS submission | Revised packed implementation targets the fixed trained workload; software/algebra checks only, live CKKS validation pending |
| 9. Dependency files | Core, development, training and CKKS dependencies separated |
| 10. README quickstart/tests | Implemented; Windows local validation and Linux/Windows CI configured |

All ten implementation tasks are complete. The previously unavailable checkpoint
was replaced through an explicitly approved retraining run, now frozen as the
published baseline. Its provenance is documented; it is not presented as recovery
of the older checkpoint. No training is needed by benchmark submitters.

Optional BERT-style server timings and `comparison.md` now accompany the primary
JSON measurements. There is no separate model-preparation stage or workload-size
menu. The old miniature graph is an internal software fixture only.

Remaining validation includes a completed encrypted run of the revised packed
CKKS example. A live check was stopped during key generation at the user's
request; no success or FHE performance result is claimed for that revision.
