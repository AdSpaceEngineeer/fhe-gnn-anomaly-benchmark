# Plaintext diagnostic submission

This adapter supports both workloads and implements the file/stage interface
without encryption. It uses the same reference mathematics as the verifier.
Use it to check installation, data loading and reporting, not to measure FHE.

```bash
python harness/run_submission.py --workload tam --submission plaintext_debug --debug-plaintext --out measurements/tam-plaintext
```

No dependencies beyond the root `requirements.txt` are needed. Each stage
reports its configured numerical-library thread limit; OS threads are sampled
separately by the harness. No secret keys are generated.
