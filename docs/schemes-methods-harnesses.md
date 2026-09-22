# Schemes, methods and harnesses

The executable integration contract is now in
[submission-contract.md](submission-contract.md). Run implementations through
`harness/run_submission.py`; do not use the earlier local correctness-runner draft.

- `submissions/template/`: scheme-independent integration entry point.
- `submissions/toy_ckks/`: real CKKS arithmetic demonstration with parameter checks.
- `submissions/plaintext_debug/`: non-encrypted pipeline test.
- `requirements.txt`: core dependencies, independent of the submission backend.

Schemes may use polynomial or LUT approximations, packing optimizations,
bootstrapping or native runtimes, provided the frozen inference target and
minimum 128-bit classical security requirement are satisfied. Evidence for novel
schemes needs review; model correctness alone does not demonstrate security.

This release does not contain OpenFHE or Concrete implementations, nor a claim
that the tiny scalar CKKS example scales to the original 100,000-node graph.
