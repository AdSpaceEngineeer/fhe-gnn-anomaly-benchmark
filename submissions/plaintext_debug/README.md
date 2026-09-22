# Plaintext pipeline check

No encryption is performed. Requires `--debug-plaintext`; reports always set
`eligible_for_comparison` to false. Uses the exact frozen reference equations.

```bash
python harness/run_submission.py --submission plaintext_debug --debug-plaintext
```
