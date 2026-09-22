# Submissions

For a ready-written CKKS starting point, copy the entire `toy_ckks/` directory to
`my_method/`, install its requirements and run `--submission my_method`. It receives
the same fixed trained workload automatically. The revised CKKS example has not
yet completed a live encrypted validation run; no such result is claimed.

Alternatively copy `template/` to a new directory and implement its `Adapter` class. The
required integration file is `adapter.py`; supporting Python/native sources,
build instructions and dependencies belong alongside it. Do not edit `harness/`
or frozen artifacts to accommodate a submission.

Include a README explaining the method, parameters, encoding/packing,
activation approximation, security evidence, dependencies and limitations.
Keep optional backend/hardware details private if desired, but security-critical
parameters and evidence are required for a comparable submission.

`toy_ckks/` implements native SEAL CKKS arithmetic. `plaintext_debug/` is a pipeline
test with no encryption. See [the contract](../docs/submission-contract.md).
