# Submissions

Copy `template/` to a new directory named after your method. Implement the
adapter methods, describe the algorithm/security/parallelism in `README.md`,
and list additional packages in `requirements.txt`. Do not edit the harness or
frozen artifacts to accommodate a submission.

The runner selects `gcn` or `tam` explicitly. A submission declares which
workloads it supports and may implement either or both. See the
[interface contract](../docs/submission-contract.md).

`plaintext_debug/` verifies installation and harness execution without
cryptography. It requires `--debug-plaintext` and cannot qualify as an FHE result.

[`toy_ckks/`](toy_ckks/README.md) is an incomplete, GCN-only CKKS scaffold. Its
small crypto round trip is separate from the benchmark. Replace its encrypted
ReLU placeholder and supply suitable full-circuit parameters before using it
as a complete submission; the unchanged scaffold is rejected at readiness check.
