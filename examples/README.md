# Example reports

`toy_ckks_report.json` is a real two-repeat run of the published toy through the
fixed runner. It contains no client keys, raw sensitive inputs or institutional
server details. Timings are an implementation smoke test on one development
machine, not a scheme leaderboard or full-data scaling result.

Use its field layout to interpret your own `report.json`. The graph/model ID,
manifest checksum and arithmetic purpose distinguish it from the
100,000-event trained bundle. Both repeats share one freshly generated key
set. Their tiny score errors are CKKS approximation error.

`frozen_plaintext_report.json` is the full 100,000-node pipeline check using the
published trained bundle, evaluated on its 20,000 test nodes. It has zero score
error against the stored reference and identifies the registered manifest. Its
`is_fhe: false`, `security.status: not_fhe` and `eligible_for_comparison: false`
are intentional. Stage names such as encrypt/decrypt and payload fields labelled
encrypted_input/output are shared harness fields; in this debug report they
measure plaintext serialization, **not encryption or FHE overhead**.

Neither report includes client-key directories. The trained bundle's detection
metrics are a synthetic-data baseline, not evidence of full-size FHE execution.
