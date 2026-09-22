# Example report

`toy_ckks_report.json` is a real two-repeat run of the published toy through the
fixed runner. It contains no client keys, raw sensitive inputs or institutional
server details. Timings are an implementation smoke test on one development
machine, not a scheme leaderboard or full-data scaling result.

Use its field layout to interpret your own `report.json`. The graph/model ID,
manifest checksum and arithmetic purpose distinguish it from the unavailable
original 100,000-event trained bundle. Both runs share one freshly generated key
set. Their tiny score errors are CKKS approximation error.
