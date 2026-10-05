# Frozen workload artifacts

| Workload | Active artifact ID | Activation | Threshold |
|---|---|---|---:|
| GCN | `scam-list-gcn-relu-100k-v2` | ReLU | 4.11271162092986 |
| TAM | `scam-list-tam-synthetic-v1` | PReLU | 0.39311713209104254 |

Each directory contains encoded data, frozen JSON weights, reference scores,
raw synthetic logs, the original PyTorch checkpoint, training metrics and
reload verification. GCN and TAM data, splits, thresholds and weights are
independent. `registry.json` pins the manifest hash for each active artifact.
The retired polynomial GCN artifact is available only through Git history.

The runner reads JSON/gzip rather than loading PyTorch checkpoints. The manifest
hashes logical uncompressed contents for `.gz` files. Altering covered files
invalidates verification. Checkpoints are included for research provenance, not
loaded during benchmark inference. Do not unpickle untrusted checkpoints.

`reference.json.gz` contains scores recalculated using independent float64
inference from the frozen float32 weights, without retraining or threshold
tuning. Small ROC-AUC/AP differences from training logs reflect numerical ties.
`baseline_metrics.json` preserves the original training-run measurements.

Synthetic records are published to make the benchmark reproducible. In each
submission run the adapter's evaluator interface receives only the designated
public inputs and encrypted sensitive columns. Public topology and correlated
features can still reveal information; this is not full graph privacy.
