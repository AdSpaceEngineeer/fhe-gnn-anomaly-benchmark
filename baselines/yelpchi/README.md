# Published YelpChi plaintext baseline

This directory contains the benchmark's recommended model bundle. `model.json`
registers `model.npz`, whose SHA-256 is:

```text
78f0bfa62d4bf88c49b82667596ed8e41ca70ac70ae77a17c1dd9d29954a4337
```

For published comparisons against this reference, it is best to use the bundle,
weights, checksum, feature scaling, activation, and threshold as-is. The harness
also accepts differently trained compatible bundles and automatically records
their identity and computes their quality metrics.

The model is a one-hop mean-aggregation polynomial message-passing GNN with 104
input features, 16 hidden features, activation `z + 0.125z²`, and frozen
threshold `0.6605174127889721`. It was trained for 200 full-batch Adam epochs
with learning rate `0.01` and seed `2026`, using Python 3.12.13 and NumPy 2.3.5.

## Measured plaintext test baseline

The fixed test split contains 18,384 nodes, including 2,672 anomalies.

| Identifier | Recall | F1 | Accuracy | Q(k) |
|---|---:|---:|---:|---:|
| 5/22 characters | 0.673653 | 0.502583 | 0.806190 | 0.962024 |
| 9/22 characters | 0.654192 | 0.497580 | 0.807985 | 0.952449 |
| 14/22 characters | 0.596183 | 0.493113 | 0.821856 | 0.943896 |
| 18/22 characters | 0.580838 | 0.508436 | 0.836760 | 0.972431 |
| 22/22 characters | 0.597305 | 0.522422 | 0.841275 | 1.000000 |

`baseline.json` is the machine-readable report, including confusion matrices,
dataset checksum, model checksum, training metadata, and exact values.
`model.training.json` records how the weights were produced. These are
plaintext quality results; they are not FHE performance claims.

Every result identifies its selected model checksum and reports protected
Recall, F1, and Accuracy together with latency, throughput, peak RAM, storage,
communication, and key-management overhead through the benchmark result schema.

## Reproduction

Prepare the canonical `YelpChi.zip` whose SHA-256 is
`3a31296b951e6c8158dddb783ff2c62709fa987d106bbe21e59f8e119053cec3`,
then run:

```bash
fhe-gnn-benchmark prepare-yelpchi YelpChi.zip prepared/yelpchi --seed 2026
fhe-gnn-benchmark train-baseline prepared/yelpchi reproduced-model.npz \
  --hidden-features 16 --epochs 200 --learning-rate 0.01 --seed 2026
fhe-gnn-benchmark evaluate-baseline prepared/yelpchi reproduced-model.npz \
  reproduced-baseline.json --retentions 0.2 0.4 0.6 0.8 1.0
```

Reproduction supports verification, compatible-model experiments, or future
benchmark-version development. Use `bundle-model` to register reproduced or
separately trained compatible weights; use the published `model.json` unchanged
when reporting against the recommended reference.
