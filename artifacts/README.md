# Frozen workload artifacts

`toy-v1/` is the installed-by-cloning arithmetic smoke test. Its four-layer GCN
has widths 8/2/2/2/8, three nodes, fixed initialized weights and illustrative
labels. It is not trained and does not reproduce the 100,000-event baseline.

The original trained 100,000-event bundle has not yet been supplied to this
checkout. Its previously reported metrics remain documented separately in
[the model/dataset notes](../docs/scam-list-gcn.md). Do not substitute toy scores
for those results or regenerate a model and label it as the original run.

Maintainers can import that existing trusted run without retraining:

```bash
python -m pip install -r requirements-training.txt
python scripts/export_artifacts.py --source /path/to/runs/scam_list_gcn --out artifacts/scam-list-gcn-100k-v1
```

The exporter checks the original model checksum, loads weights safely, preserves
the graph/features/splits/scaler/threshold, and calculates an independent float64
reference. The original float32 training report is retained. Small float-rounding
differences are possible; neither result should be silently substituted for the
other. Register the reviewed manifest SHA256 in `registry.json` when publishing.
The harness consumes JSON and does not require PyTorch for inference.

Frozen files:

- `manifest.json`: artifact ID, purpose, activation, tolerances and SHA256 hashes.
- `data.json`: features, graph COO arrays, feature schema/preprocessing and splits.
- `weights.json`: all four layers' frozen matrices and biases.
- `reference.json`: plaintext scores, fixed threshold and quality metrics.
- `transactions.csv`: original synthetic log for a trained bundle.

Files use LF line endings for stable hashes. Changes create a new workload
version. Training/generation is a maintainer operation, outside timed inference.
