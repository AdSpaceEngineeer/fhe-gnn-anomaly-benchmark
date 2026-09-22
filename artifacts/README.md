# Frozen workload artifacts

There is one published benchmark workload: `scam-list-gcn-100k-v1`.
The former miniature GCN lives only in `tests/fixtures/arithmetic/` as an internal
regression fixture and is no longer registered as a benchmark workload.

[`scam-list-gcn-100k-v1/`](scam-list-gcn-100k-v1/README.md) contains the newly
frozen September 2026 trained baseline: 100,000 events, widths 8/64/32/64/8, exact
data, weights, checkpoint and verified reference scores. These files are included
when cloning. The older reported run's checkpoint was unavailable; this is the
approved retraining run, not recovery of that lost checkpoint. Do not substitute
toy scores for trained-baseline metrics.

## Use the trained bundle

```bash
python harness/run_submission.py --submission plaintext_debug --debug-plaintext --artifacts artifacts/scam-list-gcn-100k-v1 --threads 2
```

This verifies the full frozen graph in plaintext. It is not an FHE measurement.
An FHE implementation that supports this workload uses the same `--artifacts`
option with its own submission name. It is also the runner's default: no artifact
selection is needed in the CKKS copy/install/run workflow. The revised CKKS code
targets this workload; a completed full encrypted run is not claimed.

## Maintainer import, outside inference

Maintainers can import another trusted run without retraining:

```bash
python -m pip install -r requirements-training.txt
python scripts/export_artifacts.py --source /path/to/trusted/run --out artifacts/new-version --id new-version
```

The exporter checks the original model checksum, loads weights safely, preserves
the graph/features/splits/scaler/threshold, and calculates an independent float64
reference. The original float32 training report is retained. Small float-rounding
differences are possible; neither result should be silently substituted for the
other. Register the reviewed manifest SHA256 in `registry.json` when publishing.
The harness consumes JSON and does not require PyTorch for inference. The exporter
writes uncompressed files, which the loader also supports.

Frozen files:

- `manifest.json`: artifact ID, purpose, activation, tolerances and SHA256 hashes.
- `data.json` or `data.json.gz`: features, graph COO arrays, feature schema/preprocessing and splits.
- `weights.json`: all four layers' frozen matrices and biases.
- `reference.json` or `reference.json.gz`: plaintext scores, fixed threshold and quality metrics.
- `transactions.csv.gz`: original synthetic log for the trained bundle.
- `scam_list_gcn.pt`: original trained checkpoint, checked against the manifest's model hash; not loaded by PyTorch during benchmark inference.

Large files use lossless gzip compression. The loader reads them directly without
writing decompressed copies. Manifest hashes describe their **uncompressed bytes**,
so the supplied trained manifest retains its identity. Plain and compressed copies
of the same logical file must not coexist. Training metadata and the exact package
snapshot accompany the trained bundle; they are provenance, not extra model inputs
or FHE measurements.

Files use LF line endings for stable hashes. Changes create a new workload
version. Training/generation is a maintainer operation, outside timed inference.
