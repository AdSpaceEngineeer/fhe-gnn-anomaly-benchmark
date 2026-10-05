# Maintainer utilities

These scripts support dataset/model research and artifact export. They are not
part of a submission run. Benchmark users use the published frozen artifacts.

For training, install the root inference requirements, pandas 2.2.3 and the CPU
build of PyTorch 2.6.0 in a separate environment. Python 3.12 was used for the
published training runs.

```bash
python -m pip install -r requirements.txt pandas==2.2.3
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python scripts/scam_list_gcn.py --threads 2 --outdir runs/gcn-research
python scripts/train_scam_list_tam.py --threads 2 --outdir runs/tam-research
```

Both trainers generate a synthetic dataset, train the model, verify saved
weights and independent NumPy inference, and package an archive with checksums.
Their defaults match the published architecture/generation settings. Floating
point results may vary across library versions and machines; retraining does
not replace the registered benchmark weights. Training thread settings are
configuration, not a CPU-affinity or resource-allocation policy.

`export_artifacts.py --archive <archive.tar.gz> --sha256 <archive-hash>
--workload gcn --out <new-directory>` checks a training archive and exports
frozen inference inputs. Choose `tam` for the TAM archive. It reads numeric
arrays and JSON, verifies the source checksums and scores, and does not execute
code or load pickled checkpoints from the archive.

Changing a registered workload requires a new artifact version, updated
registry hash and integration verification. Export is a maintainer operation,
not required installation or model preparation for a submission.
