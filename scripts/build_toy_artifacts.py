"""Maintainer utility: write the fixed arithmetic fixture. No trained baseline claim."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import scipy.sparse as sp
from harness.model import predict, LAYERS
from harness.metrics import quality
from harness.params import ROOT
from harness.utils import write_json, sha256


def main():
    root = ROOT / "tests" / "fixtures" / "arithmetic"
    root.mkdir(parents=True, exist_ok=True)
    x = np.array([[1, 0, 0, 0, .75, -.4, 1.2, .4],
                  [0, 1, 0, 0, .2, .1, .3, -.2],
                  [0, 0, 1, 0, 1.4, .5, 1.6, .8]], dtype=float)
    # Use the actual symmetric D^-1/2(A+I)D^-1/2 normalization.
    raw = np.array([[0., 1, 0], [1, 0, 1], [0, 1, 0]])
    degree = (raw + np.eye(3)).sum(axis=1)
    a = (raw + np.eye(3)) / np.sqrt(degree[:, None] * degree[None, :])
    graph = sp.coo_matrix(a)
    names = ["channel_wallet", "channel_bank_transfer", "channel_card", "channel_instant_pay",
             "transfer_amount_z", "source_daily_txn_count_z", "source_daily_total_amount_z", "prior_report_count_z"]
    rng = np.random.default_rng(2026)
    sizes = [8, 2, 2, 2, 8]
    weights = {}
    for i, name in enumerate(LAYERS):
        weights[name + ".weight"] = rng.uniform(-.2, .2, (sizes[i], sizes[i+1])).tolist()
        weights[name + ".bias"] = rng.uniform(-.05, .05, sizes[i+1]).tolist()
    scores = predict(x, a, weights, [4, 6, 7])
    data = {"features": x.tolist(), "feature_names": names, "sensitive_indices": [4, 6, 7],
            "labels": [0, 0, 1], "splits": {"train": [], "val": [], "test": [0, 1, 2]},
            "adjacency": {"rows": graph.row.tolist(), "cols": graph.col.tolist(), "values": graph.data.tolist()},
            "preprocessing": {"source": "fixed normalized numbers for arithmetic coverage; no fitted scaler"}}
    write_json(root / "data.json", data)
    write_json(root / "weights.json", weights)
    write_json(root / "reference.json", {"scores": scores.tolist(), "threshold": 1.,
                                         "test": quality(data["labels"], scores, 1.),
                                         "note": "Illustrative labels/threshold only, not detection evidence"})
    manifest = {"format_version": 1, "id": "toy-v1", "purpose": "arithmetic_smoke_test",
                "activation": "poly2", "atol": .001, "rtol": .001,
                "weights_provenance": "fixed random initialization, seed 2026; not trained",
                "sha256": {f: sha256(root / f) for f in ("data.json", "weights.json", "reference.json")}}
    write_json(root / "manifest.json", manifest)
    # Internal fixture generation must never replace the public workload registry.
    print("Internal arithmetic fixture:", root)


if __name__ == "__main__":
    main()
