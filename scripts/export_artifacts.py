"""Export an existing trusted Scam_List_GCN training run; never retrain it."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import scipy.sparse as sp
from harness.utils import read_json, write_json, sha256
from harness.model import predict, LAYERS
from harness.metrics import quality


def export(source, destination, identifier):
    import torch
    source, destination = Path(source), Path(destination)
    required = ["features.npy", "labels.npy", "network_adjacency_normalized.npz", "splits.json",
                "feature_schema.json", "scam_list_gcn.pt", "baseline_metrics.json", "transactions.csv"]
    missing = [f for f in required if not (source / f).is_file()]
    if missing:
        raise ValueError("Missing original run artifacts: " + ", ".join(missing))
    report = read_json(source / "baseline_metrics.json")
    if report.get("model_sha256") != sha256(source / "scam_list_gcn.pt"):
        raise ValueError("Original model checksum does not match baseline_metrics.json")
    payload = torch.load(source / "scam_list_gcn.pt", map_location="cpu", weights_only=True)
    x = np.load(source / "features.npy", allow_pickle=False).astype(float)
    labels = np.load(source / "labels.npy", allow_pickle=False).astype(int)
    schema, splits = read_json(source / "feature_schema.json"), read_json(source / "splits.json")
    a = sp.load_npz(source / "network_adjacency_normalized.npz").tocoo()
    weights = {name + suffix: payload["model_state_dict"][name + suffix].detach().cpu().numpy().tolist()
               for name in LAYERS for suffix in (".weight", ".bias")}
    activation = payload["config"]["activation"]
    scores = predict(x, a.tocsr(), weights, schema["sensitive_feature_indices"], activation)
    threshold = report["validation"]["threshold"]
    destination.mkdir(parents=True, exist_ok=False)
    data = {"features": x.tolist(), "labels": labels.tolist(), "feature_names": schema["feature_names"],
            "sensitive_indices": schema["sensitive_feature_indices"], "splits": splits,
            "adjacency": {"rows": a.row.tolist(), "cols": a.col.tolist(), "values": a.data.tolist()},
            "preprocessing": schema}
    write_json(destination / "data.json", data)
    write_json(destination / "weights.json", weights)
    write_json(destination / "reference.json", {"scores": scores.tolist(), "threshold": threshold,
               "validation": quality(labels[splits["val"]], scores[splits["val"]], threshold),
               "test": quality(labels[splits["test"]], scores[splits["test"]], threshold),
               "original_training_report": report,
               "note": "Float64 reference evaluated using the original frozen float32 weights; not retrained"})
    import shutil
    shutil.copyfile(source / "transactions.csv", destination / "transactions.csv")
    files = ("data.json", "weights.json", "reference.json", "transactions.csv")
    write_json(destination / "manifest.json", {"format_version": 1, "id": identifier,
               "purpose": "trained_scam_list_gcn", "activation": activation, "atol": .001, "rtol": .001,
               "original_model_sha256": report["model_sha256"],
               "sha256": {f: sha256(destination / f) for f in files}})
    print("Exported frozen bundle:", destination)
    print("Manifest SHA256:", sha256(destination / "manifest.json"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--id", default="scam-list-gcn-100k-v1")
    a = p.parse_args()
    export(a.source, a.out, a.id)
