#!/usr/bin/env python
"""
Scam_List_GCN: synthetic transaction-log generator plus a simplified
DOMINANT-style GCN attribute autoencoder for scam/anomaly detection.

This script does not perform FHE. It creates the plaintext dataset and frozen
GCN baseline that a later FHE benchmark can use.

Example:
    python scripts/scam_list_gcn.py --outdir runs/scam_list_gcn --num-events 100000
"""

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn


PAYMENT_CHANNELS = ("wallet", "bank_transfer", "card", "instant_pay")
SENSITIVE_RAW_FIELDS = (
    "transfer_amount",
    "source_daily_total_amount",
    "prior_report_count",
)
SENSITIVE_FEATURES = tuple(f"{name}_z" for name in SENSITIVE_RAW_FIELDS)


RUN_CONFIG_FIELDS = (
    "num_events",
    "num_accounts",
    "scam_rate",
    "num_days",
    "graph_window",
    "seed",
    "hidden_dim",
    "latent_dim",
    "epochs",
    "learning_rate",
    "weight_decay",
    "activation",
    "train_normal_only",
    "sensitive_loss_weight",
)


class RunConfig(object):
    num_events = 100_000
    num_accounts = 20_000
    scam_rate = 0.04
    num_days = 30
    graph_window = 3
    seed = 42
    hidden_dim = 64
    latent_dim = 32
    epochs = 80
    learning_rate = 0.01
    weight_decay = 5e-4
    activation = "poly2"
    train_normal_only = True
    sensitive_loss_weight = 2.0

    def __init__(
        self,
        num_events=num_events,
        num_accounts=num_accounts,
        scam_rate=scam_rate,
        num_days=num_days,
        graph_window=graph_window,
        seed=seed,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        activation=activation,
        train_normal_only=train_normal_only,
        sensitive_loss_weight=sensitive_loss_weight,
    ):
        self.num_events = num_events
        self.num_accounts = num_accounts
        self.scam_rate = scam_rate
        self.num_days = num_days
        self.graph_window = graph_window
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.activation = activation
        self.train_normal_only = train_normal_only
        self.sensitive_loss_weight = sensitive_loss_weight


def config_to_dict(config):
    return {name: getattr(config, name) for name in RUN_CONFIG_FIELDS}


def set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_account_ids(num_accounts: int) -> np.ndarray:
    return np.array([f"ACC-{idx:06d}" for idx in range(num_accounts)])


def generate_transaction_log(config: RunConfig) -> pd.DataFrame:
    rng = set_seed(config.seed)
    account_ids = make_account_ids(config.num_accounts)

    num_scam_accounts = max(50, int(config.num_accounts * 0.025))
    scam_sources = rng.choice(account_ids, size=num_scam_accounts, replace=False)
    scam_destinations = rng.choice(account_ids, size=num_scam_accounts, replace=False)

    source_prior_reports = pd.Series(
        rng.poisson(lam=0.05, size=config.num_accounts),
        index=account_ids,
        dtype="int64",
    )
    source_prior_reports.loc[scam_sources] += rng.poisson(
        lam=2.0, size=len(scam_sources)
    )

    labels = rng.binomial(1, config.scam_rate, size=config.num_events).astype(np.int64)
    timestamps = [
        datetime(2026, 1, 1, tzinfo=timezone.utc)
        + timedelta(seconds=int(value))
        for value in rng.integers(
            0, config.num_days * 24 * 60 * 60, size=config.num_events
        )
    ]

    normal_source = rng.choice(account_ids, size=config.num_events, replace=True)
    normal_dest = rng.choice(account_ids, size=config.num_events, replace=True)
    scam_source = rng.choice(scam_sources, size=config.num_events, replace=True)
    scam_dest = rng.choice(scam_destinations, size=config.num_events, replace=True)

    source = np.where(labels == 1, scam_source, normal_source)
    destination = np.where(labels == 1, scam_dest, normal_dest)

    normal_channel = rng.choice(
        PAYMENT_CHANNELS,
        size=config.num_events,
        p=(0.36, 0.24, 0.28, 0.12),
    )
    scam_channel = rng.choice(
        PAYMENT_CHANNELS,
        size=config.num_events,
        p=(0.38, 0.46, 0.04, 0.12),
    )
    payment_channel = np.where(labels == 1, scam_channel, normal_channel)

    normal_amount = rng.lognormal(mean=3.45, sigma=0.85, size=config.num_events)
    scam_amount = rng.lognormal(mean=7.15, sigma=0.75, size=config.num_events)
    transfer_amount = np.where(labels == 1, scam_amount, normal_amount)
    transfer_amount = np.clip(transfer_amount, 1.0, 75_000.0).round(2)

    df = pd.DataFrame(
        {
            "event_id": np.arange(100_001, 100_001 + config.num_events),
            "timestamp": timestamps,
            "source_account": source,
            "destination_account": destination,
            "payment_channel": payment_channel,
            "transfer_amount": transfer_amount,
            "prior_report_count": source_prior_reports.loc[source].to_numpy(),
            "scam_label": labels,
        }
    )
    df = df.sort_values(["timestamp", "event_id"]).reset_index(drop=True)
    df["event_id"] = np.arange(100_001, 100_001 + len(df))
    df["date"] = df["timestamp"].dt.date.astype(str)
    df["source_daily_txn_count"] = (
        df.groupby(["source_account", "date"]).cumcount() + 1
    )
    df["source_daily_total_amount"] = (
        df.groupby(["source_account", "date"])["transfer_amount"].cumsum().round(2)
    )
    columns = [
        "event_id",
        "timestamp",
        "source_account",
        "destination_account",
        "payment_channel",
        "transfer_amount",
        "source_daily_txn_count",
        "source_daily_total_amount",
        "prior_report_count",
        "scam_label",
    ]
    return df[columns]


def iter_temporal_account_edges(
    df: pd.DataFrame, account_column: str, graph_window: int
) -> Iterable[Tuple[int, int]]:
    grouped = df.reset_index().groupby(account_column, sort=False)["index"]
    for _, event_indices in grouped:
        values = event_indices.to_numpy(dtype=np.int64)
        if len(values) < 2:
            continue
        for offset in range(1, graph_window + 1):
            left = values[:-offset]
            right = values[offset:]
            for src, dst in zip(left, right):
                yield int(src), int(dst)
                yield int(dst), int(src)


def build_event_graph(df: pd.DataFrame, graph_window: int) -> sp.csr_matrix:
    row = []
    col = []
    for account_column in ("source_account", "destination_account"):
        for src, dst in iter_temporal_account_edges(df, account_column, graph_window):
            row.append(src)
            col.append(dst)

    n = len(df)
    data = np.ones(len(row), dtype=np.float32)
    adjacency = sp.coo_matrix((data, (row, col)), shape=(n, n), dtype=np.float32)
    adjacency.sum_duplicates()
    adjacency.data[:] = 1.0
    return adjacency.tocsr()


def normalize_adjacency(adjacency: sp.csr_matrix) -> sp.coo_matrix:
    adjacency = adjacency + sp.eye(adjacency.shape[0], dtype=np.float32, format="csr")
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_sqrt = np.power(degree, -0.5, where=degree > 0)
    inv_sqrt[degree == 0] = 0.0
    degree_matrix = sp.diags(inv_sqrt.astype(np.float32))
    return (degree_matrix @ adjacency @ degree_matrix).tocoo()


def scipy_to_torch_sparse(matrix: sp.coo_matrix, device: torch.device) -> torch.Tensor:
    indices = np.vstack((matrix.row, matrix.col)).astype(np.int64)
    values = matrix.data.astype(np.float32)
    return torch.sparse_coo_tensor(
        torch.from_numpy(indices),
        torch.from_numpy(values),
        size=matrix.shape,
        dtype=torch.float32,
        device=device,
    ).coalesce()


def make_features(
    df: pd.DataFrame, train_idx: np.ndarray, outdir: Path
) -> Tuple[np.ndarray, List[str], dict]:
    channels = pd.Categorical(df["payment_channel"], categories=PAYMENT_CHANNELS)
    feature_df = pd.get_dummies(channels, prefix="channel")
    numeric = pd.DataFrame(
        {
            "transfer_amount_log": np.log1p(df["transfer_amount"]),
            "source_daily_txn_count": df["source_daily_txn_count"].astype(float),
            "source_daily_total_amount_log": np.log1p(df["source_daily_total_amount"]),
            "prior_report_count": df["prior_report_count"].astype(float),
        }
    )

    raw_to_feature_name = {
        "transfer_amount": "transfer_amount_z",
        "source_daily_total_amount": "source_daily_total_amount_z",
        "prior_report_count": "prior_report_count_z",
        "source_daily_txn_count": "source_daily_txn_count_z",
    }
    numeric = numeric.rename(
        columns={
            "transfer_amount_log": raw_to_feature_name["transfer_amount"],
            "source_daily_txn_count": raw_to_feature_name["source_daily_txn_count"],
            "source_daily_total_amount_log": raw_to_feature_name[
                "source_daily_total_amount"
            ],
            "prior_report_count": raw_to_feature_name["prior_report_count"],
        }
    )

    scaler = StandardScaler()
    scaled_numeric = numeric.copy()
    scaler.fit(numeric.iloc[train_idx])
    scaled_numeric.loc[:, :] = scaler.transform(numeric)

    features_df = pd.concat([feature_df.astype(float), scaled_numeric], axis=1)
    feature_names = list(features_df.columns)
    sensitive_indices = [
        feature_names.index(name) for name in SENSITIVE_FEATURES if name in feature_names
    ]

    schema = {
        "raw_sensitive_fields": list(SENSITIVE_RAW_FIELDS),
        "sensitive_features": list(SENSITIVE_FEATURES),
        "sensitive_feature_indices": sensitive_indices,
        "public_features": [
            name for name in feature_names if name not in set(SENSITIVE_FEATURES)
        ],
        "feature_names": feature_names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "note": "Account IDs build the graph. They are not model features.",
    }
    (outdir / "feature_schema.json").write_text(json.dumps(schema, indent=2))
    return features_df.to_numpy(dtype=np.float32), feature_names, schema


def split_nodes(labels: np.ndarray, seed: int) -> Dict[str, np.ndarray]:
    all_idx = np.arange(len(labels))
    train_val_idx, test_idx = train_test_split(
        all_idx,
        test_size=0.20,
        random_state=seed,
        stratify=labels,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.20,
        random_state=seed + 1,
        stratify=labels[train_val_idx],
    )
    return {"train": train_idx, "val": val_idx, "test": test_idx}


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        limit = math.sqrt(6.0 / (in_features + out_features))
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.uniform_(self.weight, -limit, limit)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        support = x @ self.weight
        return torch.sparse.mm(adj_norm, support) + self.bias


class Scam_List_GCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        activation: str = "poly2",
    ) -> None:
        super().__init__()
        self.activation_name = activation
        self.encoder_1 = GraphConvolution(input_dim, hidden_dim)
        self.encoder_2 = GraphConvolution(hidden_dim, latent_dim)
        self.decoder_1 = GraphConvolution(latent_dim, hidden_dim)
        self.decoder_2 = GraphConvolution(hidden_dim, input_dim)

    def activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "relu":
            return torch.relu(x)
        if self.activation_name == "poly2":
            return x + 0.125 * (x * x)
        raise ValueError(f"Unsupported activation: {self.activation_name}")

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        h1 = self.activate(self.encoder_1(x, adj_norm))
        z = self.activate(self.encoder_2(h1, adj_norm))
        hd = self.activate(self.decoder_1(z, adj_norm))
        return self.decoder_2(hd, adj_norm)


def weighted_reconstruction_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    node_idx: torch.Tensor,
    feature_weights: torch.Tensor,
) -> torch.Tensor:
    diff = x_hat[node_idx] - x[node_idx]
    error = diff * diff
    return (error * feature_weights).mean()


def reconstruction_scores(
    x: torch.Tensor, x_hat: torch.Tensor, score_indices: List[int]
) -> np.ndarray:
    diff = x_hat[:, score_indices] - x[:, score_indices]
    score = (diff * diff).mean(dim=1)
    return score.detach().cpu().numpy()


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    candidates = np.unique(np.quantile(scores, np.linspace(0.50, 0.995, 200)))
    best_f1 = -1.0
    best_value = float(candidates[-1])
    for threshold in candidates:
        pred = (scores >= threshold).astype(int)
        value = f1_score(y_true, pred, zero_division=0)
        if value > best_f1:
            best_f1 = value
            best_value = float(threshold)
    return best_value, best_f1


def compute_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> Dict[str, float]:
    pred = (scores >= threshold).astype(int)
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        metrics["average_precision"] = float(average_precision_score(y_true, scores))
    return metrics


def train_model(
    features: np.ndarray,
    labels: np.ndarray,
    adj_norm: sp.coo_matrix,
    splits: Dict[str, np.ndarray],
    schema: dict,
    config: RunConfig,
) -> Tuple[Scam_List_GCN, np.ndarray, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(features, dtype=torch.float32, device=device)
    y = labels.astype(np.int64)
    adj_torch = scipy_to_torch_sparse(adj_norm, device)

    model = Scam_List_GCN(
        input_dim=features.shape[1],
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        activation=config.activation,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    train_idx_np = splits["train"]
    if config.train_normal_only:
        train_idx_np = train_idx_np[labels[train_idx_np] == 0]
    train_idx = torch.tensor(train_idx_np, dtype=torch.long, device=device)

    feature_weights = torch.ones(features.shape[1], dtype=torch.float32, device=device)
    feature_weights[schema["sensitive_feature_indices"]] = config.sensitive_loss_weight

    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad()
        x_hat = model(x, adj_torch)
        loss = weighted_reconstruction_loss(x, x_hat, train_idx, feature_weights)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            model.eval()
            with torch.no_grad():
                scores = reconstruction_scores(
                    x, model(x, adj_torch), schema["sensitive_feature_indices"]
                )
            val_threshold, val_f1 = best_threshold(y[splits["val"]], scores[splits["val"]])
            val_metrics = compute_metrics(y[splits["val"]], scores[splits["val"]], val_threshold)
            row = {
                "epoch": float(epoch),
                "train_loss": float(loss.detach().cpu()),
                "val_f1": float(val_f1),
                "val_recall": float(val_metrics["recall"]),
                "val_roc_auc": float(val_metrics.get("roc_auc", float("nan"))),
            }
            history.append(row)
            print(
                f"epoch={epoch:03d} loss={row['train_loss']:.6f} "
                f"val_f1={row['val_f1']:.4f} val_recall={row['val_recall']:.4f}"
            )

    model.eval()
    with torch.no_grad():
        final_scores = reconstruction_scores(
            x, model(x, adj_torch), schema["sensitive_feature_indices"]
        )

    val_threshold, _ = best_threshold(y[splits["val"]], final_scores[splits["val"]])
    report = {
        "device": str(device),
        "activation": config.activation,
        "train_loss_nodes": int(len(train_idx_np)),
        "score_features": [
            schema["feature_names"][idx] for idx in schema["sensitive_feature_indices"]
        ],
        "validation": compute_metrics(y[splits["val"]], final_scores[splits["val"]], val_threshold),
        "test": compute_metrics(y[splits["test"]], final_scores[splits["test"]], val_threshold),
        "history": history,
    }
    return model, final_scores, report


def write_outputs(
    outdir: Path,
    df: pd.DataFrame,
    features: np.ndarray,
    labels: np.ndarray,
    adjacency: sp.csr_matrix,
    adj_norm: sp.coo_matrix,
    splits: Dict[str, np.ndarray],
    feature_names: List[str],
    model: Scam_List_GCN,
    scores: np.ndarray,
    report: dict,
    config: RunConfig,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "transactions.csv", index=False)
    np.save(outdir / "features.npy", features)
    np.save(outdir / "labels.npy", labels)
    sp.save_npz(outdir / "network_adjacency.npz", adjacency)
    sp.save_npz(outdir / "network_adjacency_normalized.npz", adj_norm)
    (outdir / "splits.json").write_text(
        json.dumps({key: value.tolist() for key, value in splits.items()}, indent=2)
    )
    pd.DataFrame({"event_id": df["event_id"], "scam_label": labels, "score": scores}).to_csv(
        outdir / "anomaly_scores.csv", index=False
    )

    model_path = outdir / "scam_list_gcn.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config_to_dict(config),
            "feature_names": feature_names,
            "sensitive_features": list(SENSITIVE_FEATURES),
            "model_class": "Scam_List_GCN",
        },
        model_path,
    )
    report["model_sha256"] = sha256_file(model_path)
    report["dataset"] = {
        "num_events": int(len(df)),
        "num_accounts": int(
            pd.concat([df["source_account"], df["destination_account"]]).nunique()
        ),
        "num_edges": int(adjacency.nnz),
        "scam_rate": float(labels.mean()),
    }
    report["config"] = config_to_dict(config)
    (outdir / "baseline_metrics.json").write_text(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Scam_List_GCN baseline.")
    parser.add_argument("--outdir", default="runs/scam_list_gcn")
    parser.add_argument("--num-events", type=int, default=RunConfig.num_events)
    parser.add_argument("--num-accounts", type=int, default=RunConfig.num_accounts)
    parser.add_argument("--scam-rate", type=float, default=RunConfig.scam_rate)
    parser.add_argument("--num-days", type=int, default=RunConfig.num_days)
    parser.add_argument("--graph-window", type=int, default=RunConfig.graph_window)
    parser.add_argument("--seed", type=int, default=RunConfig.seed)
    parser.add_argument("--hidden-dim", type=int, default=RunConfig.hidden_dim)
    parser.add_argument("--latent-dim", type=int, default=RunConfig.latent_dim)
    parser.add_argument("--epochs", type=int, default=RunConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=RunConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=RunConfig.weight_decay)
    parser.add_argument("--activation", choices=("poly2", "relu"), default=RunConfig.activation)
    parser.add_argument(
        "--train-all-nodes",
        action="store_true",
        help="Train reconstruction loss on all train nodes instead of train normals only.",
    )
    parser.add_argument(
        "--sensitive-loss-weight",
        type=float,
        default=RunConfig.sensitive_loss_weight,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RunConfig(
        num_events=args.num_events,
        num_accounts=args.num_accounts,
        scam_rate=args.scam_rate,
        num_days=args.num_days,
        graph_window=args.graph_window,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        activation=args.activation,
        train_normal_only=not args.train_all_nodes,
        sensitive_loss_weight=args.sensitive_loss_weight,
    )
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic transaction log...")
    df = generate_transaction_log(config)
    labels = df["scam_label"].to_numpy(dtype=np.int64)
    print(df.head().to_string(index=False))

    print("Building event graph...")
    adjacency = build_event_graph(df, config.graph_window)
    adj_norm = normalize_adjacency(adjacency)
    print(f"events={len(df)} edges={adjacency.nnz} scam_rate={labels.mean():.4f}")

    splits = split_nodes(labels, config.seed)
    features, feature_names, schema = make_features(df, splits["train"], outdir)
    print(f"features={features.shape[1]} sensitive={schema['sensitive_features']}")

    print("Training Scam_List_GCN...")
    model, scores, report = train_model(
        features=features,
        labels=labels,
        adj_norm=adj_norm,
        splits=splits,
        schema=schema,
        config=config,
    )

    write_outputs(
        outdir=outdir,
        df=df,
        features=features,
        labels=labels,
        adjacency=adjacency,
        adj_norm=adj_norm,
        splits=splits,
        feature_names=feature_names,
        model=model,
        scores=scores,
        report=report,
        config=config,
    )

    print(json.dumps({"validation": report["validation"], "test": report["test"]}, indent=2))
    print(f"Saved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
