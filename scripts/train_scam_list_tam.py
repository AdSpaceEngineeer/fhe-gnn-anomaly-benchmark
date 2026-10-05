#!/usr/bin/env python
"""Scam_List_TAM: plaintext TAM-inspired synthetic account anomaly workload.

Run in a Python 3.12 environment with requirements-tam.txt and CPU torch 2.6.0.
Smoke: python -u train_scam_list_tam.py --threads 8 --num-accounts 2000 --epochs 2 --outdir runs/tam-smoke
Full:  python -u train_scam_list_tam.py

Reference: https://github.com/mala-lab/TAM-master (model.py, train.py, utils.py).
This is NOT a reproduction of the TAM paper or the original T-Finance dataset.
Differences: synthetic sparse accounts; 10->64->32; one seeded NSGT-style cut;
one model; sampled nonedge regularization; 80 final-epoch training steps;
train-anchor-only loss; score=1-mean(cosine), without global min/max scaling.
Real PReLU and standard L2 normalization, with its explicit zero-norm guard.
No polynomial approximation, LUT, cryptography, or GPU use.
Full-graph transductive evaluation, NOT chronological/inductive evaluation.
Labels create/stratify the synthetic data and select validation thresholds, but
are never passed to model.forward, graph truncation, or the training loss.
"""
import argparse
import json
import os
from pathlib import Path
import random
import shutil

from benchmark_training_common import (dump, split_nodes, metrics,
                                       choose_threshold, environment, package)

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(key, '2')

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
import torch.nn.functional as F


FIELDS = [
    'account_age_days', 'login_count_30d', 'interaction_count_30d',
    'failed_login_count_30d', 'active_login_days_30d',
    'distinct_login_devices_30d', 'days_since_last_login',
    'unique_counterparties_30d', 'days_since_last_transfer',
    'cross_border_partner_fraction_30d',
]
SENSITIVE = ['account_age_days', 'interaction_count_30d', 'cross_border_partner_fraction_30d']
DEFINITIONS = [
    'Days since registration at the snapshot; integer >=30 in this initial generator.',
    'Successful logins in the preceding 30 days; integer.',
    'Transfers sent plus received in the preceding 30 days; counts repeated transfers.',
    'Failed login attempts in the preceding 30 days; integer.',
    'Distinct days with successful logins in the preceding 30 days; 0..30.',
    'Distinct devices with successful logins in the preceding 30 days; integer.',
    'Days since latest successful login; 0..29 here (all generated accounts active).',
    'Distinct accounts with any transfer relationship in the preceding 30 days.',
    'Days since most recent transfer; derived from the synthetic edge log.',
    'Foreign-country distinct partners divided by all distinct partners; 0..1.',
]


def generate_dataset(n, anomaly_rate, mean_partners, seed):
    """All data are synthetic; transfer counts/fractions agree with the raw edge log.

    Three latent usage communities provide correlated behavioural features.
    Anomalies combine overlapping behavioural shifts and cross-community links.
    Ground truth is generated, not evidence of real-world detection validity.
    """
    rng = np.random.default_rng(seed)
    community = rng.integers(0, 3, n)
    country = rng.integers(0, 3, n)
    labels = np.zeros(n, dtype=np.int64)
    labels[rng.choice(n, max(10, round(n * anomaly_rate)), replace=False)] = 1
    abnormal = labels.astype(bool)
    profile = community.copy()
    profile[abnormal] = (profile[abnormal] + rng.integers(1, 3, abnormal.sum())) % 3
    age = np.clip(rng.lognormal(6.0 + 0.25 * profile, 0.85, n), 30, 5000).astype(int)
    age[abnormal] = np.clip(age[abnormal] * rng.uniform(0.15, 0.8, abnormal.sum()), 30, 5000).astype(int)
    logins = 1 + rng.poisson(np.array([10., 35., 85.])[profile])
    failed = rng.poisson(0.3 + 0.04 * logins)
    failed[abnormal] += rng.poisson(1.0, abnormal.sum())
    # Never expose a simple sum/difference that recovers interaction_count.
    active_days = np.minimum(30, np.maximum(1, (np.sqrt(logins) * rng.uniform(1, 2.5, n)).astype(int)))
    active_days = np.minimum(active_days, logins)
    devices = np.minimum(logins, 1 + rng.poisson(0.8, n))
    last_login = rng.integers(0, 30 - active_days + 1)
    pools = {(g, c): np.flatnonzero((community == g) & (country == c))
             for g in range(3) for c in range(3)}
    all_accounts = np.arange(n)
    sources, destinations = [], []
    # Ring per community prevents isolates without using labels in graph repair.
    for g in range(3):
        nodes = np.flatnonzero(community == g)
        if len(nodes) > 1:
            sources.extend(nodes.tolist())
            destinations.extend(np.roll(nodes, -1).tolist())
    for i in range(n):
        degree = max(2, int(rng.poisson(mean_partners)))
        foreign_probability = 0.65 if abnormal[i] else 0.12
        for _ in range(degree):
            target_country = ((country[i] + rng.integers(1, 3)) % 3
                              if rng.random() < foreign_probability else country[i])
            target_group = int(rng.integers(0, 3)) if rng.random() < (0.65 if abnormal[i] else 0.12) else community[i]
            candidates = pools[(target_group, target_country)]
            j = int(rng.choice(candidates if len(candidates) else all_accounts))
            if j != i:
                sources.append(i)
                destinations.append(j)
    pairs = np.unique(np.column_stack([sources, destinations]), axis=0)
    src, dst = pairs.T
    transfer_count = 1 + rng.poisson(1.0 + 0.5 * profile[src] + 1.5 * labels[src])
    edge_last_day = rng.integers(0, 30, len(src))
    interactions = np.bincount(src, weights=transfer_count, minlength=n)
    interactions += np.bincount(dst, weights=transfer_count, minlength=n)
    last_transfer = np.full(n, 29, dtype=np.int64)
    np.minimum.at(last_transfer, src, edge_last_day)
    np.minimum.at(last_transfer, dst, edge_last_day)
    graph = sp.coo_matrix((np.ones(len(src), dtype=np.float32), (src, dst)), shape=(n, n)).tocsr()
    graph = graph.maximum(graph.T)
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph.data[:] = 1
    rows, cols = graph.nonzero()
    degree = np.asarray(graph.sum(axis=1)).ravel()
    foreign = np.bincount(rows, weights=(country[rows] != country[cols]), minlength=n)
    fraction = np.divide(foreign, degree, out=np.zeros(n), where=degree > 0)
    ids = np.array(['ACC-%06d' % i for i in range(n)])
    accounts = pd.DataFrame(dict(
        account_id=ids, country_code=np.array(['C0', 'C1', 'C2'])[country],
        account_age_days=age, login_count_30d=logins,
        interaction_count_30d=interactions.astype(np.int64),
        failed_login_count_30d=failed, active_login_days_30d=active_days,
        distinct_login_devices_30d=devices, days_since_last_login=last_login,
        unique_counterparties_30d=degree.astype(np.int64),
        days_since_last_transfer=last_transfer,
        cross_border_partner_fraction_30d=fraction, anomaly_label=labels))
    transfers = pd.DataFrame(dict(source_account=ids[src], destination_account=ids[dst],
                                 transfer_count_30d=transfer_count,
                                 days_since_last_transfer=edge_last_day))
    return accounts, transfers, graph, labels


def encode(accounts, train):
    values = accounts[FIELDS].to_numpy(dtype=np.float64)
    # Standard statistical preprocessing, not an encrypted activation workaround.
    values[:, :9] = np.log1p(values[:, :9])
    scaler = StandardScaler().fit(values[train])
    features = scaler.transform(values).astype(np.float32)
    schema = dict(feature_names=[name + '_z' for name in FIELDS],
                  raw_sensitive_fields=SENSITIVE,
                  sensitive_features=[name + '_z' for name in SENSITIVE],
                  sensitive_feature_indices=[FIELDS.index(name) for name in SENSITIVE],
                  public_features=[name + '_z' for name in FIELDS if name not in SENSITIVE],
                  scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(),
                  preprocessing='log1p first nine columns; fraction unchanged; train-fitted z-score for all ten',
                  fields=[dict(name=name, definition=meaning, encrypted=name in SENSITIVE)
                          for name, meaning in zip(FIELDS, DEFINITIONS)],
                  excluded_from_evaluator=['account_id', 'country_code', 'anomaly_label', 'raw transfer counts'],
                  privacy_caveat='Public graph/other features and feature-dependent truncation can leak correlated information; not full-account privacy.',
                  provenance='Fully synthetic T-Finance-inspired schema, not verified original T-Finance column names.')
    return features, schema


def truncate_graph(graph, features, seed):
    """One NSGT-style round: row thresholds followed by symmetric union.

    Distances only on existing edges. Symmetric union retains an edge if either
    endpoint retains it, matching the upstream graph_nsgt symmetrization rule.
    This feature-dependent trusted preprocessing is frozen outside inference.
    """
    coo = graph.tocoo()
    distances = np.empty(coo.nnz, dtype=np.float64)
    for start in range(0, coo.nnz, 50000):
        end = min(start + 50000, coo.nnz)
        delta = features[coo.row[start:end]].astype(np.float64) - features[coo.col[start:end]]
        distances[start:end] = np.linalg.norm(delta, axis=1)
    nonzero = distances[distances > 0]
    mean = float(nonzero.mean()) if len(nonzero) else 0.
    row_max = np.zeros(graph.shape[0])
    np.maximum.at(row_max, coo.row, distances)
    rng = np.random.default_rng(seed)
    thresholds = mean + rng.random(graph.shape[0]) * np.maximum(row_max - mean, 0)
    keep = (distances <= thresholds[coo.row]) | (distances <= thresholds[coo.col])
    cut = sp.coo_matrix((np.ones(keep.sum(), dtype=np.float32),
                        (coo.row[keep], coo.col[keep])), shape=graph.shape).tocsr()
    return cut, dict(method='single NSGT-style random threshold + symmetric union',
                     seed=seed, global_mean_edge_distance=mean,
                     original_directed_entries=int(graph.nnz), retained_directed_entries=int(cut.nnz))


def normalize_graph(graph):
    with_loops = graph + sp.eye(graph.shape[0], dtype=np.float32, format='csr')
    degree = np.asarray(with_loops.sum(axis=1)).ravel()
    d = sp.diags(degree ** -0.5)
    return (d @ with_loops @ d).tocsr()


def torch_sparse(graph):
    coo = graph.tocoo()
    return torch.sparse_coo_tensor(torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64)),
                                   torch.from_numpy(coo.data.astype(np.float32)), coo.shape).coalesce()


class GraphConvolution(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(inputs, outputs))
        self.bias = nn.Parameter(torch.zeros(outputs))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, graph):
        return torch.sparse.mm(graph, x @ self.weight) + self.bias


class Scam_List_TAM(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, embedding_dim=32):
        super().__init__()
        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.prelu1 = nn.PReLU(num_parameters=1, init=0.25)
        self.gcn2 = GraphConvolution(hidden_dim, embedding_dim)
        self.prelu2 = nn.PReLU(num_parameters=1, init=0.25)

    def forward(self, x, graph):
        return self.prelu2(self.gcn2(self.prelu1(self.gcn1(x, graph)), graph))


def affinity_scores(embedding, scoring_graph, degree, eps=1e-12):
    # Standard L2 normalization: denominator=max(||h_i||_2, eps).
    # No polynomial, LUT, hidden normalization constants, or FHE approximations.
    normalized = F.normalize(embedding, p=2, dim=1, eps=eps)
    neighbor_sum = torch.sparse.mm(scoring_graph, normalized)
    affinity = (normalized * neighbor_sum).sum(dim=1) / degree
    return 1.0 - affinity, normalized


def negative_pairs(graph, anchors, rng, per_anchor=2):
    src = np.repeat(anchors, per_anchor)
    dst = rng.integers(0, graph.shape[0], len(src))
    invalid = (src == dst) | (np.asarray(graph[src, dst]).ravel() != 0)
    for _ in range(100):
        if not invalid.any():
            return torch.from_numpy(src), torch.from_numpy(dst)
        dst[invalid] = rng.integers(0, graph.shape[0], invalid.sum())
        invalid = (src == dst) | (np.asarray(graph[src, dst]).ravel() != 0)
    raise RuntimeError('Cannot sample nonedges; graph too dense.')


def train(features, graph, encoder_graph, labels, splits, args):
    x = torch.from_numpy(features)
    encoder = torch_sparse(encoder_graph)
    scoring_csr = graph + sp.eye(graph.shape[0], dtype=np.float32, format='csr')
    scoring = torch_sparse(scoring_csr)
    degree = torch.from_numpy(np.asarray(scoring_csr.sum(axis=1)).ravel())
    anchors = torch.from_numpy(splits['train'])
    model = Scam_List_TAM(10, args.hidden_dim, args.embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    negative_rng = np.random.default_rng(args.seed + 201)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        embedding = model(x, encoder)
        scores, normalized = affinity_scores(embedding, scoring, degree)
        left, right = negative_pairs(graph, splits['train'], negative_rng)
        negative_affinity = (normalized[left] * normalized[right]).sum(dim=1).mean()
        # Minimize 1-positive_affinity + lambda*nonedge_affinity. Labels unused.
        loss = scores[anchors].mean() + args.negative_weight * negative_affinity
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss.')
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                current, _ = affinity_scores(model(x, encoder), scoring, degree)
            values = current.numpy()
            v = splits['val']
            threshold = choose_threshold(labels[v], values[v])
            result = metrics(labels[v], values[v], threshold)
            history.append(dict(epoch=epoch, loss=float(loss.detach()), validation=result,
                                score_std=float(values.std())))
            print('epoch=%03d loss=%.6f val_f1=%.4f val_recall=%.4f score_std=%.6f'
                  % (epoch, float(loss.detach()), result['f1'], result['recall'], values.std()), flush=True)
    model.eval()
    with torch.no_grad():
        embedding = model(x, encoder)
        scores, _ = affinity_scores(embedding, scoring, degree)
    scores = scores.numpy()
    if not np.isfinite(scores).all() or scores.std() < 1e-8:
        raise RuntimeError('Non-finite or collapsed anomaly scores; refusing to freeze.')
    threshold = choose_threshold(labels[splits['val']], scores[splits['val']])
    norms = torch.linalg.vector_norm(embedding, dim=1).detach().numpy()
    report = dict(validation=metrics(labels[splits['val']], scores[splits['val']], threshold),
                  test=metrics(labels[splits['test']], scores[splits['test']], threshold), history=history,
                  checkpoint_selection='Final epoch, not best validation epoch',
                  activation='PReLU (learned scalar slope in each of two layers)',
                  normalization=dict(method='h/max(L2_norm(h),epsilon)', epsilon=1e-12,
                                     min_observed_norm=float(norms.min()), max_observed_norm=float(norms.max()),
                                     num_guarded_nodes=int((norms < 1e-12).sum())),
                  evaluation='Full-graph transductive; train-anchor loss; test labels only used after training',
                  training_objective='mean_train(1-neighbor_affinity) + negative_weight*mean_sampled_nonedge_cosine',
                  note='Synthetic benchmark utility only; not evidence of real financial-crime detection.')
    return model, scores, report, scoring_csr


def verify_saved(outdir):
    checkpoint = torch.load(outdir / 'scam_list_tam.pt', map_location='cpu', weights_only=True)
    config = checkpoint['config']
    model = Scam_List_TAM(10, config['hidden_dim'], config['embedding_dim'])
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    x = np.load(outdir / 'features.npy', allow_pickle=False)
    encoder = sp.load_npz(outdir / 'network_adjacency_normalized.npz')
    scoring = sp.load_npz(outdir / 'scoring_adjacency.npz')
    degree = torch.from_numpy(np.asarray(scoring.sum(axis=1)).ravel())
    with torch.no_grad():
        scores, _ = affinity_scores(model(torch.from_numpy(x), torch_sparse(encoder)), torch_sparse(scoring), degree)
    scores = scores.numpy()
    reference = pd.read_csv(outdir / 'anomaly_scores.csv')['score'].to_numpy()
    np.testing.assert_allclose(scores, reference, atol=1e-6, rtol=1e-6)
    labels = np.load(outdir / 'labels.npy', allow_pickle=False)
    splits = json.loads((outdir / 'splits.json').read_text())
    report = json.loads((outdir / 'baseline_metrics.json').read_text())
    for split, key in [('val', 'validation'), ('test', 'test')]:
        idx = np.asarray(splits[split])
        result = metrics(labels[idx], scores[idx], report['validation']['threshold'])
        for name, value in result.items():
            np.testing.assert_allclose(value, report[key][name], rtol=1e-6, atol=1e-8)
    # Independent NumPy/SciPy forward computation; never execute archive code.
    state = json.loads((outdir / 'weights.json').read_text())
    h = x.astype(np.float64)
    for layer in (1, 2):
        h = encoder @ (h @ np.asarray(state['gcn%d.weight' % layer])) + np.asarray(state['gcn%d.bias' % layer])
        slope = float(state['prelu%d.weight' % layer][0])
        h = np.where(h >= 0, h, slope * h)
    h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
    independent = 1 - np.sum(h * (scoring @ h), axis=1) / np.asarray(scoring.sum(axis=1)).ravel()
    np.testing.assert_allclose(independent, scores, atol=1e-5, rtol=1e-5)
    return dict(passed=True, reload_max_absolute_error=float(np.max(np.abs(scores - reference))),
                independent_numpy_max_absolute_error=float(np.max(np.abs(independent - scores))),
                reload_tolerance=dict(atol=1e-6, rtol=1e-6),
                independent_tolerance=dict(atol=1e-5, rtol=1e-5))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--outdir', default='runs/scam-list-tam-synthetic-v1')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--num-accounts', type=int, default=39357)
    parser.add_argument('--mean-partners', type=float, default=8.)
    parser.add_argument('--anomaly-rate', type=float, default=0.0458)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--embedding-dim', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--negative-weight', type=float, default=1.)
    args = parser.parse_args()
    if (args.num_accounts < 500 or not 0.01 <= args.anomaly_rate <= 0.2 or
        args.epochs < 1 or not 2 <= args.mean_partners <= 30 or
        min(args.hidden_dim, args.embedding_dim) < 1 or args.learning_rate <= 0 or args.negative_weight <= 0):
        parser.error('Invalid size, rates, dimensions or epoch count.')
    return args


def main():
    args = arguments()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    outdir = Path(args.outdir)
    if any(path.exists() for path in (outdir, Path(str(outdir) + '.tar.gz'), Path(str(outdir) + '.tar.gz.sha256'))):
        raise SystemExit('Output already exists. Choose a new --outdir; nothing overwritten.')
    outdir.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print('Generating T-Finance-inspired synthetic accounts (NOT real T-Finance)...', flush=True)
    accounts, transfers, graph, labels = generate_dataset(args.num_accounts, args.anomaly_rate, args.mean_partners, args.seed)
    print(accounts.head().to_string(index=False), flush=True)
    splits = split_nodes(labels, args.seed)
    features, schema = encode(accounts, splits['train'])
    cut, cut_info = truncate_graph(graph, features, args.seed + 101)
    encoder = normalize_graph(cut)
    print('accounts=%d undirected_edges=%d anomaly_rate=%.5f retained_edges=%d'
          % (len(labels), graph.nnz // 2, labels.mean(), cut.nnz // 2), flush=True)
    print('Features=10; future encrypted fields:', SENSITIVE, flush=True)
    model, scores, report, scoring = train(features, graph, encoder, labels, splits, args)
    config = {k: v for k, v in vars(args).items() if k not in ('outdir', 'threads')}
    config.update(workload='Scam_List_TAM', input_dim=10, activation='prelu', norm_epsilon=1e-12)
    accounts.to_csv(outdir / 'accounts.csv', index=False)
    transfers.to_csv(outdir / 'transfer_edges.csv', index=False)
    np.save(outdir / 'features.npy', features)
    np.save(outdir / 'labels.npy', labels)
    for name, matrix in [('network_adjacency', graph), ('encoder_adjacency', cut),
                         ('network_adjacency_normalized', encoder), ('scoring_adjacency', scoring)]:
        sp.save_npz(outdir / (name + '.npz'), matrix)
    dump(outdir / 'splits.json', {k: v.tolist() for k, v in splits.items()})
    dump(outdir / 'feature_schema.json', schema)
    dump(outdir / 'model_config.json', config)
    dump(outdir / 'graph_preparation.json', cut_info)
    report.update(config=config, sensitive_features=schema['sensitive_features'], dataset=dict(
        kind='fully synthetic T-Finance-inspired', num_accounts=len(labels), feature_count=10,
        anomaly_count=int(labels.sum()), anomaly_rate=float(labels.mean()),
        raw_undirected_edges=graph.nnz // 2, raw_directed_entries=graph.nnz,
        encoder_undirected_edges=cut.nnz // 2, splits={k: len(v) for k, v in splits.items()}))
    dump(outdir / 'baseline_metrics.json', report)
    pd.DataFrame(dict(account_id=accounts.account_id, anomaly_label=labels, score=scores)).to_csv(outdir / 'anomaly_scores.csv', index=False)
    torch.save(dict(model_state_dict=model.state_dict(), config=config, feature_names=schema['feature_names']), outdir / 'scam_list_tam.pt')
    dump(outdir / 'weights.json', {k: v.detach().numpy().tolist() for k, v in model.state_dict().items()})
    dump(outdir / 'environment.json', environment())
    verified = verify_saved(outdir)
    dump(outdir / 'reload_verification.json', verified)
    for name in ('train_scam_list_tam.py', 'benchmark_training_common.py', 'requirements-tam.txt'):
        shutil.copyfile(Path(__file__).parent / name, outdir / name)
    names = ['accounts.csv', 'transfer_edges.csv', 'features.npy', 'labels.npy',
             'network_adjacency.npz', 'encoder_adjacency.npz', 'network_adjacency_normalized.npz',
             'scoring_adjacency.npz', 'splits.json', 'feature_schema.json', 'model_config.json',
             'graph_preparation.json', 'baseline_metrics.json', 'anomaly_scores.csv',
             'scam_list_tam.pt', 'weights.json', 'environment.json', 'reload_verification.json',
             'train_scam_list_tam.py', 'benchmark_training_common.py', 'requirements-tam.txt']
    print(json.dumps({k: report[k] for k in ('validation', 'test')}, indent=2), flush=True)
    print('Reload + independent NumPy verification: PASSED', flush=True)
    print('Saved outputs to:', outdir.resolve(), flush=True)
    package(outdir, names, dict(kind='training-upload-package-not-yet-harness-bundle',
                               workload='Scam_List_TAM', activation='prelu', num_accounts=len(labels)))


if __name__ == '__main__':
    main()
