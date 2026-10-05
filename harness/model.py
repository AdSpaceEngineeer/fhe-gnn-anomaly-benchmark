"""Dispatch the published plaintext reference by workload."""
import numpy as np
import scipy.sparse as sp
from harness.workloads import gcn, tam


def adjacency(graph, n):
    rows, cols, values = (np.asarray(graph[key]) for key in ('rows', 'cols', 'values'))
    if any(a.ndim != 1 for a in (rows, cols, values)) or not (len(rows) == len(cols) == len(values)):
        raise ValueError('Invalid COO graph lengths')
    if rows.dtype.kind not in 'iu' or cols.dtype.kind not in 'iu':
        raise ValueError('Graph indices must be integers')
    if (rows < 0).any() or (cols < 0).any() or (rows >= n).any() or (cols >= n).any():
        raise ValueError('Graph index out of bounds')
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid graph values')
    return sp.coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()


def predict(features, public):
    matrix = adjacency(public['adjacency'], len(features))
    if public['workload'] == 'gcn':
        if public['activation'] != 'relu':
            raise ValueError('Scam_List_GCN requires ReLU')
        return gcn.predict(features, matrix, public)
    if public['workload'] == 'tam':
        if public['activation'] != 'prelu' or public['norm_epsilon'] != 1e-12:
            raise ValueError('Scam_List_TAM requires PReLU and the published norm guard')
        return tam.predict(features, matrix, adjacency(public['scoring_adjacency'], len(features)), public)
    raise ValueError('Unknown workload')
