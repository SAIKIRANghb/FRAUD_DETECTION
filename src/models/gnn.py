"""Shared GraphSAGE model + adjacency builder for the GNN experiments.

The same architecture serves two roles in this project:
1. embedding producer (src/data/generate_graphsage.py) -> 16-dim node features
2. end-to-end fraud classifier (src/models/train_gnn.py) -> direct predictions
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGE(nn.Module):
    """2-layer mean-aggregation GraphSAGE.

    h  = ReLU(W1 · (A_norm @ X))          # neighbor mean-aggregate + transform
    h  = dropout(h, 0.25)
    e  = ReLU(W2 · (A_norm @ h))          # second aggregation -> 16-dim embedding
    z  = W3 · e + b3                      # fraud logit
    """

    def __init__(self, in_dim, hidden=64, out_dim=16):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden, bias=True)
        self.w2 = nn.Linear(hidden, out_dim, bias=True)
        self.out = nn.Linear(out_dim, 1, bias=True)

    def forward(self, X, A):
        h = F.relu(self.w1(torch.sparse.mm(A, X)))
        h = F.dropout(h, p=0.25, training=self.training)
        emb = F.relu(self.w2(torch.sparse.mm(A, h)))
        logits = self.out(emb).view(-1)
        return logits, emb


def build_normalized_adjacency(tx_ids, edges_df, n, inductive_eval_mask=None):
    """Symmetric-normalized adjacency with self-loops: A_norm = D^-1/2 (A+I) D^-1/2.

    If `inductive_eval_mask` (bool array, True = evaluation node) is given, edges
    touching evaluation nodes are dropped so they only keep their self-loop:
    aggregation can no longer read the future/test subgraph (inductive variant).
    """
    idx = {tid: i for i, tid in enumerate(tx_ids)}
    u = edges_df["txId1"].map(idx)
    v = edges_df["txId2"].map(idx)
    mask = u.notna() & v.notna()
    u = u[mask].astype(int).values
    v = v[mask].astype(int).values

    if inductive_eval_mask is not None:
        keep = ~(inductive_eval_mask[u] | inductive_eval_mask[v])
        u, v = u[keep], v[keep]

    # undirected + self loops
    su = np.concatenate([u, v, np.arange(n)])
    sv = np.concatenate([v, u, np.arange(n)])
    A = torch.sparse_coo_tensor(
        torch.tensor(np.stack([su, sv]), dtype=torch.long),
        torch.ones(len(su), dtype=torch.float32),
        size=(n, n),
    ).coalesce()

    deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1.0)
    d_inv_sqrt = (1.0 / torch.sqrt(deg)).view(-1, 1)
    A_norm = torch.sparse_coo_tensor(
        A.indices(),
        A.values() * (d_inv_sqrt[A.indices()[0]] * d_inv_sqrt[A.indices()[1]]).view(-1),
        size=(n, n),
    ).coalesce()
    return A_norm


def standardize_on_train(X, train_mask):
    """Standardize features using train-node statistics only (no future leakage)."""
    mu = X[train_mask].mean(0, keepdims=True)
    sd = X[train_mask].std(0, keepdims=True) + 1e-6
    return (X - mu) / sd
