"""Generate real (learnable) GraphSAGE node embeddings using PyTorch."""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config

cfg = load_config()
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(BASE)

def main():
    t0 = time.time()
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    df = pd.read_parquet(os.path.join(BASE, cfg["data"]["output_path"]))
    tx_ids = df["txId"].astype(int).values
    n = len(tx_ids)
    idx = {tid: i for i, tid in enumerate(tx_ids)}

    # node features
    fcols = [c for c in df.columns if c.startswith("f_")]
    X = df[fcols].values.astype(np.float32)
    # standardize
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-6
    X = (X - mu) / sd

    time_step = df["time_step"].values
    labeled = df["labeled"].values.astype(bool)
    y = df["is_fraud"].values.astype(np.float32)

    # adjacency
    edges = pd.read_parquet(os.path.join(BASE, "data", "elliptic_edges.parquet"))
    u = edges["txId1"].map(idx)
    v = edges["txId2"].map(idx)
    mask = u.notna() & v.notna()
    u = u[mask].astype(int).values
    v = v[mask].astype(int).values
    # undirected + self loops
    su = np.concatenate([u, v, np.arange(n)])
    sv = np.concatenate([v, u, np.arange(n)])
    vals = np.ones(len(su), dtype=np.float32)
    A = torch.sparse_coo_tensor(
        torch.tensor([su, sv], dtype=torch.long),
        torch.tensor(vals),
        size=(n, n),
    ).coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1.0)
    d_inv_sqrt = (1.0 / torch.sqrt(deg)).view(-1, 1)
    # symmetric normalization: D^-1/2 A D^-1/2
    A_norm = torch.sparse_coo_tensor(
        A.indices(),
        A.values() * (d_inv_sqrt[A.indices()[0]] * d_inv_sqrt[A.indices()[1]]).view(-1),
        size=(n, n),
    ).coalesce()

    X_t = torch.tensor(X)
    # train mask: labeled & early time steps
    train_mask = torch.tensor(labeled & (time_step <= 34))
    y_t = torch.tensor(y)
    pos = y_t[train_mask].sum().item()
    neg = train_mask.sum().item() - pos
    pos_weight = torch.tensor(neg / max(pos, 1), dtype=torch.float32)
    print(f"train nodes {train_mask.sum().item()} pos {pos} neg {neg}", flush=True)

    class GraphSAGE(nn.Module):
        def __init__(self, in_dim, hidden, out_dim):
            super().__init__()
            self.w1 = nn.Linear(in_dim, hidden, bias=True)
            self.w2 = nn.Linear(hidden, out_dim, bias=True)
            self.out = nn.Linear(out_dim, 1, bias=True)
        def forward(self, X, A):
            h = torch.sparse.mm(A, X)          # (N, in_dim) mean-aggregate (incl self)
            h = F.relu(self.w1(h))             # (N, hidden)
            h = F.dropout(h, p=0.25, training=self.training)
            emb = self.w2(torch.sparse.mm(A, h))  # (N, out_dim)
            emb = F.relu(emb)
            logits = self.out(emb).view(-1)
            return logits, emb

    model = GraphSAGE(in_dim=X.shape[1], hidden=64, out_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    epochs = 60
    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, emb = model(X_t, A_norm)
        loss = loss_fn(logits[train_mask], y_t[train_mask])
        loss.backward()
        optimizer.step()
        if (ep + 1) % 5 == 0:
            with torch.no_grad():
                model.eval()
                logits, emb = model(X_t, A_norm)
                pred = (torch.sigmoid(logits) >= 0.5).float()
                acc = (pred[train_mask] == y_t[train_mask]).float().mean().item()
            print(f"epoch {ep+1}/{epochs} loss={loss.item():.4f} train_acc={acc:.4f}", flush=True)

    # embeddings
    model.eval()
    with torch.no_grad():
        _, emb = model(X_t, A_norm)
    emb_np = emb.detach().cpu().numpy().astype(np.float32)

    # overwrite graphsage_emb_* columns
    sage_cols = [f"graphsage_emb_{i}" for i in range(emb_np.shape[1])]
    for c in sage_cols:
        if c in df.columns:
            df = df.drop(columns=[c])
    sage_df = pd.DataFrame(emb_np, columns=sage_cols)
    sage_df["txId"] = tx_ids
    df = df.merge(sage_df, on="txId", how="left").fillna(0.0)

    # Also remove old pseudo GraphSAGE columns if leftover under same names done by overwrite above.

    out = os.path.join(BASE, cfg["data"]["output_path"])
    df.to_parquet(out, index=False)
    print(f"GraphSAGE embeddings saved: {sage_cols} shape {emb_np.shape} time {round(time.time()-t0,1)}s", flush=True)

if __name__ == "__main__":
    main()
