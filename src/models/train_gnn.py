"""Train GraphSAGE as an end-to-end fraud classifier and compare with classical models.

Uses the exact same protocol as src/models/train.py:
  * temporal split (train <=34, val 35-40, test >=41)
  * threshold chosen on validation to minimize expected business cost
  * same compute_metrics / MLflow experiment / reports/model_comparison.csv

Two variants:
  * gnn__sage            transductive (standard full-graph training)
  * gnn__sage_inductive  eval edges dropped during aggregation (cannot read the future subgraph)

Node features = time_step + f_* + g_* (180) -- the raw+graph ablation feature set, NOT the
GraphSAGE embeddings (that would be circular). 3 seeds each, probabilities averaged (ensemble).

Run:  PYTHONPATH=src python src/models/train_gnn.py
"""
import os
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import sys
import time
import pickle
import numpy as np
import pandas as pd
import mlflow

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from models.gnn import GraphSAGE, build_normalized_adjacency, standardize_on_train
from utils import load_config
from evaluation.metrics import (optimal_threshold, compute_metrics,
                                plot_roc, plot_confusion, plot_cost_curve)

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(os.cpu_count() or 4)

cfg = load_config()
SEEDS = [42, 43, 44]
MAX_EPOCHS, PATIENCE, LR, WD = 200, 15, 1e-2, 1e-4
cost_fn = cfg["business"]["cost_false_negative"]
cost_fp = cfg["business"]["cost_false_positive"]

# ---------- data: ALL nodes (graph structure), labels only where available ----------
df = pd.read_parquet(os.path.join(BASE, cfg["data"]["output_path"]))
tx_col = "txId" if "txId" in df.columns else "TxId"
tx_ids = df[tx_col].astype(int).values
n = len(tx_ids)

feat_cols = ["time_step"] + [c for c in df.columns if c.startswith("f_")] \
            + [c for c in df.columns if c.startswith("g_")]
train_stat_mask = (df["labeled"].values == 1) & (df["time_step"].values <= 34)
X = standardize_on_train(df[feat_cols].values.astype(np.float32), train_stat_mask)
time_step = df["time_step"].values
labeled = df["labeled"].values.astype(bool)
y = df["is_fraud"].values.astype(np.float32)

train_mask_np = labeled & (time_step <= 34)
val_mask_np   = labeled & (time_step >= 35) & (time_step <= 40)
test_mask_np  = labeled & (time_step >= 41)
eval_mask_np  = val_mask_np | test_mask_np
print(f"nodes {n} | train {int(train_mask_np.sum())} val {int(val_mask_np.sum())} "
      f"test {int(test_mask_np.sum())} | features {len(feat_cols)}", flush=True)

edges = pd.read_parquet(os.path.join(BASE, "data", "elliptic_edges.parquet"))
X_t = torch.tensor(X)
y_t = torch.tensor(y)
train_mask = torch.tensor(train_mask_np)
val_mask = torch.tensor(val_mask_np)
pos = y_t[train_mask].sum().item()
neg = int(train_mask.sum()) - pos
pos_weight = torch.tensor(neg / max(pos, 1), dtype=torch.float32)

# ---------- MLflow ----------
mlflow.set_tracking_uri(f"sqlite:///{os.path.join(BASE, 'mlflow.db')}")
try:
    mlflow.create_experiment(cfg["mlflow"]["experiment"],
                             artifact_location=f"file:{os.path.join(BASE, 'mlruns_artifacts')}")
except Exception:
    pass
mlflow.set_experiment(cfg["mlflow"]["experiment"])


def train_one(variant, seed):
    """Train GraphSAGE for one seed. Returns val/test probs, fwd latency, train time, state dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    inductive = variant.endswith("inductive")
    A_norm = build_normalized_adjacency(tx_ids, edges, n,
                                        inductive_eval_mask=eval_mask_np if inductive else None)
    model = GraphSAGE(in_dim=X.shape[1], hidden=64, out_dim=16)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state, best_auc, bad, ep = None, -1.0, 0, 0
    t0 = time.time()
    for ep in range(MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        logits, _ = model(X_t, A_norm)
        loss = loss_fn(logits[train_mask], y_t[train_mask])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            logits, _ = model(X_t, A_norm)
            val_prob = torch.sigmoid(logits[val_mask]).numpy()
        auc = float(roc_auc_only(y_t[val_mask].numpy(), val_prob))
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= PATIENCE:
            break
    train_s = time.time() - t0

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        t0 = time.time()
        for _ in range(3):
            logits, _ = model(X_t, A_norm)
        fwd_ms = (time.time() - t0) / 3 * 1000.0
        probs = torch.sigmoid(logits).numpy()
    print(f"  [{variant} seed{seed}] epochs={ep+1} best_val_auc={best_auc:.4f} "
          f"train {train_s:.0f}s fwd {fwd_ms:.0f}ms", flush=True)
    return probs[val_mask_np], probs[test_mask_np], fwd_ms, train_s, best_auc, best_state


def roc_auc_only(y_true, y_prob):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_prob)


results = []
for variant in ["gnn__sage", "gnn__sage_inductive"]:
    print(f"\n=== {variant} ({len(SEEDS)} seeds) ===", flush=True)
    with mlflow.start_run(run_name=variant):
        mlflow.log_param("model", "gnn_sage")
        mlflow.log_param("variant", variant)
        mlflow.log_param("features", "time_step+f_+g_ (embeddings excluded: circular)")
        mlflow.log_param("hidden", 64)
        mlflow.log_param("embed_dim", 16)
        mlflow.log_param("lr", LR)
        mlflow.log_param("patience", PATIENCE)
        mlflow.log_param("seeds", str(SEEDS))
        mlflow.log_param("early_stopping_on", "val_roc_auc")

        v_probs, t_probs, fwds, trains, val_aucs, states = [], [], [], [], [], []
        for s in SEEDS:
            vp, tp, fm, ts, va, st = train_one(variant, s)
            v_probs.append(vp); t_probs.append(tp); fwds.append(fm)
            trains.append(ts); val_aucs.append(va); states.append(st)
            mlflow.log_metric(f"val_auc_seed{s}", va)

        # ensemble = mean probability across seeds
        v_prob = np.mean(v_probs, axis=0)
        t_prob = np.mean(t_probs, axis=0)
        fwd_ms = float(np.mean(fwds))
        train_s = float(np.mean(trains))

        thr, val_cost, _ = optimal_threshold(y[val_mask_np], v_prob, cost_fn, cost_fp)
        metrics = compute_metrics(y[test_mask_np], t_prob, thr, cost_fn, cost_fp)
        metrics["train_time_s"] = train_s
        # amortized ms per 1000 rows: full-graph forward (fwd_ms over all n nodes) scaled to 1000
        metrics["ms_per_1000_rows"] = fwd_ms / max(n, 1) * 1000.0
        metrics["full_graph_fwd_ms"] = fwd_ms
        metrics["n_features"] = len(feat_cols)
        metrics["val_auc_mean"] = float(np.mean(val_aucs))

        for k, v in metrics.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                mlflow.log_metric(k, float(v))
        mlflow.log_metric("val_cost", float(val_cost))

        # save artifact: real trained weights of the ensemble
        os.makedirs(os.path.join(BASE, "artifacts"), exist_ok=True)
        art = {"variant": variant, "feat_cols": feat_cols, "seeds": SEEDS,
               "hidden": 64, "embed_dim": 16, "in_dim": X.shape[1],
               "states": states, "thr": float(thr),
               "note": "transductive full-graph model: rebuild A_norm (see src/models/gnn.py) "
                       "over the current graph, then sigmoid(mean logits over states)."}
        art_path = os.path.join(BASE, "artifacts", f"{variant}.pkl")
        with open(art_path, "wb") as f:
            pickle.dump(art, f)
        size_kb = os.path.getsize(art_path) / 1024.0
        metrics["model_size_kb"] = size_kb
        mlflow.log_metric("model_size_kb", size_kb)
        mlflow.log_artifact(art_path)

        fig_dir = os.path.join(BASE, "reports", "figures")
        os.makedirs(fig_dir, exist_ok=True)
        for tag, yv, pv in (("val", y[val_mask_np], v_prob), ("test", y[test_mask_np], t_prob)):
            roc_p = os.path.join(fig_dir, f"roc_{variant}_{tag}.png")
            plot_roc(yv, pv, f"{variant} ({tag})", roc_p)
            mlflow.log_artifact(roc_p, artifact_path="figures")
        cost_p = os.path.join(fig_dir, f"cost_{variant}.png")
        _, _, costs = optimal_threshold(y[test_mask_np], t_prob, cost_fn, cost_fp)
        plot_cost_curve(costs, cost_p, variant)
        cm_p = os.path.join(fig_dir, f"cm_{variant}.png")
        plot_confusion(y[test_mask_np], t_prob, thr, cm_p, variant)
        for p in (cost_p, cm_p):
            mlflow.log_artifact(p, artifact_path="figures")

        row = {"run": variant, "transform": "standardized_f+g", "model": "gnn_sage"}
        row.update(metrics)
        results.append(row)
        print(f"  {variant}: AUC={metrics['roc_auc']:.4f} PR={metrics['pr_auc']:.4f} "
              f"cost=${metrics['expected_cost']:,.0f} thr={thr:.2f} "
              f"full-graph fwd {fwd_ms:.0f}ms (val_auc {metrics['val_auc_mean']:.4f})", flush=True)

# ---------- append GNN rows to the existing comparison table (idempotent) ----------
comp_path = os.path.join(BASE, "reports", "model_comparison.csv")
old = pd.read_csv(comp_path)
old = old[~old["run"].isin([r["run"] for r in results])].copy()
merged = pd.concat([old, pd.DataFrame(results)], ignore_index=True)
merged = merged.sort_values(["expected_cost", "roc_auc"], ascending=[True, False])
merged.to_csv(comp_path, index=False)

print("\n=== Overall best by expected cost (classical + GNN) ===")
print(merged[["run", "roc_auc", "pr_auc", "recall", "expected_cost", "savings", "ms_per_1000_rows"]]
      .head(10).to_string(index=False))
best = merged.iloc[0]
print(f"\nchampion: {best['run']}  cost=${best['expected_cost']:,.0f}  auc={best['roc_auc']:.4f}")
