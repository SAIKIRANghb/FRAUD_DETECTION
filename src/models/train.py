"""Train and compare multiple fraud models across preprocessing transforms.

Tracks experiments in MLflow and writes a comparison report.
"""
import os
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config, set_seed
from preprocessing.transforms import make_preprocessor, make_model, TRANSFORMS, MODELS
from evaluation.metrics import (optimal_threshold, compute_metrics, plot_roc, plot_confusion, plot_cost_curve)

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

cfg = load_config()
set_seed(cfg["seed"])
cost_fn = cfg["business"]["cost_false_negative"]
cost_fp = cfg["business"]["cost_false_positive"]

# ---------- data ----------
df = pd.read_parquet(os.path.join(BASE, cfg["data"]["output_path"]))
# use only labeled transactions (illicit vs licit)
df = df[df["labeled"] == 1].copy()
target = "is_fraud"
drop_cols = ["txId", "cls", "is_licit", "labeled", target]
X = df.drop(columns=drop_cols)
y = df[target].values

# identify numeric vs categorical
cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if c not in cat_cols]

# temporal split (avoid leakage): train early, validate mid, test late
tr_mask = df["time_step"] <= 34
va_mask = (df["time_step"] >= 35) & (df["time_step"] <= 40)
te_mask = df["time_step"] >= 41
X_train, y_train = X[tr_mask], y[tr_mask]
X_val, y_val = X[va_mask], y[va_mask]
X_test, y_test = X[te_mask], y[te_mask]
print("Split sizes:", len(X_train), len(X_val), len(X_test), "fraud prevalence:", y.mean())

mlflow.set_tracking_uri(f"sqlite:///{os.path.join(BASE, 'mlflow.db')}")
exp_name = cfg["mlflow"]["experiment"]
artifact_loc = f"file:{os.path.join(BASE, 'mlruns_artifacts')}"
try:
    mlflow.create_experiment(exp_name, artifact_location=artifact_loc)
except Exception:
    pass
mlflow.set_experiment(exp_name)

results = []

# helper to measure latency
def measure_latency(pipe, X_sample, n=50):
    Xs = X_sample.iloc[:min(len(X_sample), 200)]
    # warmup
    pipe.predict_proba(Xs.head(5))
    t0 = time.perf_counter()
    for _ in range(n):
        pipe.predict_proba(Xs)
    dt = time.perf_counter() - t0
    return (dt / n) * 1000.0 / max(len(Xs), 1) * 1000  # ms per 1000 rows? We'll report ms per batch of 1000
# simpler: ms per 1000 rows
def ms_per_1000(pipe, X_sample):
    Xs = X_sample.iloc[:min(len(X_sample), 1000)]
    pipe.predict_proba(Xs.head(5))
    t0 = time.perf_counter()
    reps = 3
    for _ in range(reps):
        pipe.predict_proba(Xs)
    dt = (time.perf_counter() - t0) / reps
    rows = len(Xs)
    return dt / rows * 1000.0

best = None

for t_kind in TRANSFORMS:
    for m_name in MODELS:
        run_name = f"{m_name}__{t_kind}"
        print("\n=== Training", run_name, "===")
        with mlflow.start_run(run_name=run_name):
            pre = make_preprocessor(num_cols, cat_cols, t_kind)
            model = make_model(m_name, class_weight=True)
            # set class-imbalance ratio dynamically for xgboost instead of fixed guess
            if m_name == "xgboost":
                ratio = float((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)
                model.set_params(scale_pos_weight=ratio)
            pipe = __import__("sklearn.pipeline", fromlist=["Pipeline"]).Pipeline([
                ("pre", pre), ("model", model)
            ])
            mlflow.log_param("transform", t_kind)
            mlflow.log_param("model", m_name)
            mlflow.log_param("n_train", len(X_train))

            t0 = time.time()
            pipe.fit(X_train, y_train)
            train_time = time.time() - t0

            val_prob = pipe.predict_proba(X_val)[:, 1]
            test_prob = pipe.predict_proba(X_test)[:, 1]

            # threshold optimize on validation
            thr, val_cost, _ = optimal_threshold(y_val, val_prob, cost_fn, cost_fp)
            metrics = compute_metrics(y_test, test_prob, thr, cost_fn, cost_fp)
            metrics["train_time_s"] = train_time
            metrics["ms_per_1000_rows"] = ms_per_1000(pipe, X_test) * 1000.0
            metrics["n_features"] = len(num_cols) + len(cat_cols)

            # log params/metrics
            for k in ["transform", "model"]:
                pass
            for mk, mv in metrics.items():
                if isinstance(mv, (int, float, np.integer, np.floating)):
                    mlflow.log_metric(mk, float(mv))
            mlflow.log_metric("val_cost", float(val_cost))
            mlflow.log_metric("train_time_s", float(train_time))

            # model size (pickle)
            os.makedirs(os.path.join(BASE, "artifacts"), exist_ok=True)
            model_path = os.path.join(BASE, "artifacts", f"{run_name}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(pipe, f)
            size_kb = os.path.getsize(model_path) / 1024.0
            metrics["model_size_kb"] = size_kb
            mlflow.log_metric("model_size_kb", size_kb)
            mlflow.log_artifact(model_path)

            # class balance info
            mlflow.log_text(json.dumps({"num_cols": num_cols, "cat_cols": cat_cols}, indent=2), "features.json")

            # collect
            row = {"run": run_name, "transform": t_kind, "model": m_name}
            row.update(metrics)
            results.append(row)

            # plots
            fig_dir = os.path.join(BASE, "reports", "figures")
            os.makedirs(fig_dir, exist_ok=True)
            roc_p=os.path.join(fig_dir, f"roc_{run_name}.png")
            cm_p=os.path.join(fig_dir, f"cm_{run_name}.png")
            cost_p=os.path.join(fig_dir, f"cost_{run_name}.png")
            plot_roc(y_test, test_prob, run_name, roc_p)
            plot_confusion(y_test, test_prob, thr, cm_p, run_name)
            _, _, costs = optimal_threshold(y_test, test_prob, cost_fn, cost_fp)
            plot_cost_curve(costs, cost_p, run_name)
            for p in (roc_p, cm_p, cost_p):
                mlflow.log_artifact(p, artifact_path="figures")

            # feature importance for tree models
            try:
                model_obj = pipe.named_steps["model"]
                if hasattr(model_obj, "feature_importances_"):
                    pre_fitted = pipe.named_steps["pre"]
                    feat_names = list(pre_fitted.get_feature_names_out())
                    importances = model_obj.feature_importances_
                    top = sorted(zip(feat_names, importances), key=lambda x: -x[1])[:25]
                    fi = pd.DataFrame(top, columns=["feature", "importance"])
                    fi_path = os.path.join(BASE, "artifacts", f"fi_{run_name}.csv")
                    fi.to_csv(fi_path, index=False)
                    mlflow.log_artifact(fi_path)
            except Exception as e:
                print("fi error", e)

            # track best by expected_cost then roc_auc
            if best is None or metrics["expected_cost"] < best[1]["expected_cost"] or \
               (metrics["expected_cost"] == best[1]["expected_cost"] and metrics["roc_auc"] > best[1]["roc_auc"]):
                best = (run_name, dict(metrics), pipe)

            print(f"  AUC={metrics['roc_auc']:.4f} F1={metrics['f1']:.4f} cost={metrics['expected_cost']:.0f} thr={metrics['threshold']:.2f}")

# Save best model
if best is not None:
    best_name, best_metrics, best_pipe = best
    best_path = os.path.join(BASE, "artifacts", "best_model.pkl")
    with open(best_path, "wb") as f:
        pickle.dump({"name": best_name, "metrics": best_metrics, "pipeline": best_pipe}, f)
    print("\nBEST:", best_name, best_metrics["expected_cost"], best_metrics["roc_auc"])
    # register model in MLflow (best run) - log as model
    # re-open a run? Instead log model artifact via direct save already. Use mlflow.register not needed.

# Comparison CSV
res_df = pd.DataFrame(results)
res_df = res_df.sort_values(["expected_cost", "roc_auc"], ascending=[True, False])
res_df.to_csv(os.path.join(BASE, "reports", "model_comparison.csv"), index=False)
print("\nSaved comparison to reports/model_comparison.csv")
print(res_df[["run", "roc_auc", "pr_auc", "f1", "expected_cost", "savings", "ms_per_1000_rows", "model_size_kb"]].to_string(index=False))
