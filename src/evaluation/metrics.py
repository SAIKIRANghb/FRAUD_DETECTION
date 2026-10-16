"""Evaluation helpers: metrics, cost curves, threshold optimization."""
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                            precision_score, recall_score, accuracy_score,
                            confusion_matrix, precision_recall_curve, roc_curve)
import matplotlib.pyplot as plt
import os


def optimal_threshold(y_true, y_prob, cost_fn, cost_fp):
    """Find probability threshold minimizing expected cost."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best = None
    costs = []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        cost = fn * cost_fn + fp * cost_fp
        costs.append((t, cost))
        if best is None or cost < best[1]:
            best = (t, cost)
    return best[0], best[1], costs


def compute_metrics(y_true, y_prob, threshold, cost_fn, cost_fp):
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    # value: prevented fraud (TP*cost_fn) minus interventions ( (tp+fp)*intervention )
    # Use cost framing: net benefit relative to "do nothing"
    baseline_cost = int((y_true == 1).sum()) * cost_fn
    model_cost = fn * cost_fn + fp * cost_fp
    savings = baseline_cost - model_cost
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "expected_cost": float(model_cost),
        "baseline_cost": float(baseline_cost),
        "savings": float(savings),
        "interventions": int(tp + fp),
    }


def plot_roc(y_true, y_prob, label, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC"); plt.legend()
    plt.tight_layout(); plt.savefig(out_path); plt.close()


def plot_confusion(y_true, y_prob, threshold, out_path, title=""):
    pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title(title); plt.colorbar()
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"]); plt.yticks([0, 1], ["True 0", "True 1"])
    plt.tight_layout(); plt.savefig(out_path); plt.close()


def plot_cost_curve(costs, out_path, title=""):
    ts = [c[0] for c in costs]; cs = [c[1] for c in costs]
    plt.figure()
    plt.plot(ts, cs)
    plt.xlabel("Threshold"); plt.ylabel("Expected Cost"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path); plt.close()
