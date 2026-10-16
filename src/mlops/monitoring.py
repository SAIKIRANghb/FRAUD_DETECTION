"""Drift detection and simple monitoring helpers."""
import numpy as np
import pandas as pd


def psi(expected, actual, buckets=10):
    """Population Stability Index for a single numeric feature."""
    # bin edges from expected
    quantiles = np.linspace(0, 100, buckets + 1)
    edges = np.percentile(expected, quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_pct = (e_counts + 1) / (len(expected) + len(edges))
    a_pct = (a_counts + 1) / (len(actual) + len(edges))
    val = np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))
    return float(val)


def drift_report(train_df, prod_df, feature_cols, threshold=0.25):
    rows = []
    for col in feature_cols:
        if col in train_df.columns and col in prod_df.columns:
            p = psi(train_df[col].values, prod_df[col].values)
            rows.append({"feature": col, "psi": p, "drift": p > threshold})
    return pd.DataFrame(rows)
