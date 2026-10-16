"""Example monitoring job: PSI drift report between early (reference) and late (production) data."""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config
from mlops.monitoring import drift_report

cfg = load_config()
df = pd.read_parquet(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), cfg["data"]["output_path"]))
df = df[df["labeled"] == 1].copy()
ref = df[df["time_step"] <= 20]
prod = df[df["time_step"] >= 40]
feature_cols = [c for c in df.columns if c.startswith(("f_", "g_", "emb_", "graphsage_emb_"))]
rep = drift_report(ref, prod, feature_cols, threshold=cfg["mlops"]["drift_threshold_psi"])
rep = rep.sort_values("psi", ascending=False)
out = "reports/drift_report.csv"
os.makedirs("reports", exist_ok=True)
rep.to_csv(out, index=False)
print(rep.head(15).to_string(index=False))
print("Drifted features:", int(rep["drift"].sum()), "of", len(rep))
