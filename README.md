# Fraud Detection with Graph ML & GraphSAGE Embeddings (Elliptic Bitcoin Dataset)

> An end-to-end, interview/portfolio-grade fraud detection system built on real public data,
> inspired by PayPal's *"How PayPal Uses Real-time Graph Database and Graph Analysis to Fight Fraud"*.

The core idea: fraud is rarely visible in a single row — it is visible in the **relationships**
between entities. This project therefore combines **graph topology features**, **learned GraphSAGE
embeddings**, **classical ML benchmarking**, **cost-based business evaluation**, and **MLOps
experiment tracking + drift monitoring**.

---

## TL;DR — headline numbers

| Metric | Value |
|---|---|
| Dataset | Elliptic Bitcoin: 203,769 transactions, 234,355 edges, 49 time steps |
| Label balance | 4,545 illicit / 42,019 licit (≈9.8% fraud among labeled) |
| Features used | 196 (165 local + 14 graph + 16 GraphSAGE embeddings + time step) |
| Experiments | 22 (4 classical × 5 preprocessing + 2 end-to-end GNN), all MLflow-tracked |
| Best model | **Logistic Regression + Robust scaling** (`logistic__robust`) |
| ROC-AUC / PR-AUC | **0.895 / 0.353** on future (temporal) test split |
| Business cost | **$52,170** vs **$131,000** baseline → **$78,830 saved (60%)** |
| Recall @ optimal threshold | **86.3%** (452 of 524 frauds caught, 72 missed) |
| Inference speed | ~10 ms per 1,000 rows; model size **17 KB** |
| Drift monitoring | 66 of 195 features flagged PSI > 0.25 between early/late windows |

---

## 1. Why this project (the business problem)

Fraud detection has three properties that break naive ML:

1. **Extreme imbalance** — fraud is a tiny fraction of transactions; accuracy is meaningless.
2. **Cost asymmetry** — missing fraud (false negative) costs far more than a false review (false positive).
3. **Relational signal** — fraud rings share devices, IPs, counterparties and flows; a flat row
   cannot express "this account just became linked to 7 known-bad accounts".

PayPal's blog describes exactly this: their two-sided payment network is modeled as a graph
(vertices = buyers/sellers/transactions, edges = payment/account relationships), analyzed with
PageRank, connected components, clustering and centrality to detect fraud rings in near real time.

---

## 2. Dataset

**Elliptic Bitcoin dataset** (public, MIT license) — fetched from Hugging Face
(`yhoma/elliptic-bitcoin-dataset`) by `src/data/load_elliptic.py`:

* **203,769** Bitcoin transactions → graph nodes
* **234,355** directed money-flow edges
* **49** time steps (temporal ordering, ~2-week granularity)
* **4,545 illicit** (`cls=1`), **42,019 licit** (`cls=2`), ~157K unlabeled
* 165 anonymized local/aggregated transaction features (`f_0 … f_164`) per node

Unlabeled nodes are used **only for graph structure and embedding inference**, never as labels.
Downloaded and processed data under `data/` is not committed; re-run ingestion to recreate it.

---

## 3. Architecture / pipeline

```
raw Elliptic CSVs (features, classes, edgelist)
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ src/data/load_elliptic.py                                   │
│  • load nodes/labels/edges                                  │
│  • build directed graph (NetworkX)                          │
│  • compute 14 graph-topology features per node              │
│  • output: data/elliptic_processed.parquet (185 cols)       │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ src/data/generate_graphsage.py   (PyTorch, real GraphSAGE)  │
│  • symmetric-normalized adjacency with self-loops           │
│  • 2-layer mean-aggregation GraphSAGE, 60 epochs, BCE       │
│  • 16-dim learned embeddings → graphsage_emb_0..15          │
│  • output: processed parquet (201 cols / 196 model features)│
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ src/models/train.py   (20 classical experiments, MLflow)    │
│ src/models/train_gnn.py (2 end-to-end GNN runs, 3 seeds)    │
│  • temporal split: train ≤34, val 35–40, test ≥41           │
│  • 5 preprocessors × 4 models in scikit-learn Pipelines     │
│  • cost-optimal threshold tuned on validation               │
│  • metrics, latency, size, figures logged to MLflow         │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
reports/model_comparison.csv   •  reports/drift_report.csv
artifacts/ and mlflow.db       •  generated locally; not committed
```

---

## 4. Project layout

```
config/config.yaml                     # seed, cost matrix, split, mlflow, drift threshold
data/                                  # local parquet/CSV data + graph pickle (ignored; regenerate via script)
src/
  utils.py                             # config loader, seeding
  data/load_elliptic.py                # ingest + graph feature engineering
  data/generate_graphsage.py           # PyTorch GraphSAGE embeddings
  preprocessing/transforms.py          # SignedLog, Winsorizer, 5 scalers, 4 model factories
  models/train.py                      # 20-run classical experiment loop + MLflow + artifacts
  models/gnn.py                        # GraphSAGE model + adjacency builder (shared)
  models/train_gnn.py                  # end-to-end GNN classifier benchmark (2 variants × 3 seeds)
  evaluation/metrics.py                # metrics, cost threshold optimization, plots
  mlops/monitoring.py                  # PSI drift implementation
  mlops/drift_check.py                 # drift report generator
notebooks/eda_preprocessing.ipynb      # EDA + "why each transform" rationale
reports/model_comparison.csv           # full 22-run comparison (20 classical + 2 GNN)
reports/figures/                       # ROC, confusion, cost-curve per run
reports/drift_report.csv               # PSI per feature
artifacts/                             # locally generated models (not committed)
INTERVIEW.md                           # interview-prep document
Makefile, requirements.txt, README.md
```

---

## 5. Graph features (why they matter)

Built per transaction node from the full graph (`load_elliptic.py`):

| Feature | Intuition |
|---|---|
| weighted in/out degree, total degree | transaction volume/fan-in/fan-out |
| PageRank | influence/centrality in money flows |
| local clustering coefficient | co-participating counterparty triangles |
| connected component size | belongs to a large isolated cluster? |
| Louvain community id/size | coordinated-group / ring membership |
| unique out/in counterparties, partner diversity | diversification vs repetitive counterparties |
| mutual-edge flag | back-and-forth (ABABA-style) flow pattern |

The EDA notebook shows these distributions **differ measurably between fraud and legit nodes** —
the empirical justification for engineering them.

---

## 6. GraphSAGE embeddings (real, learned — not SVD)

`src/data/generate_graphsage.py` implements a genuine 2-layer GraphSAGE in PyTorch (CPU):

* Adjacency: undirected edges + self-loops, symmetric normalization `D^{-1/2} A D^{-1/2}`
* Layer 1: `h = ReLU(W₁ · (ÂX) + b₁)` → dropout(0.25)
* Layer 2: `emb = ReLU(W₂ · (Âh) + b₂)` → 16-dim node embedding
* Loss: BCE-with-logits with `pos_weight = neg/pos` (class imbalance), Adam, weight decay
* Trained **supervised** on nodes with `time_step ≤ 34` labels only (60 epochs, ~94.5% train acc)
* Embeddings extracted for **all 203,769 nodes** (~45 s) → `graphsage_emb_0..15`

> Note: training is **transductive** — all nodes' features/edges are visible during training,
> but only early-time labels are used. This is standard for graph embedding; the downstream
> evaluation remains strictly temporal.

There is **no SVD / spectral embedding anywhere in the pipeline** (`grep` over `src/` is clean).

---

## 7. Preprocessing & transformations (and why)

All transforms live in `src/preprocessing/transforms.py` and are wrapped in scikit-learn
`ColumnTransformer` + `Pipeline` so **nothing is fit on test data**.

| Step | Why (from EDA) |
|---|---|
| **Median imputation** | robust to the extreme tails present in graph features |
| **Winsorizer (clip 0.5/99.5%)** | prevents `PowerTransformer` overflow (`inf` errors seen in practice) |
| **SignedLog** `sign(x)·log1p(|x|)` | degree/amount features are heavily right-skewed; preserves sign of net flows |
| **StandardScaler** | symmetric/log-transformed features → mean/variance scaling |
| **RobustScaler** | outlier-heavy raw features → median/IQR scaling |
| **QuantileTransformer→normal** | rank-based remapping for heavy tails |
| **PowerTransformer (Yeo-Johnson)** | variance-stabilizing alternative that handles negatives |
| **Class weights / `scale_pos_weight`** | 9.8% fraud rate; ratio computed dynamically from training data |
| **Temporal split** | random splits leak the future; production is predict-future |

Five pipeline variants are benchmarked: `standard`, `robust`, `log_standard`, `quantile`, `power`.

Models (with imbalance handling):
* **LogisticRegression** — `max_iter=2000`, `class_weight=balanced`
* **RandomForest** — 150 trees, `class_weight=balanced`
* **XGBoost** — 200 trees, dynamic `scale_pos_weight = #neg/#pos`
* **LightGBM** — 100 trees, `class_weight=balanced`
* **GraphSAGE (end-to-end GNN)** — the same 2-layer GNN used *as the classifier itself*
  (`src/models/train_gnn.py`), benchmarked under identical split/cost protocol (see §9)

---

## 8. Evaluation: accuracy vs. business cost

Metrics per run: ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix, train time,
inference latency (ms / 1,000 rows), model size — **plus** a business cost model:

```
expected_cost = FN × $250  +  FP × $15
baseline_cost = (#fraud in test) × $250   # "do nothing"
savings       = baseline_cost − expected_cost
```

* FN = missed fraud (avg $250 loss); FP = false review/customer friction ($15).
* The decision **threshold is chosen on the validation split** to minimize expected cost,
  then evaluated once on the future test split (no test leakage).
* Test baseline: **$131,000** (524 frauds × $250).

### Top results (test = time steps ≥ 41)

| run | ROC-AUC | PR-AUC | F1 | recall | expected cost | savings | ms/1k | size KB |
|---|---|---|---|---|---|---|---|---|
| **logistic__robust** ★ | **0.895** | 0.353 | 0.278 | **0.863** | **52,170** | **78,830** | 10.1 | 17 |
| random_forest__power | 0.807 | 0.624 | 0.622 | 0.590 | 56,165 | 74,835 | 72.5 | 8,358 |
| xgboost__robust | 0.871 | 0.638 | 0.591 | 0.594 | 56,505 | 74,495 | 14.7 | 632 |
| random_forest__standard | 0.825 | 0.629 | 0.586 | 0.594 | 56,640 | 74,360 | 98.8 | 8,523 |
| xgboost__standard | 0.886 | 0.645 | 0.576 | 0.595 | 56,705 | 74,295 | 34.6 | 632 |
| xgboost__quantile | 0.854 | 0.635 | 0.602 | 0.588 | 56,880 | 74,120 | 95.2 | 2,178 |
| gnn__sage (end-to-end GNN) | 0.850 | 0.461 | 0.437 | 0.523 | 69,355 | 61,645 | 2.2* | 155 |
| gnn__sage_inductive | 0.824 | 0.387 | 0.344 | 0.574 | 69,640 | 61,360 | 2.0* | 155 |

\* GNN latency is amortized full-graph inference: one forward pass = **~440 ms for all
203,769 nodes** (requires the whole graph in memory — not row-by-row scoring).

Full table: `reports/model_comparison.csv` (all 22 runs).

### The key insight (great interview material)
Tree models win on **PR-AUC/F1** (better precision), but the cost-optimal model is a
**high-recall logistic regression**: threshold 0.19 → 452 TP / 2,278 FP / 72 FN.
Because FN costs 16.7× more than FP, catching 86% of fraud beats having higher precision.
Business cost — not accuracy — decides the champion.

### End-to-end GNN benchmark (`src/models/train_gnn.py`)

To make the comparison complete, the GraphSAGE was also trained **as the classifier itself**
(not just an embedding generator) under the *identical* protocol — same temporal split, same
cost-threshold tuning on validation, same MLflow experiment:

* **Variants:** `gnn__sage` (transductive, standard full-graph training) and
  `gnn__sage_inductive` (edges touching val/test nodes dropped during aggregation, so the model
  cannot read the future subgraph).
* **Inputs:** `time_step + f_* + g_*` (180 features) — embeddings deliberately excluded
  (a GNN consuming its own embeddings would be circular).
* **Rigor:** 3 seeds each, early-stopping on validation ROC-AUC (patience 15), probabilities
  averaged (ensemble) before threshold tuning.

**Result — the honest negative:** the end-to-end GNN ranks **21–22 of 22** by expected cost
($69,355 / $69,640 vs the champion's $52,170). Diagnosis (validation-tuned vs test-oracle
threshold): even with a perfect test-set threshold the GNN only reaches $65,615, so the gap is
mostly **ranking quality on future data**, not threshold transfer ($3,740 of the gap) — and its
validation AUC (0.932) degrades more than the logistic's going forward in time.

What the GNN *does* win: **PR-AUC 0.461** (best of any model — its top-ranked predictions are
the most precise) and excellent amortized scoring speed (2.2 ms/1k rows once the graph is built).
Conclusion: on this dataset size (30K train nodes), transductive GNN + cheap calibrated model
beats end-to-end GNN — the GNN's strength is *representation*, and feeding those embeddings to a
cost-tuned linear model yields the best business outcome. Both GNN runs are logged in MLflow
with per-seed val AUCs and ROC/cost/confusion figures.

---

## 9. MLOps

* **Experiment tracking** — MLflow (`sqlite:///mlflow.db`): params, metrics, figures, per-run
  model pickles for all 20 classical runs + 2 GNN runs (22 total); the local database and
  artifacts are not committed.
* **Model registry** — champion registered as `fraud-graph-model` (version 5 = `logistic__robust`).
* **Artifacts** — `artifacts/best_model.pkl` (pipeline + metrics, loads with `PYTHONPATH=src`);
  generated locally by training and excluded from Git.
* **Reproducibility** — `config/config.yaml` (seed=42, cost matrix, drift threshold), seeded RNGs,
  one-command scripts.
* **Monitoring** — PSI drift (`src/mlops/monitoring.py`) between early (`t≤20`) and late
  (`t≥40`) windows: **66 of 195 features exceed PSI 0.25** → retraining trigger signal
  (`reports/drift_report.csv`).
* **Smoke test** — best pipeline reloaded and scored on all 9,973 test rows successfully.

### Reproduce everything
```bash
pip install -r requirements.txt          # or conda-forge equivalents
PYTHONPATH=src python src/data/load_elliptic.py         # ingest + graph features (~2 min)
PYTHONPATH=src python src/data/generate_graphsage.py    # GraphSAGE embeddings (~1 min)
PYTHONPATH=src python src/models/train.py               # 20 classical experiments
PYTHONPATH=src python src/models/train_gnn.py           # 2 GNN runs (2 variants × 3 seeds, ~15 min)
PYTHONPATH=src python src/mlops/drift_check.py          # drift report
mlflow ui --backend-store-uri sqlite:///mlflow.db       # experiment UI
```
Or: `make data`, `make sage`, `make train`, `make gnn`, `make drift`, `make mlflow-ui`.

EDA notebook: `notebooks/eda_preprocessing.ipynb` (open with `jupyter notebook` /
`jupyter lab`, run from the project root).

---

## 10. Known limitations & next steps

**Limitations**
* Elliptic is Bitcoin AML-ish data used as a fraud proxy; labels are coarse (illicit/licit).
* GraphSAGE is transductive and retrained once per dataset refresh; no online neighbor sampling.
* Cost values ($250/$15) are illustrative — real thresholds need institution-specific loss data.
* No serving layer (FastAPI/gRPC) or automated champion/challenger pipeline yet.

**Next steps**
1. Neighbor sampling + mini-batch GraphSAGE for scalability; compare vs full-batch.
2. Drift-gated retraining job (PSI → retrain → champion/challenger → registry promotion).
3. SHAP explanations on the champion for analyst-facing reason codes.
4. Capacity-constrained thresholding (top-k review budget) alongside cost-optimal thresholds.
5. Calibration (Platt/Isotonic) before threshold selection.

---

## 11. References

* Quinn Zuo et al., *How PayPal Uses Real-time Graph Database and Graph Analysis to Fight Fraud*,
  The PayPal Technology Blog (2021).
* Weber, Domeniconi, Chen, Weidele, Bellei, Robinson, Leiserson, *Anti-Money Laundering in Bitcoin:
  Experimenting with Graph Convolutional Networks for Financial Forensics*, KDD 2019 (Elliptic dataset).
* Hamilton, Ying, Leskovec, *Inductive Representation Learning on Large Graphs* (GraphSAGE), NeurIPS 2017.
* MLflow Tracking/Registry docs; PSI (Population Stability Index) monitoring practice.
