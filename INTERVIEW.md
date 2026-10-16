# Interview Prep: Fraud Detection with Graph ML & GraphSAGE

Everything here is based on the project in this repo — numbers are real outputs from
`reports/model_comparison.csv`, `reports/ablation_feature_sets.csv`, and
`artifacts/best_model.pkl`.

---

## 1. The 60-second pitch

> "I built an end-to-end fraud detection system on the public **Elliptic Bitcoin dataset** —
> 203,769 transactions connected by 234,355 money-flow edges. Fraud is a *relational* problem,
> so the pipeline has three layers of signal: **14 hand-engineered graph features** (PageRank,
> communities, component size, mutual-edge patterns), **16 learned GraphSAGE embeddings** from a
> 2-layer PyTorch GNN, and the raw 165 transaction features. I benchmarked **22 experiments**
> (4 classical models × 5 preprocessing pipelines, plus 2 end-to-end GNN runs) with a
> **temporal split** to avoid future leakage, and
> picked the winner by **business cost** rather than accuracy: missed fraud costs $250, a false
> review costs $15. The best model — Logistic Regression with robust scaling — cut estimated loss
> from **$131,000 to $52,170 (60% savings)** with **86% fraud recall**, while being tiny and fast
> (17 KB, ~10 ms per 1,000 rows) — and it *beat* the end-to-end GNN, a comparison I validated by
> actually training the GNN as a classifier too. Everything is tracked in MLflow with a model
> registry and PSI drift monitoring."

---

## 2. Two-minute technical walkthrough

1. **Data** — Elliptic (MIT): 203K nodes, 234K edges, 49 time steps, 4,545 illicit / 42,019 licit.
   Unlabeled nodes used only for graph structure, never as labels.
2. **Graph features** — built a NetworkX graph, computed degree, PageRank, clustering,
   connected-component size, Louvain communities, counterparty diversity, mutual-edge flag.
3. **GraphSAGE embeddings** — PyTorch, symmetric-normalized adjacency with self-loops,
   2 mean-aggregation layers (64→16), BCE with class pos-weight, Adam, 60 epochs on early-time
   labels; produced `graphsage_emb_0..15` for all 203K nodes (~45s).
4. **Benchmarking** — 5 preprocessing pipelines (standard / robust / signed-log / quantile /
   power) × 4 models (Logistic, RandomForest, XGBoost, LightGBM) = 20 MLflow runs, all inside
   leakage-safe `Pipeline`s.
5. **Split** — temporal: train ≤34, validation 35–40 (threshold tuning), test ≥41 (unseen future).
6. **Evaluation** — ROC-AUC, PR-AUC, F1, latency, size **plus** expected business cost; threshold
   chosen on validation to minimize cost.
7. **Result** — champion `logistic__robust`: ROC-AUC 0.895, recall 0.863, cost $52,170 vs
   $131,000 baseline, registered as MLflow `fraud-graph-model` v5; drift monitor flags 66/195
   features (PSI>0.25).
8. **GNN benchmark** — same GNN also trained *end-to-end* as classifier (`train_gnn.py`,
   2 variants × 3 seeds): costs $69,355–69,640, ranks 21–22/22 → on 30K training nodes the
   embedding+linear combination beats the end-to-end GNN (verified, not assumed).

---

## 3. Results table (memorize the top 3)

| run | ROC-AUC | PR-AUC | F1 | recall | cost $ | savings $ | notes |
|---|---|---|---|---|---|---|---|
| **logistic__robust ★** | 0.895 | 0.353 | 0.278 | **0.863** | **52,170** | **78,830** | cheapest, fastest, smallest |
| random_forest__power | 0.807 | 0.624 | 0.622 | 0.590 | 56,165 | 74,835 | best F1-ish, 8 MB |
| xgboost__robust | 0.871 | 0.638 | 0.591 | 0.594 | 56,505 | 74,495 | balanced |

Champion confusion at threshold 0.19: **TP 452, FP 2,278, FN 72** of 524 test frauds.

### GNN rows (end-to-end GraphSAGE classifier)

| run | ROC-AUC | PR-AUC | recall | cost $ | note |
|---|---|---|---|---|---|
| gnn__sage | 0.850 | **0.461** | 0.523 | 69,355 | best PR-AUC of all 22 runs, ranks 21st on cost |
| gnn__sage_inductive | 0.824 | 0.387 | 0.574 | 69,640 | eval edges dropped from aggregation |

Both are 3-seed ensembles, early-stopped on validation AUC, threshold tuned on validation like
everything else. Full-graph inference: ~440 ms for all 203K nodes (2.2 ms/1k rows amortized).

---

## 4. Deep-dive sections

### 4.1 Why a graph? (PayPal alignment)
Payment fraud is visible in relationships: fraud rings share devices/IPs/assets and move money in
back-and-forth patterns. PayPal models buyers/sellers/transactions as vertices and uses PageRank,
connected components, clustering and centrality to spot rings in real time. A flat table forces
expensive multi-joins; a graph makes 2-hop queries sub-second. This project reproduces that
philosophy on public data.

### 4.2 The 14 graph features and their intuition
| feature | intuition |
|---|---|
| weighted in/out degree | volume and fan-out of flows |
| PageRank | influence/centrality in the flow network |
| clustering coefficient | triangle closure / counterparties knowing each other |
| connected component size | large isolated clusters (mixers/rings) |
| Louvain community id + size | coordinated group membership |
| unique out/in counterparties, partner diversity | repetitive vs diversified behavior |
| mutual flag | ABABA-style back-and-forth (repeat-fraudster pattern) |

EDA confirms fraud vs legit distributions differ on these → they earn their place.

### 4.3 GraphSAGE implementation (expect "show me the math")
```
Â = D^{-1/2}(A+I)D^{-1/2}          # symmetric normalization + self loops
h¹ = ReLU(W₁·(ÂX) + b₁),  dropout 0.25
emb = ReLU(W₂·(Âh¹) + b₂)          # 16-dim node embedding
logits = W₃·emb + b₃               # scalar fraud logit
loss = BCEWithLogits(pos_weight = #neg/#pos)
```
* Trained on labeled nodes with `time_step ≤ 34` only; 60 epochs; ~94.5% train accuracy.
* Embeddings then extracted for **all** nodes and fed as 16 features into classical models.
* Why embeddings instead of using the GNN as the final classifier? Interpretability, cheap
  threshold/cost tuning, and fair comparison of linear vs tree learners on identical inputs.

### 4.4 Preprocessing — "why each transform?" (very likely asked)
| step | reason |
|---|---|
| median imputation | robust to outliers, no leakage (fit on train in Pipeline) |
| winsorize 0.5/99.5% | fixes real `inf` blow-ups in PowerTransformer |
| signed log1p | compresses skewed degree/amount tails, keeps sign of net flows |
| StandardScaler | for log-shaped (≈symmetric) features |
| RobustScaler (median/IQR) | for raw outlier-heavy features |
| Quantile→normal / Yeo-Johnson | rank-based & variance-stabilizing alternatives, benchmarked empirically |
| class weights / dynamic scale_pos_weight | handle 9.8% fraud without SMOTE |
| temporal split | production predicts the future; random split leaks |

### 4.5 Evaluation — why cost, not accuracy
`expected_cost = FN×250 + FP×15`; baseline = 524×250 = $131,000. Threshold tuned on validation,
evaluated once on test. FN costs 16.7× an FP, so a high-recall linear model wins even with lower
F1 — precision-oriented trees look better on PR-AUC but cost $4K–$5K more.

### 4.6 MLOps
* MLflow 22 runs (20 classical + 2 GNN; params/metrics/figures/pickles) in `mlflow.db`
* Registry: `fraud-graph-model` v5 = champion
* `artifacts/best_model.pkl` (pipeline + metrics), reloaded and smoke-tested
* PSI drift monitoring: 66/195 features > 0.25 between `t≤20` and `t≥40` → retrain trigger
* Single-command reproducible pipeline + `config.yaml` (seed 42, cost matrix, thresholds)

### 4.7 Feature ablation — "why all 196 features, not just the embeddings?"
Very likely question after any embeddings talk. I ran it: **same model** (logistic + robust +
validation-tuned threshold), **only the feature set changes**, evaluated once on the future test
split (`reports/ablation_feature_sets.csv`, baseline = $131,000):

| feature set | #feat | ROC-AUC | PR-AUC | recall | expected cost | vs full |
|---|---|---|---|---|---|---|
| **all (time + f + g + emb)** | 196 | **0.895** | 0.353 | **0.863** | **$52,170** ★ | — |
| embeddings only | 16 | 0.818 | **0.398** | 0.496 | $73,875 | +$21,705 worse |
| raw + graph (no emb) | 180 | 0.891 | 0.316 | 0.800 | $54,870 | +$2,700 worse |
| raw only | 166 | 0.886 | 0.322 | 0.815 | $56,260 | +$4,090 worse |
| graph features only | 15 | 0.592 | 0.070 | 0.584 | $136,805 | +$84,635 worse |

Four reasons (each backed by a row above):

1. **Embeddings alone cost $21,705 more.** They're a 16-dim *compression* of the 165 raw
   features through two GNN layers — necessarily lossy. They rank fraud well (best PR-AUC 0.398,
   best F1 0.40) but catch only **49.6% of fraud** at any usable threshold vs 86.3% with
   everything. Node-level detail (`f_52`, `f_88`, fee/value patterns) doesn't survive compression.
2. **Every block adds measurable dollars:** raw alone $56,260 → +graph $54,870 (−$1,390) →
   +embeddings $52,170 (−$2,700 more). The three views fail *differently*: raw sees the node,
   `g_*` sees hand-picked topology, embeddings see learned neighborhood context.
3. **Graph features alone are worse than doing nothing** ($136,805 vs $131,000 baseline) —
   topology without transaction content can't tell fraud from legit. Graph signal is only valuable
   *combined* with node content, which is the whole argument for feature fusion.
4. **Embeddings are computed *from* the raw features** — using both isn't double-counting: the
   GNN smooths a node into its neighborhood, the raw columns preserve what's unique about it.

One-liner: *"Embeddings are a lossy summary, not a substitute — the ablation shows embeddings-only
costs $21.7K more than the full set."*

### 4.8 End-to-end GNN comparison — "so why not just train the GNN as the model?"
I actually ran it: `src/models/train_gnn.py` trains the same 2-layer GraphSAGE **as the
classifier itself**, under the identical protocol (same temporal split, threshold tuned on
validation for cost, same MLflow experiment). 2 variants × 3 seeds each, early-stopped on val
AUC, seed probabilities ensembled.

| run | ROC-AUC | PR-AUC | recall | precision | cost $ |
|---|---|---|---|---|---|
| **logistic__robust** (champion) | **0.895** | 0.353 | **0.863** | 0.166 | **52,170** |
| gnn__sage (transductive) | 0.850 | **0.461** | 0.523 | 0.375 | 69,355 |
| gnn__sage_inductive (no future edges) | 0.824 | 0.387 | 0.574 | 0.245 | 69,640 |

**The GNN loses on cost — $69,355 vs $52,170 (+$17,185), ranks 21–22 of 22.** What the
diagnostics showed (worth saying — this is the "I didn't just run it, I understood it" part):

* **It's mostly ranking, not threshold transfer.** Test-oracle threshold only gets the GNN to
  $65,615 (gap $3,740); the rest of the $17K is genuinely worse separation of fraud on future
  data — its val AUC (0.932) degrades more going forward in time than the logistic's does.
* **Where the GNN wins:** best **PR-AUC of all 22 runs (0.461)** — its top-of-list predictions
  are the most precise — plus fast amortized scoring (2.2 ms/1k rows after a 440 ms full-graph
  pass). It just doesn't reach the high recall the cost matrix demands.
* **Why it happens:** 29,894 training nodes is small for a neural model; transductive GNN +
  cost-tuned linear head benefits from the GNN as a *representation learner* while the linear
  model does cheap, stable, well-calibrated-ish thresholding on top.
* **Inductive variant** (eval-node edges dropped from aggregation) costs only $285 more than
  transductive → the advantage isn't from peeking at future subgraph structure; it's model class.

This also updated an earlier claim: "GNN end-to-end is a natural next step" became "I ran it,
and here's the measured reason the embedding approach wins on this dataset."

---

## 5. Resume bullets (pick 4–6)

* Built fraud detection on the **Elliptic Bitcoin graph** (203K nodes / 234K edges) inspired by
  PayPal's real-time graph fraud platform.
* Engineered **14 graph-topology features** (PageRank, Louvain communities, components,
  clustering, mutual-flow flags) proven useful via EDA.
* Implemented a **2-layer PyTorch GraphSAGE** and produced 16-dim learned embeddings for 203K nodes.
* Benchmarked **22 experiments** (LogReg, RF, XGB, LGBM × 5 preprocessing pipelines **+ 2
  end-to-end GraphSAGE GNN runs**) with leakage-safe sklearn Pipelines and a temporal split.
* Optimized decision thresholds by **expected business cost**, reducing fraud loss **$131K→$52K
  (60%)** with 86% recall on unseen future data.
* Ran a **feature-set ablation** proving fusion matters: embeddings-only costs $21.7K more, graph-only
  is worse than baseline — full fusion (raw + topology + GraphSAGE) wins at $52,170.
* Trained the GNN **both as embedding producer and as end-to-end classifier**; diagnosed why
  embedding+linear beats end-to-end GNN on cost ($52K vs $69K) via oracle-threshold analysis.
* Delivered **MLOps**: MLflow tracking + model registry, PSI drift monitoring (66/195 features
  flagged), versioned artifacts, and a one-command reproducible pipeline.

---

## 6. Q&A bank

**Q: Why GraphSAGE?**
It aggregates neighbor features, so shared-device/counterparty fraud rings propagate into the
embedding. Manual features describe a node; GraphSAGE describes a node *in context*.

**Q: Transductive or inductive? Why okay?**
Transductive — full graph visible during training, only early labels used. Standard for offline
embedding. In production I'd do mini-batch inductive sampling for new nodes.

**Q: Why not just use the GNN end-to-end? Did you try?**
I did — `train_gnn.py`, same split, same cost-tuned threshold, 2 variants × 3 seeds. It costs
**$69,355 vs $52,170** (ranks 21/22). Oracle-threshold analysis shows only ~$3,740 of that is
threshold transfer — the rest is weaker ranking on future data (val AUC 0.932 degrades more than
the logistic's). But the GNN has the **best PR-AUC of all 22 runs (0.461)**: its top picks are
precise, it just can't reach high recall under the cost matrix. So on 30K training nodes the GNN
earns its place as a *representation learner*; the cost-tuned linear model on top of its
embeddings is the better business outcome.

**Q: Why temporal split?**
Fraud evolves; random splits leak future into training and inflate metrics.

**Q: Why cost over F1/accuracy?**
Fraud cost asymmetry (FN 16.7× FP). F1 weights classes equally; the business doesn't.

**Q: Why did your best model have LOW precision/F1?**
Threshold 0.19 catches 86% of fraud; reviews are cheap ($15), misses are expensive ($250).
Under that cost function it's optimal. Under a review-capacity constraint I'd re-tune to top-k.

**Q: Class imbalance — did you use SMOTE?**
No — class weights / dynamic `scale_pos_weight`, which avoids synthetic-sample artifacts and
keeps the pipeline simple. PR-AUC reported alongside ROC-AUC.

**Q: What about leakage?**
All transforms fit inside the Pipeline on train only; threshold on validation only; test touched
once. Graph features/embeddings use structure but not test labels.

**Q: Hardest bug?**
PowerTransformer produced `inf` on validation tails → model crash. Fixed by adding a
Winsorizer (0.5/99.5% clip) fitted in the imputer step of every pipeline.

**Q: How would you deploy?**
Export champion pipeline → FastAPI scoring service (batch: parquet jobs), MLflow registry
aliases `champion`/`challenger`, PSI + performance monitoring, drift-triggered retraining with
champion/challenger promotion.

**Q: What would you improve?**
Neighbor-sampled mini-batch GraphSAGE, SHAP reason codes for analysts, calibration before
thresholding, capacity-constrained thresholds, automated drift-gated retraining.

**Q: Why not just use the 16 embeddings — aren't they the "real" ML?**
Ran that ablation: embeddings-only costs **$73,875 vs $52,170** for the full feature set —
$21,705 worse, and recall drops 86%→50%. A 16-dim embedding is a lossy compression of the raw
features, so it's a summary, not a replacement. Adding graph features to raw saves $1,390 more;
adding embeddings saves another $2,700 — each block earns its place (`reports/ablation_feature_sets.csv`).

**Q: How do you know the graph features help?**
Explicit ablation with everything else held constant (`reports/ablation_feature_sets.csv`):
raw-only $56,260 → raw+graph $54,870 (graph features save **$1,390**) → +embeddings $52,170.
Also: graph-only alone is useless ($136,805 — worse than baseline), so topology matters only
when fused with transaction content. EDA distributions of `g_component_size`, `g_community_size`
differ between fraud/legit, consistent with this.

---

## 7. Numbers to memorize

* 203,769 nodes • 234,355 edges • 49 time steps • 4,545 illicit • ~9.8% fraud rate
* 196 model features = 165 local + 14 graph + 16 GraphSAGE + time step
* 20 experiments • 4 models • 5 transforms **+ 2 end-to-end GNN runs (2 variants × 3 seeds) = 22**
* GNN end-to-end: cost $69,355 (rank 21/22), **best PR-AUC of all 22 (0.461)**, full-graph
  forward ~440 ms / 2.2 ms per 1k rows; oracle-threshold gap only $3,740 → ranking, not threshold
* Ablation: all-196 = $52,170 • embeddings-only = $73,875 (+$21,705) • graph-only = $136,805
  (worse than $131,000 baseline); graph features save $1,390 over raw-only
* Champion: ROC-AUC 0.895 • recall 0.863 • threshold 0.19 • TP/FP/FN = 452/2278/72
* Cost: $131,000 → $52,170 (saved $78,830, 60%)
* 17 KB model • ~10 ms per 1,000 rows
* MLflow registry `fraud-graph-model` v5 • 66/195 features drifting (PSI > 0.25)

---

## 8. Key files

| topic | file |
|---|---|
| graph features | `src/data/load_elliptic.py` |
| GraphSAGE embeddings | `src/data/generate_graphsage.py` |
| 20-run experiment loop | `src/models/train.py` |
| GNN classifier benchmark | `src/models/gnn.py`, `src/models/train_gnn.py` |
| transforms & models | `src/preprocessing/transforms.py` |
| cost/threshold metrics | `src/evaluation/metrics.py` |
| drift monitoring | `src/mlops/monitoring.py`, `src/mlops/drift_check.py` |
| config (cost, seed, splits) | `config/config.yaml` |
| EDA + transform rationale | `notebooks/eda_preprocessing.ipynb` |
| full comparison | `reports/model_comparison.csv` |
| feature-set ablation | `reports/ablation_feature_sets.csv` |
| champion artifact | `artifacts/best_model.pkl` |
