# Session Summary — Fraud Detection Project (carry-forward)

**Working dir:** `/home/suguru/Fraud detection` · Python: `/home/suguru/miniforge` (PATH export needed) · run scripts with `PYTHONPATH=src`

## Objective (ongoing)
PayPal-style graph fraud detection on the **Elliptic Bitcoin dataset** (203,769 nodes / 234,355 edges / 49 time steps / 4,545 illicit), comparing models on **cost vs value, accuracy, MLOps**, with real GraphSAGE embeddings (no SVD), resume-ready README + INTERVIEW.md, EDA rationale notebook.

## Current state — champion & results
- **Champion: `logistic__robust`** — ROC-AUC 0.8953, recall 0.863, threshold **0.19**, cost **$52,170** (baseline $131,000, savings $78,830), 17 KB, ~10 ms/1k rows, MLflow `fraud-graph-model` **v5** (unchanged this session).
- `reports/model_comparison.csv` now has **22 rows** (20 classical + 2 GNN).

## Work completed this session

### 1. Explained all 196 input features
`time_step` (1) + `f_0..f_164` (165, anonymized ~94 local + ~71 2-hop aggregated; range -13.1..287.9; top corr f_52=0.26) + `g_*` (14 graph; `g_component_size` best corr 0.122) + `graphsage_emb_*` (16, range 0..184; emb_5 corr 0.70 overall → 0.52 on future test). 0 NaNs. Dropped: `txId, cls, is_licit, labeled, is_fraud`.
**Found:** `g_partner_diversity` ≡ 1.0 and `g_mutual_flag` ≡ 0.0 for ALL nodes (Elliptic has no mutual edges) — **degenerate features, known honest flaw** (drop or redefine against weighted graph).

### 2. Feature-set ablation → `reports/ablation_feature_sets.csv`
Same model (logistic+robust), only features change, temporal test:
| set | roc | recall | cost |
|---|---|---|---|
| all 196 | 0.895 | 0.863 | **$52,170** |
| embeddings-only 16 | 0.818 | 0.496 | $73,875 (+$21,705) |
| raw+graph 180 | 0.891 | 0.800 | $54,870 |
| raw-only 166 | 0.886 | 0.815 | $56,260 |
| graph-only 15 | 0.592 | 0.584 | $136,805 (worse than baseline) |
Argument: embeddings are a **lossy summary, not a substitute**; each block earns its place (graph −$1,390, embeddings −$2,700).

### 3. Threshold (cost context) explained
Sweep on test: 0.05→$58,340 · 0.10→$54,385 · **0.19→$52,170 (min)** · 0.30→$57,565 · 0.50→$60,870 · 0.70→$64,465. Chosen on **validation** (val cost $28,320); 0.50 default would cost +$8,700. Config keys: `business.cost_false_negative: 250` / `cost_false_positive: 15`.

### 4. Probability/calibration audit
All 20 runs use `predict_proba()[:,1]` (`train.py:108-109`). Brier: xgb 0.0259 best, rf 0.0269, lgb 0.0290, **logistic 0.0828 worst** (mean P 0.164 vs actual 0.0525 — `class_weight=balanced` distorts scale; buckets non-monotone). **Insight: cost only needs ranking + tuned threshold; miscalibration absorbed by threshold.** Next step: Platt/Isotonic or `sample_weight`.

### 5. End-to-end GNN added (main deliverable)
- **New:** `src/models/gnn.py` (`GraphSAGE`, `build_normalized_adjacency()` with optional `inductive_eval_mask`, `standardize_on_train()`), `src/models/train_gnn.py`, Makefile `make gnn`.
- Protocol identical to train.py: same split, threshold tuned on val for cost, same metrics/MLflow/CSV (CSV append idempotent). **2 variants × 3 seeds (42/43/44)**, early stop on val AUC (max 200 ep, patience 15, lr 1e-2, wd 1e-4, hidden 64, embed 16), seed probs **ensembled**. Features = `time_step+f_*+g_*` (180), embeddings excluded (circular). Real weights saved to `artifacts/gnn__sage*.pkl`.
- Results: `gnn__sage` roc 0.850 / **pr 0.461 (best of all 22)** / recall 0.523 / **$69,355** / thr 0.08 / fwd ~441 ms (2.16 ms/1k amortized) / 155 KB; `gnn__sage_inductive` roc 0.824 / **$69,640** / thr 0.62. **Ranks 21–22 of 22 → embedding+linear beats end-to-end GNN.**
- **Oracle-threshold diagnostic:** logistic gap $0; gnn gap only **$3,740** (oracle $65,615) → gap is mostly **ranking degradation on future data**, not threshold transfer; inductive costs only +$285 → not future-peeking.
- Seed val AUCs: trans 0.934/0.933/0.928 (167/102/120 ep); ind 0.931/0.943/0.926 (17/46/29 ep).

### 6. Docs updated
- **README.md** (280→~310): 22 experiments everywhere, GNN in model list, results table + GNN rows (with latency footnote), new *"End-to-end GNN benchmark"* subsection with diagnosis, reproduce cmds incl. `train_gnn.py`, layout, `make gnn`.
- **INTERVIEW.md** (~300 lines): pitch = 22 experiments + "beat the end-to-end GNN", walkthrough step 8, GNN results table, **deep-dive 4.8** (why GNN loses + diagnostics), Q&A *"Why not GNN end-to-end?"* rewritten from "natural next step" → "I ran it, here's the measured reason", resume bullets, memorize numbers, key-files table.

## Bugs found/fixed this session
1. `train_gnn.py` latency had **double ×1000** (2162 ms/1k) → fixed to 2.16; CSV back-corrected.
2. Config cost keys are `business.cost_false_negative/cost_false_positive` (not `cfg["cost"]`).
3. `optimal_threshold()` returns **3-tuple** `(thr, val_cost, costs)`.
4. Per-run `artifacts/*.pkl` = bare sklearn `Pipeline` (unpickle needs `PYTHONPATH=src`); only `best_model.pkl` is dict `{name, metrics, pipeline}`.
5. (Carried) watch file tails for stray `PY` after writes — verified clean this session.

## Reproduce
```bash
export PATH=/home/suguru/miniforge/bin:$PATH
PYTHONPATH=src python src/data/load_elliptic.py       # ingest + graph features
PYTHONPATH=src python src/data/generate_graphsage.py  # embeddings (regenerates graphsage_emb_0..15)
PYTHONPATH=src python src/models/train.py             # 20 classical runs (45–90 min)
PYTHONPATH=src python src/models/train_gnn.py         # 2 GNN runs (~15 min, log: gnn_train.log)
PYTHONPATH=src python src/mlops/drift_check.py        # PSI report (66/195 drifting)
```

## Possible next steps
- Drop/redefine degenerate `g_partner_diversity` + `g_mutual_flag`.
- Calibration (Platt/Isotonic) experiment; capacity-constrained (top-k) thresholds; SHAP reason codes.
- Register GNN in MLflow registry if desired; mini-batch/neighbor-sampled GraphSAGE; seed-variance CIs.