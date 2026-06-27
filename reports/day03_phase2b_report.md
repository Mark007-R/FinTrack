# FinTrack Production Upgrade — Day 3 of 10

**Date:** 2026-06-27 (catch-up run; scheduled Day 3 = Jun 27)
**Phase:** Phase 2b — Expense-categorizer comparison
**Field:** Financial NLP (multi-class text classification)

---

## Resume gap progress

**Gap:** FinTrack has **no expense categorizer at all** — transaction `description` is
free text typed by the user, and the only "categorization" anywhere is the Day-1 keyword
baseline (macro-F1 0.658 on the held-out split). The missing ML categorizer is the
project's headline add.

**Today's contribution:** Built and benchmarked the categorizer on a stratified 70/30
split (train 420 / test 180, 10 classes). A trained **TF-IDF + LinearSVC** lifts macro-F1
**0.658 → 0.975** and a fine-tuned **DistilBERT** reaches **0.994**, both matching a blind
zero-shot **Claude** (1.00 on the shared 60-row subset) — at a fraction of the cost and
latency. This is the model that becomes `src/categorization/classifier.py` on Day 5.

---

## Files touched

| File | Lines | Change |
|------|-------|--------|
| `results/run_day3_categorize.py` | new (~250) | stratified split, 4 trained models, full-test leaderboard, subset dump, chart |
| `results/score_llm_categorize.py` | new (~60) | scores blind LLM on the shared 60-row subset |
| `results/phase2b_categorize.csv` | new | full-test leaderboard + per-class F1 |
| `results/phase2b_llm_subset.csv` | new | apples-to-apples sub-table (same 60 rows) |
| `results/phase2b_categorize_macroF1.png` | new | macro-F1 + keyword blind-spot classes |
| `results/samples/{_llm_cat_blind_input,phase2b_subset_truth,llm_cat_predictions,phase2b_llm_scored,phase2b_samples}.json` | new | blind input, hidden truth, preds, samples |
| `results/metrics.json` | appended | Day-3 block |

No production code modified (Day 5 integrates the champion into `src/`).

---

## Setup

- **Compute:** CPU (system Python 3.11). DistilBERT fine-tune ~72 s; SBERT encode ~19 s;
  TF-IDF + LinearSVC ~0.1 s.
- **Dataset:** `data/eval/transactions.csv` — 600 labeled transactions, 10 categories
  (synthetic, media-discipline-compliant; merchant-style descriptions). Stratified 70/30
  split, seed 20260627. **All methods scored on the same held-out test set.**
- **Models:** keyword (Day-1); TF-IDF(word 1-2 + char_wb 3-5) + {LightGBM, LinearSVC};
  `all-MiniLM-L6-v2` + LightGBM; `distilbert-base-uncased` fine-tuned (4 epochs, weighted CE).

---

## Experiments

### Experiment 1 — Trained categorizers vs the keyword baseline (full test, n=180)

**Hypothesis:** With ~100 examples/class, a trained model beats keyword matching,
especially on classes where the literal category word never appears in the description.

| Method | macro-F1 | accuracy | fit+pred (s) | utilities F1 | other F1 |
|--------|---------:|---------:|-------------:|-------------:|---------:|
| keyword (baseline) | 0.658 | 0.606 | 0.0 | 0.636 | 0.315 |
| tfidf_lgbm | 0.850 | 0.861 | 3.7 | 0.600 | 0.903 |
| sbert_lgbm | 0.939 | 0.939 | 18.7 | 0.929 | 0.909 |
| **tfidf_linsvc** | **0.975** | 0.978 | **0.08** | 0.889 | 0.968 |
| **distilbert** | **0.994** | 0.994 | 71.7 | 1.000 | 1.000 |

**Interpretation:** Every trained model beats keyword. The keyword baseline's failure is
**literal-word-absence**, not class rarity: `entertainment` merchants like `PLAYSTATION
NTWK`, `HBO MAX`, `TICKETMASTER` contain no "entertainment"/"netflix"/"spotify" keyword so
they all fall through to `other`; `utilities` (`PG&E`, `SCE&G`, `Comcast`) F1 = 0.64. The
trained models learn the merchant→category mapping and fix exactly these.

### Experiment 2 — The TF-IDF + LightGBM trap (genuine insight)

**Hypothesis (initial):** TF-IDF + LightGBM is a strong, standard baseline.

**What actually happened:** A naive **word-only** TF-IDF + LightGBM with default
`min_child_samples=20` scored **macro-F1 0.116** and **could not even fit the training set
(train acc 0.32)**. Diagnosis: merchant strings are short and high-cardinality, so word
TF-IDF yields only ~3.8 non-zero features per row; LightGBM (built for *dense* tabular
data) starves on such an ultra-sparse matrix and collapses to predicting `transport`.

**Fix and lesson:**

| Variant | macro-F1 | why |
|---------|---------:|-----|
| word TF-IDF + LightGBM (`min_child_samples=20`) | 0.116 | can't split sparse features, fails to fit |
| word+char TF-IDF + LightGBM (`min_child_samples=5`) | 0.850 | char n-grams (~31 nnz/row) + smaller leaves |
| word+char TF-IDF + **LinearSVC** | **0.975** | a linear head is the *right* model for sparse text |

**The features were never the problem — the head was.** Gradient-boosted trees are the
wrong learner for sparse TF-IDF; a linear SVM on the *identical* features jumps to 0.975.
This is the counterintuitive, share-worthy finding of the session.

### Experiment 3 — Frontier LLM zero-shot, blind (shared 60-row subset)

**Method:** 60 stratified test descriptions shown to Claude with labels withheld; every
trained model's macro-F1 recomputed on those same 60 rows for an apples-to-apples row.

| Method (same 60 rows) | macro-F1 | accuracy |
|-----------------------|---------:|---------:|
| keyword | 0.646 | 0.600 |
| tfidf_lgbm | 0.839 | 0.833 |
| sbert_lgbm | 0.948 | 0.950 |
| tfidf_linsvc | 0.983 | 0.983 |
| **distilbert** | **1.000** | 1.000 |
| **llm_zeroshot (Claude)** | **1.000** | 1.000 |

**Interpretation:** The zero-shot LLM is perfect (0 disagreements) — and so is the
fine-tuned DistilBERT, while TF-IDF + LinearSVC is within ~0.02. **The specialized models
*match* the frontier LLM on this task at a fraction of the cost/latency** (LinearSVC: 0.08 s
to fit, microseconds to predict, $0 vs per-call API spend). With ≥100 labeled examples per
class, there is no accuracy reason to pay for an LLM at inference time here.

---

## Head-to-Head Comparison (running leaderboard)

| Rank | Method | macro-F1 | cost/latency | verdict |
|-----:|--------|---------:|--------------|---------|
| 1 | distilbert (fine-tune) | 0.994 | 72 s train, ~ms infer, $0 | accuracy ceiling, heaviest |
| 1 | llm_zeroshot (Claude) | 1.000* | API $ + ~1 s/call | ceiling, but paid per call (*N=60) |
| 2 | **tfidf_linsvc** | 0.975 | **0.08 s train**, µs infer, $0 | **champion candidate** (cost/accuracy) |
| 3 | sbert_lgbm | 0.939 | 19 s, ~ms, $0 | strong, semantic generalization |
| 4 | tfidf_lgbm | 0.850 | 3.7 s, ~ms, $0 | GBT is wrong head for sparse text |
| 5 | keyword (baseline) | 0.658 | 0 | fails on literal-word-absence |

---

## Key Findings

1. **No-categorizer → a trained categorizer closes the gap visibly: keyword 0.658 →
   TF-IDF+LinearSVC 0.975 → DistilBERT 0.994.** The keyword baseline's blind spot is
   literal-word-absence (entertainment merchants, `PG&E`/`SCE&G` utilities), not rarity.
2. **TF-IDF + LightGBM is a trap on short text** — word-only collapses to 0.116 and can't
   fit train; the fix is char n-grams + a *linear* head (0.975). GBTs are for dense tabular.
3. **A 0.08-second-to-train LinearSVC matches a frontier LLM (within 0.02 macro-F1) at $0
   inference** — the textbook case for a specialized categorizer once you have labels.

## What didn't work (and why)

- **Word-only TF-IDF + LightGBM** (0.116) — see Experiment 2; kept in the report as the
  instructive failure, fixed in the same table.
- **Honesty caveat:** the eval data is *synthetic* with fairly clean merchant→category
  mappings, so the absolute ceiling numbers (0.99–1.00) are optimistic vs real, noisy bank
  data (ambiguous `other`, multi-category merchants). The **relative** ordering — trained ≫
  keyword, linear-head ≫ GBT-on-sparse, specialized ≈ LLM at far lower cost — is the
  transferable result. Day 6 error-analysis + Day 8 frontier run use a fresh held-out set.

## Sample Outputs Saved

- `results/phase2b_categorize.csv` — full-test leaderboard + per-class F1
- `results/phase2b_llm_subset.csv` — apples-to-apples LLM sub-table
- `results/phase2b_categorize_macroF1.png` — macro-F1 + keyword blind-spot classes
- `results/samples/phase2b_samples.json` — keyword-fails-but-model-wins examples
- `results/samples/llm_cat_predictions.json` / `phase2b_llm_scored.json` — blind LLM run
- `results/metrics.json` — appended Day-3 block

## Next Day (Day 4)

Phase 2c — anomaly detection (IsolationForest vs robust z-score vs STL-residual; inject 20
synthetic anomalies, measure precision/recall), recurring-bill/subscription detection
(cluster by merchant ~amount ~period), and a cash-flow forecast baseline (naive seasonal vs
Prophet vs gradient-boosted; report MAPE). → `results/phase2c_anomaly_forecast.csv`.

## Code Changes

New analysis scripts under `results/`; no production-code edits. The Day-5 champion
(`tfidf_linsvc`, pending Day-6 tuning) will be saved to `models/expense_classifier.joblib`
and wired into `src/categorization/classifier.py`.
