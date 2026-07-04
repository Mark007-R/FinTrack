# Model Card — FinTrack Expense Categorizer

**Model id:** `tfidf_linsvc` · **Artifact:** `models/expense_classifier.joblib`
**Owner:** Mark Rodrigues · **Last updated:** 2026-07-04 (Day 10)

---

## 1. Model details

| Field | Value |
|---|---|
| Task | Multi-class text classification (transaction description → spending category) |
| Architecture | `TfidfVectorizer` (word 1–2gram **+** char 3–5gram, union) → `LinearSVC` |
| Classes (10) | groceries, dining, transport, utilities, rent, entertainment, health, shopping, income, other |
| Confidence | Softmax over the `decision_function` margins (LinearSVC has no `predict_proba`) |
| Post-processing | Day-6 high-precision multi-word disambiguation layer (`uber eats`→dining, `costco gas`→transport, …) |
| Fallback | Keyword rules (the Day-1 baseline) load automatically if the artifact is absent |
| Fit time | 0.08 s · **Inference cost:** $0 (local CPU, no API) |
| Framework | scikit-learn 1.x, joblib-serialized |

## 2. Intended use

Auto-labels a free-text transaction `description` (typed by the user or read off a
bill) into one of ten budgeting categories, powering the category heat-map,
anomaly context-features, and cash-flow forecast. Designed for **on-device,
zero-marginal-cost** inference inside the FinTrack Flask app and FastAPI service.

**Out of scope:** merchant identity resolution, fraud adjudication, and any
regulated financial decision. It is a budgeting aid, not an authority.

## 3. Training data

- **Source:** synthetic personal-finance transactions (`data/eval/transactions.csv`,
  seed `20260625`) — public/synthetic only, per the project media-discipline rule.
  **No real user financial data is ever used.**
- **Size:** 600 rows, 10 classes; class distribution is imbalanced (income 21 →
  shopping 97). Train/test split 70/30, stratified.

## 4. Evaluation

Bake-off on the held-out test split (Day-3 Phase-2b), macro-F1 over 10 classes:

| Model | macro-F1 | accuracy | fit+pred (s) | $/1k inf |
|---|---|---|---|---|
| Keyword baseline (Day-1) | 0.658 | 0.606 | 0.0 | 0 |
| TF-IDF(word) + LightGBM | 0.850 | 0.861 | 3.71 | 0 |
| SBERT + LightGBM | 0.939 | 0.939 | 18.71 | 0 |
| **TF-IDF(word+char) + LinearSVC (champion)** | **0.975** | **0.978** | **0.08** | **0** |
| DistilBERT fine-tune (ceiling) | 0.994 | 0.994 | 71.65 | 0 |
| Claude Opus zero-shot (frontier) | 1.00 | 1.00 | ~0.9/doc | ~$0.14 |

**Why the champion isn't DistilBERT or the LLM:** LinearSVC reaches **98% of the
DistilBERT macro-F1 at 1/900th of the train time and zero inference cost**, and
ties the LLM in-distribution while running offline. The frontier LLM is reserved
as a low-confidence fallback, not the default.

Per-class F1 (champion): dining 1.00, entertainment 1.00, groceries 0.98,
health 0.96, income 1.00, other 0.97, rent 1.00, shopping 1.00, transport 0.95,
**utilities 0.89** (weakest — see limitations).

## 5. Limitations & honest failure modes

- **Novel-merchant collapse (the key caveat).** On a Day-8 *out-of-distribution*
  set of unseen merchants, macro-F1 falls to **0.24** (vs 1.00 in-distribution),
  while the LLM held 1.00. TF-IDF memorizes the training vocabulary; a merchant
  string sharing no tokens with training data is near-guesswork. **This is why the
  low-confidence LLM fallback exists** and why an active-learning retrain loop
  (Day-7) routes hard cases to review.
- **Utilities** is the weakest class (F1 0.89) — it overlaps lexically with rent
  and generic bills.
- **Confidence is a softmax over SVM margins**, not a calibrated probability; treat
  the threshold as a routing heuristic, not a true likelihood.
- Trained on synthetic English-language data; real-world and non-English
  descriptions will drift.

## 6. Ethical & data considerations

Media-discipline compliant: **no real personal financial data** in training or
evaluation. The categorizer never sees account numbers or PII beyond the free-text
description string. Multi-tenancy is enforced upstream (JWT + per-user scoping),
so one user's transactions never enter another user's inference path.

## 7. Reproduce

```bash
python -m src.categorization.train            # writes models/expense_classifier.joblib
python results/run_day3_categorize.py         # regenerates the bake-off table
pytest tests/test_categorization.py -q        # 10 behavioural tests
```
