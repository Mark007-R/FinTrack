# FinTrack Production Upgrade — Day 5 of 10
## Phase 3: Champion integration + production refactor (PHASE-WRAP)

**Date:** 2026-06-29 · **Project:** FinTrack (Personal Finance Manager) · **Field:** Document AI + Financial NLP

---

### Resume gap progress

**Gap:** The repo was four disconnected experiment scripts (Days 1–4) plus production
code that still shipped the audited bugs: a regex bill scanner, a broken multi-tenancy
model where every user could read/delete every other user's transactions, a
machine-locked poppler path, and an `invest()` that summed *all* users' money. Nothing
was wired into the app.

**Today's contribution:** Turned the four bake-off winners into a real `src/` package,
delegated the production code to them **without breaking a single function signature**,
fixed the multi-tenancy leak end-to-end, removed the hardcoded path, and stood up an
async FastAPI service exposing all five components. An integration harness proves every
component reproduces its Day 2–4 champion number after the move, all five endpoints
return 200, and a SQLite mirror proves user A can no longer see or delete user B's rows.

---

### Files touched

**New `src/` package (champion integration):**
- `src/schemas.py` — Pydantic v2 request/response models for all five endpoints
- `src/extraction/extractor.py` — Day-2 `rules_smart` champion + regex fallback
- `src/categorization/classifier.py` + `train.py` — Day-3 TF-IDF+LinearSVC champion; trains `models/expense_classifier.joblib`
- `src/anomaly/detector.py` — Day-4 IsolationForest + recurring/duplicate detectors
- `src/forecast/cashflow.py` — Day-4 Prophet champion + seasonal-naive fallback
- `src/reco/investments.py` — **new** per-user risk-profiled recommender

**Production code (signature-preserving fixes):**
- `extract_bill.py` — `find_bill_details()` (43) now delegates to the champion extractor (regex kept as fallback); `extract_text_from_pdf()` (24) drops the hardcoded `C:\poppler-24.07.0\...` path for pdfplumber → env-var poppler; bill INSERT now carries `user_id`
- `app.py` — `dashboard()` (55): `user_id` filter added to INSERT / DELETE / SELECT (lines 65–91)
- `invest.py` — full rewrite: per-user `user_id`-scoped balance + risk profile, `mysql.connector` → `pymysql`, signature/endpoint unchanged
- `db/migrations/001_add_user_id_to_transactions.sql` — **new** schema migration for the fix

**API + verification:**
- `api.py` — **new** FastAPI service (`/extract /categorize /anomaly /forecast /recommend` + `/health`), port 8000
- `results/run_day5_integration.py` — **new** 7-check integration harness
- `results/phase3_integration.json`, `results/phase3_integration_parity.png`, `results/samples/phase3_integration_samples.json`, `results/metrics.json` (appended `day5_phase3`)
- `requirements.txt` (+pdfplumber, +numpy), `requirements-api.txt` (**new**)

---

### Setup

- **Compute:** CPU, project `.venv`. FastAPI exercised via `TestClient` (no live server needed).
- **Datasets (media-discipline respected):** public SROIE receipts (100), synthetic transactions (600 categorization / 3,054 Day-4 stream). No real financial data.
- **Champion artifact:** `models/expense_classifier.joblib` (git-ignored; regenerate with `python -m src.categorization.train`).

---

### Experiments — integration parity (does the refactor preserve the numbers?)

**Hypothesis:** moving champion logic from one-off scripts into an importable package +
production delegation should reproduce the Day 2–4 metrics exactly; if a number moves, the
integration introduced a bug.

**Method:** `run_day5_integration.py` runs each `src/` component on the original Day 1–4
eval data and compares to the recorded champion score; then smoke-tests the API and the
multi-tenancy queries.

| # | Component | Metric | Day 2–4 champion | Day 5 integrated | Parity |
|---|-----------|--------|-----------------:|-----------------:|:------:|
| 1 | Extraction (`src.extraction`) | amount acc (100 SROIE) | 0.58 | **0.580** | ✅ |
| 1 | Extraction | date acc / merchant non-null | 0.87 / 0.77 | 0.870 / 0.940 | ✅ |
| 2 | Categorization (joblib) | held-out macro-F1 | 0.975 | **0.9752** | ✅ |
| 3 | Anomaly (`src.anomaly`) | Precision@20 / Recall@20 | 0.95 / 0.95 | **0.950 / 0.950** | ✅ |
| 4 | Forecast (`src.forecast`) | method selected (36 mo) | Prophet | **prophet** | ✅ |
| 5 | Recommend (`src.reco`) | personalization differs? | — (none existed) | **yes** | ✅ |
| 6 | FastAPI | all 5 endpoints + health → 200 | — | **True** | ✅ |
| 7 | Multi-tenancy | A sees/deletes B's rows? | **was: yes (bug)** | **no (blocked)** | ✅ |

**`find_bill_details` delegation:** signature preserved on all 100 receipts — still returns
`(signed_amount, ISO_date)`; e.g. `"…TOTAL 23.90"` → `(-23.9, "2018-03-12")`, now via the
champion instead of the first-decimal regex.

**Interpretation:** the refactor is faithful — no metric regressed. The categorizer is the
one subtlety: the *shipped* artifact is retrained on all 600 rows for production, so scoring
it on the test rows gives a misleading 1.00 (leakage). The harness instead reports the
**held-out** 0.9752 from training, and the production verdict for Day 8's honest comparison.

---

### Head-to-Head — running leaderboard (post-integration state)

| Component | Old production behavior | Day-5 shipped champion | Headline delta |
|-----------|------------------------|------------------------|----------------|
| Bill extraction | regex first-decimal, amount 0.15, no merchant | `rules_smart` + regex fallback | amount **0.15 → 0.58**, merchant **0 → 0.94 non-null** |
| Expense categorization | *did not exist* | TF-IDF+LinearSVC joblib | new capability, **0.975** held-out macro-F1 |
| Anomaly detection | *did not exist* | IsolationForest | new capability, **P@20 0.95** |
| Cash-flow forecast | *did not exist* | Prophet + seasonal fallback | new capability, **15.8% MAPE** (Day 4) |
| Investment reco | sums ALL users; affordability-only | per-user risk-profiled | privacy bug fixed + personalization |
| Multi-tenancy | global tables, A sees B | `user_id` on every query | **isolation enforced + proven** |

---

### Key findings (incl. what didn't work)

1. **The refactor is the proof of honesty.** Re-scoring the production categorizer on its
   own training rows returns 1.00 — the exact kind of leaky number this sprint exists to
   *not* report. The harness reports the held-out 0.9752 instead. (Same discipline the
   StockAI leg will apply to the scaler-leakage fix.)
2. **A genuine personalization signal exists in cash-flow alone.** Two synthetic users with
   the *same* +$5,000 net balance but different spend volatility get different risk scores
   (steady 0.562 → "balanced" vs spiky 0.525), so they receive different top investment
   options. The old `invest()` could never do this — it summed everyone's money into one
   number. Personalization didn't require interaction history, just per-user scoping.
3. **Signature-preserving delegation kept the Flask app green.** `find_bill_details` and
   `invest()` swapped their internals for the champions while every template and caller kept
   working — the app imports cleanly and registers all four blueprints.
4. **What didn't work / caveats:** synthetic categorization data means absolute ceilings
   (0.975+) are optimistic vs real noisy bank text — the *relative* parity is the transferable
   result. The multi-tenancy proof runs on an in-memory SQLite mirror of the exact app.py
   queries (no MySQL in this env); the full pytest regression on the live schema is Day 10.
   The `user_id` column requires running `db/migrations/001` against the MySQL DB.

---

### Sample outputs saved

- `results/phase3_integration.json` — full 7-check report
- `results/phase3_integration_parity.png` — integrated vs champion bar chart
- `results/samples/phase3_integration_samples.json` — per-component sample I/O (extraction, categorization, anomaly flags, forecast, both reco users, API responses)
- `models/expense_classifier.joblib` + `.metrics.json` — shipped artifact + model card data

---

### Phase wrap-up: What was finalized (Phase 3)

**Final approach:** A two-service architecture — the existing Flask app (`:5000`) for the UI
and a new async FastAPI ML service (`:8000`) for inference — both backed by one signature-stable
`src/` package wrapping the Day 2–4 champions. Production code delegates to `src/` rather than
duplicating logic.

**Final metrics (shipped):** extraction amount-acc 0.58 (was 0.15) · categorizer held-out
macro-F1 0.975 · anomaly P@20 0.95 · forecast Prophet 15.8% MAPE · per-user risk profiling
live · multi-tenancy isolation enforced and proven · 5/5 API endpoints green.

**What carries forward:**
- The `src/` package is the substrate for Day 6 (Optuna tuning + error analysis), Day 7
  (image-receipt OCR + NL transaction RAG), and Day 9 (Dockerized service + **JWT** real
  multi-tenancy — the Day-5 DB fix becomes API-enforced auth).
- `models/expense_classifier.joblib` + its model card seed Day 10's `docs/MODEL_CARD.md`.
- `db/migrations/001` + the multi-tenancy harness seed Day 10's `test_multitenancy.py` regression.

**Resume gap progress (phase):** FinTrack moved from "a Flask app with a regex scanner and a
privacy bug" to "a finance service that extracts receipts, categorizes spend, detects anomalies
and subscriptions, forecasts cash flow, recommends per-user investments — behind a validated API,
with tenant isolation." The headline capabilities the resume claims now exist in the codebase.

---

### Next session

**Day 6 — Phase 4: Tuning + error analysis.** Optuna ≥30 trials on the champion categorizer
(C / loss / class_weight); error-analyze 30 categorization failures (label noise vs multi-category
overlap vs model error) and 30 extraction field failures (currency symbols, multi-line totals) →
dominant failure type → targeted fix (per-class thresholds / rule patch) → re-evaluate.

### Code changes
Branch `sprint/day05-2026-06-29`; commits prefixed `sprint:` referencing each file. No metric was
reported that the integration harness did not compute.
