# FinTrack — Document-AI Personal Finance Manager

FinTrack started as a Flask personal-finance app whose "smart bill scanner" was a
one-line regex and whose multi-tenancy was **broken** — every user could see and
delete every other user's transactions. This repo is a 10-day production upgrade
that turned it into a **measured, multi-tenant ML service**: a real receipt
extractor, a trained expense categorizer, anomaly + recurring-charge detection,
cash-flow forecasting, and per-user risk-profiled investment recommendations —
each benchmarked against a baseline **and** a frontier LLM, wrapped in a
JWT-scoped FastAPI service, and covered by 66 tests.

> Data discipline: every experiment uses **public** (SROIE receipts) or
> **synthetic** transaction data. No real personal financial data is used anywhere.

---

## The audit — what was actually wrong

| # | Finding | Where | Fixed |
|---|---|---|---|
| 1 | "Smart bill scanner" is a naive regex — takes the **first** `\d+\.\d{2}` match as the total (grabs phone numbers, line items, tax) | `extract_bill.py:find_bill_details` | Day 2/5 |
| 2 | **Multi-tenancy broken** — `dashboard()` and `invest()` query a global table with **no `user_id` filter**; any user sees/deletes any other's data | `app.py`, `invest.py` | Day 5 + 9 |
| 3 | Poppler path **hardcoded** to one machine (`C:\poppler-24.07.0\...`) | `extract_bill.py:28` | Day 5 |
| 4 | **No expense categorizer at all** — `description` is untyped free text | — (missing) | Day 3 |
| 5 | `invest()` sums **all users'** money into one balance; rule-match, no personalization | `invest.py` | Day 5 |

Full write-up: [`docs/COMPONENT_AUDIT.md`](docs/COMPONENT_AUDIT.md).

## Headline results

### Receipt extraction (100 SROIE receipts — field accuracy)

| Method | amount | date | merchant | exact (amt+date) | sec/doc | $/1k |
|---|---|---|---|---|---|---|
| `find_bill_details` regex (original) | 0.15 | 0.49 | 0.00 | 0.07 | 0.0002 | 0 |
| **`rules_smart` (champion, local)** | **0.83**¹ | **0.87** | **0.77** | 0.48 | 0.0002 | **0** |
| Claude Opus zero-shot (frontier ceiling) | 1.00 | 0.85 | 0.95 | 0.85 | ~1.8 | ~$3 |

¹ 0.58 at Day-2; the Day-6 GST two-column-total fix lifted amount accuracy to **0.83**.
The champion recovers **97%** of the regex's field errors at **$0** — the LLM is kept
as a low-confidence fallback for novel layouts, not the default.

### Expense categorization (600 synthetic txns, 10 classes — macro-F1)

| Method | macro-F1 | accuracy | fit (s) | $/1k |
|---|---|---|---|---|
| Keyword baseline (original) | 0.658 | 0.606 | 0 | 0 |
| **TF-IDF(word+char) + LinearSVC (champion)** | **0.975** | **0.978** | **0.08** | **0** |
| DistilBERT fine-tune (ceiling) | 0.994 | 0.994 | 71.7 | 0 |
| Claude Opus zero-shot (frontier) | 1.00 | 1.00 | — | ~$0.14 |

Champion = **98% of DistilBERT's F1 at 1/900th the train time, $0 inference**. Model
card: [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

### Anomaly / recurring / forecast (Day-4)

| Task | Champion | Metric |
|---|---|---|
| Anomaly | IsolationForest (context-relative features) | **AP 0.979** vs robust-z 0.400 |
| Recurring / subscription | merchant-cadence + amount-stability cluster | **F1 0.994** |
| Cash-flow forecast | Prophet (≥24 mo) / seasonal-naive fallback | **MAPE 15.8%** vs naive 26.3% |

> **Genuine insight (Day-4):** a *global* amount threshold flags your own **rent** as
> fraud. Switching to context-relative features (amount ÷ category-median) is what
> jumps anomaly AP from 0.40 → 0.98.

### The honest caveat (Day-8 frontier + ablation)

On a held-out set of **novel merchants**, the $0 categorizer collapses to
**macro-F1 0.24** while the LLM holds **1.00** — TF-IDF memorizes training
vocabulary. The specialized model wins on cost/latency/consistency *in
distribution*; the LLM wins zero-shot on the long tail. FinTrack ships the cheap
model as default and routes low-confidence rows to the LLM + an active-learning
review queue (Day-7). See [`results/frontier_comparison.csv`](results/frontier_comparison.csv).

## Architecture

```
                         ┌───────────────────────────────────────────────┐
   PDF / photo  ─────▶   │  extract_text_from_pdf (pdfplumber, no hardcode)│
                         └───────────────────┬───────────────────────────┘
                                             ▼
   ┌──────────────────────────── FastAPI  (api.py, :8000) ──────────────────────────┐
   │  JWT auth  →  get_current_user  →  per-user scope (no body-supplied user_id)     │
   │                                                                                  │
   │   /extract    → src/extraction   ReceiptExtractor (rules_smart + regex fallback) │
   │   /categorize → src/categorization ExpenseClassifier (TF-IDF+LinearSVC, cached)  │
   │   /anomaly    → src/anomaly       IsolationForest + recurring + duplicate        │
   │   /forecast   → src/forecast      Prophet / seasonal-naive                       │
   │   /recommend  → src/reco          per-user risk profile → ranked options         │
   │   Redis cache (extract/categorize) · per-request telemetry → logs/telemetry.jsonl │
   └──────────────────────────────────────────────────────────────────────────────────┘
             ▲                                                     ▲
   Flask app (app.py) — user_id-scoped SQL          Streamlit dashboard (dashboard/)
   find_bill_details / invest() delegate to src/    heat-map · anomalies · cash-flow
```

The original Flask app keeps working: `find_bill_details` and `invest()` retain
their signatures and **delegate** to the `src/` champions, so the templates are
untouched while the logic is upgraded and tenant-scoped.

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate      # (Windows) or source .venv/bin/activate
pip install -r requirements-api.txt                 # API/service stack
python -m src.categorization.train                  # build models/expense_classifier.joblib

uvicorn api:app --port 8000                          # ML API  → http://localhost:8000/docs
# or the full stack:
docker compose up                                    # FastAPI + Redis
```

```bash
# reproduce any day's experiments (research stack):
pip install -r requirements-experiments.txt
python results/run_day2_extraction.py               # extraction bake-off
python results/run_day3_categorize.py               # categorizer bake-off
python results/run_day4_anomaly_forecast.py         # anomaly + forecast
```

## Tests

**66 tests, all passing** (`pytest tests/ -q`):

| File | Tests | Covers |
|---|---|---|
| `test_extraction.py` | 12 | totals vs subtotal/tax, GST two-column, empty/non-receipt, EU dates, signature preserved |
| `test_categorization.py` | 10 | artifact load, disambiguation, unseen merchant, keyword fallback |
| `test_anomaly.py` | 8 | injected outlier, inflow-ignore, recurring, duplicate charges |
| `test_forecast.py` | 6 | history guard, horizon contract, non-negativity |
| `test_multitenancy.py` | 7 | **regression: user A cannot see/delete user B's data** |
| `test_auth.py` | 9 | password hashing, JWT round-trip, tampered-token 401 |
| `test_api.py` | 14 | golden path + edge cases + HTTP-level tenant isolation |

## Repo layout

```
src/            extraction · categorization · anomaly · forecast · reco · serving (auth/store/cache/telemetry)
api.py          async FastAPI ML service (JWT, cache, telemetry)
app.py          original Flask app (now user_id-scoped)
dashboard/      Streamlit ops dashboard
results/        every experiment's runnable script + CSV + charts + samples/
reports/        day01–day10 daily research reports
docs/           COMPONENT_AUDIT.md · MODEL_CARD.md
tests/          66 pytest tests
```

## License

MIT — see [LICENSE](LICENSE).
