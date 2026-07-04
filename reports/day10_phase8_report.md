# FinTrack Production Upgrade — Day 10 / 10

**Phase 8 — Tests + README + demo + PROJECT COMPLETE**
**Date:** 2026-07-04

---

## Resume gap progress

**Gap:** the repo *looked* like an ML product but had zero tests, a README that
described a regex as a "smart scanner", no model documentation, and no way for an
interviewer to see any of the ten days of work actually run.

**Today's contribution:** shipped the production-quality closeout — **66 passing
pytest tests** across 7 files (spec asked for 40+), a README rewritten as a
mini-engineering report with results tables + an architecture diagram, a
`docs/MODEL_CARD.md` for the categorizer, and a **runnable 60-second `demo.py`**
that executes the whole upgraded pipeline live with no server/DB/API-key. The
project is now demonstrably real, measured, and honest about its limits.

## Files touched

| File | Change |
|---|---|
| `tests/conftest.py` | new — repo-root sys.path + eval-receipt / per-user txn fixtures |
| `tests/test_extraction.py` | 12 tests — totals vs subtotal/tax, GST two-column, empty/non-receipt, EU dates, `find_bill_details` signature |
| `tests/test_categorization.py` | 10 tests — artifact load, disambiguation, unseen merchant, keyword fallback |
| `tests/test_anomaly.py` | 8 tests — injected outlier, inflow-ignore, recurring, duplicate |
| `tests/test_forecast.py` | 6 tests — history guard, horizon contract, non-negativity |
| `tests/test_multitenancy.py` | 7 tests — **regression: user A cannot see/delete user B's data** |
| `tests/test_auth.py` | 9 tests — password hashing, JWT round-trip, tampered-token 401 |
| `tests/test_api.py` | 14 tests — golden path + edge cases + HTTP-level tenant isolation |
| `README.md` | full rewrite — audit table, results tables, architecture diagram, quickstart |
| `docs/MODEL_CARD.md` | new — categorizer model card (intended use, eval, limitations) |
| `docs/DEMO.md` | new — 60-second demo talk-track |
| `demo.py` | new — runnable end-to-end terminal demo |

## Setup

- **Compute:** local CPU, Python 3.11 venv. Test deps: `pytest`, `httpx` (TestClient).
- **Test substrate:** public SROIE receipts (`data/eval/receipts.jsonl`) + synthetic
  transactions; in-memory `UserStore`/`TransactionStore` so tests run anywhere with
  no MySQL/Redis. Trained artifact `models/expense_classifier.joblib` (Day-3).

## Experiments

### Experiment 1 — behavioural test coverage of every component

- **Hypothesis:** each Day-2→9 component can be pinned by fast, deterministic tests
  that would fail if the audited bug ever regressed.
- **Method:** 66 tests targeting real signatures (not mocks), run with `pytest`.

| Suite | Tests | Result |
|---|---|---|
| Extraction | 12 | pass |
| Categorization | 10 | pass |
| Anomaly | 8 | pass |
| Forecast | 6 | pass |
| Multi-tenancy | 7 | pass |
| Auth | 9 | pass |
| API | 14 | pass |
| **Total** | **66** | **66 passed in 5.4s** |

- **Interpretation:** the two audited security/correctness bugs now have explicit
  regression locks: `test_multitenancy.py` proves user A cannot read/delete user
  B's rows at the store layer, and `test_api.py` proves the same over HTTP (user B
  GET sees 0 of A's rows; DELETE of A's id → 404).

### Experiment 2 — the demo tells the *honest* story, verified live

- **Hypothesis:** the anomaly "rent" insight is demonstrable, not just asserted.
- **Method:** `demo.py` contrasts a global MAD z-score vs the shipped IsolationForest
  on a 6-month stream with a stable rent and one genuine $600 dining anomaly.

| Detector | Top-2 flags | Verdict |
|---|---|---|
| Global z-score (naive) | LANDLORD −1800, LANDLORD −1800 | **false alarm** on recurring rent; misses the real one |
| IsolationForest (context-relative, shipped) | FANCY RESTO −600 (*24× dining median*), LANDLORD −1800 | catches the genuine anomaly, ranks it #1 |

- **Interpretation:** I initially wrote the demo to claim "rent is never flagged",
  ran it, and it flagged rent — because in a *single user's* stream rent is rare
  and large. Rather than fake it, I rebuilt the demo around the *true* mechanism:
  the anomaly that matters (−600 dining) is small in dollars but extreme for its
  category, and only context-relative features rank it above rent. **The demo now
  proves the exact thing the Day-4 AP-0.98 result measured.**

## Head-to-Head Comparison (running leaderboard — final)

| Component | Original | Champion (shipped) | Frontier ceiling |
|---|---|---|---|
| Receipt amount acc (SROIE-100) | 0.15 (regex) | **0.83** (rules_smart, $0) | 1.00 (LLM, ~$3/1k) |
| Categorizer macro-F1 (in-dist) | 0.658 (keyword) | **0.975** (TF-IDF+LinearSVC, $0) | 1.00 (LLM) |
| Anomaly AP | 0.400 (robust-z) | **0.979** (IsolationForest) | — |
| Cash-flow MAPE | 26.3% (naive) | **15.8%** (Prophet) | — |
| Multi-tenancy | broken (no filter) | **user_id-scoped + JWT + regression test** | — |
| Tests | 0 | **66** | — |

## Key findings

1. **66 tests > 40 required**, with the two audited bugs (multi-tenancy leak,
   regex mis-extraction) each pinned by a dedicated regression test.
2. **Honesty survived contact with reality:** the anomaly demo was rewritten when
   the first version's claim didn't reproduce; the final demo verifies the real
   context-relative insight instead of asserting a convenient one.
3. **The whole 10-day arc runs in one command** (`python demo.py`, ~60s), which is
   what makes it interview-ready — extraction, categorization, anomaly, and
   tenant-isolation all execute live and end on the honest novel-merchant caveat.

### What didn't work (and why)

- The first `demo.py` used Unicode box-drawing/arrow glyphs → `UnicodeEncodeError`
  on the Windows cp1252 console. Fixed by using ASCII only, so it runs on any
  terminal.
- The first anomaly demo overclaimed ("rent is NOT flagged"); IsolationForest
  flagged it on a small single-user stream. Root cause: rent is genuinely a rare,
  large row for one user, so it's a global outlier. Fixed by demonstrating the
  *ranking* difference on a contextually-extreme anomaly instead.

## Sample outputs saved

- `demo.py` terminal transcript (reproduced in this report)
- `docs/MODEL_CARD.md`, `docs/DEMO.md`
- Full test run: `pytest tests/ -q` → `66 passed`

---

## Phase wrap-up — What was finalized (Day 10 = PROJECT COMPLETE)

**Final approach.** FinTrack is now a multi-tenant Document-AI finance service:
`pdfplumber`-backed extraction → keyword-anchored `rules_smart` extractor →
TF-IDF+LinearSVC categorizer → IsolationForest anomaly + recurring/duplicate
detectors → Prophet/seasonal-naive cash-flow forecast → per-user risk-profiled
investment recommender, all behind a JWT-scoped async FastAPI service (Redis
cache + telemetry), with the original Flask app delegating to the same `src/`
champions while keeping its signatures.

**Final metrics (honest, shipped):**
- Receipt extraction: amount **0.83**, date 0.87, merchant 0.77 (vs regex 0.15/0.49/0.00)
- Categorizer: macro-F1 **0.975** in-distribution, **0.24** on novel merchants (the caveat)
- Anomaly AP **0.979**; recurring F1 0.994; forecast MAPE **15.8%**
- Multi-tenancy: enforced + regression-tested; **66 tests green**

**What carries forward (future-sprint seeds):** OCR for photographed receipts
(Day-7 started this), bank-statement CSV import + reconciliation, PostgreSQL +
Alembic migrations to retire raw SQL, calibrated categorizer confidence, and the
active-learning review queue promoted to a scheduled weekly retrain.

**Resume gap progress — CLOSED.** The line "built a smart bill scanner" is now
defensible: *"replaced a regex that grabbed the wrong number 85% of the time with
a measured extractor, added the missing ML (categorizer / anomaly / forecast),
fixed a multi-tenancy data-leak, wrapped it in a tested JWT API, and benchmarked
every component against a frontier LLM — including where the small model loses."*

## Next

FinTrack cycle complete (Days 1–10 done, PRs #1–#12). The sprint advances to
**Project B — CineSemantics** on **2026-07-05 (Day 1)**: audit the
"recommendation engine" that has zero offline evaluation, build the eval harness +
honest baseline metrics, and align MovieLens interactions to the TMDB catalog.

## Code changes

Branch `sprint/day10-2026-07-04`; commit `sprint: add 66-test suite + README/model-card/demo — PROJECT COMPLETE (Day 10 Phase 8)`; PR base `main`, squash-merged.
