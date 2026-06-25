# FinTrack Production Upgrade — Day 1 of 10

**Date:** 2026-06-25 · **Phase 1 — Audit + eval sets + baseline**
**Field:** Document AI + Financial NLP (multi-component)

## Resume gap progress

**Gap:** The repo describes a "smart bill scanner" and an "investment
recommender," but both are rule/regex heuristics with no evaluation, plus a
broken multi-tenancy model and a machine-locked file path. There was no eval
harness, no baseline, and no expense categorizer — so no honest claim could be
made about any component.

**Today's contribution:** Audited the three core files against the actual code,
built two reproducible eval sets (100 public SROIE receipts + 600 synthetic
labeled transactions), and measured honest baselines for the regex extractor and
keyword categorizer. These numbers are the floor every Phase-2 model must beat.

## Files touched

| File | Lines / scope | Change |
|------|---------------|--------|
| `docs/COMPONENT_AUDIT.md` | new | Evidence-backed audit of all 4 defects |
| `data/eval/build_transactions.py` | new | Seeded synthetic transaction generator |
| `data/eval/receipts.jsonl` | new | 100 SROIE receipts w/ GT amount/date/merchant |
| `data/eval/transactions.csv` | new | 600 txns, 10 classes |
| `results/run_baselines.py` | new | Runs the REAL `find_bill_details` + keyword/majority |
| `results/baseline_metrics.json` | new | Honest baseline numbers |
| `results/*.png` | new | Field-accuracy + per-class-F1 charts |
| `extract_bill.py` / `invest.py` / `app.py` | read-only (lines 43, 28, 20–22, 65–91) | Audited, not yet modified |

## Setup

- **Compute:** CPU, Python 3.11 venv. No GPU needed for Day 1.
- **Datasets:** SROIE (`darentang/sroie`, public, 347 test receipts → first 100
  with a labeled total) via the HF datasets-server; synthetic transactions
  generated locally (seed `20260625`), media-discipline compliant (no real
  financial data).
- **Components measured:** `find_bill_details` (the real function, imported, not
  reimplemented) and keyword/majority categorization baselines.

## Experiments

### Experiment 1 — Receipt extraction baseline (the real regex)
- **Hypothesis:** `find_bill_details` taking the *first* `\d+\.\d{2}` match grabs
  line items, not totals, and extracts no merchant.
- **Method:** Import the actual function from `extract_bill.py`; run it on 100
  SROIE receipts; compare to GT amount/date/merchant.

| Field | Regex accuracy |
|-------|---------------:|
| Amount | **0.15** |
| Date | 0.49 |
| Merchant | **0.00** (no merchant field) |
| Exact (amount + date) | **0.07** |

- **Interpretation:** Confirmed — wrong number on **85%** of real receipts;
  merchant never extracted; date swaps day/month. This is the component to replace.

### Experiment 2 — Expense categorization baseline
- **Hypothesis:** Keyword matching beats majority but fails when the literal
  category word is absent from the merchant string.
- **Method:** Majority-class and keyword-rule predictors on 600 labeled txns;
  macro-F1 + per-class F1.

| Baseline | Accuracy | Macro-F1 |
|----------|---------:|---------:|
| Majority (`shopping`) | 0.162 | 0.028 |
| Keyword | 0.638 | **0.675** |

- **Interpretation:** Keyword weakest on `utilities` (F1 0.48, *not* rare) and the
  `other` catch-all (0.35), strong on `income` (0.73, *rare*). Rarity is not the
  predictor of failure — **literal-word-absence is.**

## Head-to-Head Comparison (running leaderboard — Day 1)

| Component | Method | Primary metric | Score | Status |
|-----------|--------|----------------|------:|--------|
| Extraction | regex `find_bill_details` | amount accuracy | 0.15 | baseline |
| Extraction | regex `find_bill_details` | date accuracy | 0.49 | baseline |
| Categorization | majority class | macro-F1 | 0.028 | baseline |
| Categorization | keyword rules | macro-F1 | 0.675 | baseline |

## Key findings

1. **The "smart" scanner gets the amount right 15% of the time** on real receipts —
   it grabs the first decimal (a line item), not the total.
2. **It extracts no merchant and swaps day/month dates** — three of the four fields
   a finance app actually needs are unusable.
3. **Keyword categorization's blind spot is literal-word-absence, not rarity.**
   `utilities` (PG&E, Con Edison, Comcast) scores 0.48 despite 51 samples; `income`
   scores 0.73 on 21. The learned model's expected win is precisely here.
4. **Keyword collisions actively mislead:** `ENTERPRISE RENT-A-CAR` → `rent`.

## What didn't work / honest notes

- Most public HF transaction-categorization datasets are gated (HTTP 401) or lack
  clean category labels, so categorization uses a **seeded synthetic** set (allowed
  by media discipline). Receipt extraction uses fully **public** SROIE data.
- SROIE totals are clean US-style decimals, which is a *fair* test for the
  US-style regex — the 15% is not a currency-format artifact; failures are
  line-item grabs and day/month swaps (see samples).

## Sample outputs saved

- `results/baseline_metrics.json`
- `results/samples/extraction_baseline_samples.json` (10 receipts, pred vs GT)
- `results/samples/categorization_baseline_samples.json` (10 keyword misses)
- `results/baseline_extraction_field_accuracy.png`
- `results/baseline_categorization_f1.png`

## Next day (Day 2 — Phase 2a)

Receipt-extraction bake-off: regex vs `pdfplumber`+rules vs Donut
(`naver-clova-ix/donut-base-finetuned-cord-v2`) vs LayoutLMv3 vs Claude Vision /
GPT-4o-mini zero-shot on the same 100 receipts. Report field accuracy, exact-match,
runtime, and cost per doc → `results/phase2a_extraction.csv`.

## Code changes

No production code modified today (audit + baseline only). `extract_bill.py`,
`invest.py`, `app.py` were read and line-referenced; modifications begin Day 5
(champion integration), with the multi-tenancy and poppler-path fixes.
