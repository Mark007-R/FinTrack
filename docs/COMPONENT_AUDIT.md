# FinTrack — Component Audit (Day 1)

**Date:** 2026-06-25 · **Scope:** `extract_bill.py`, `invest.py`, `app.py`
**Purpose:** Establish, with evidence, what the current "smart" components actually
are before the 10-day production upgrade. Every claim below is backed by a line
reference and a measured baseline (`results/baseline_metrics.json`).

---

## 1. Bill extraction is a naive regex, not a model

`extract_bill.py:find_bill_details` (line 43) is advertised as a "smart bill
scanner." It is not ML and has no document understanding:

```python
amount_match = re.findall(r'\b\d+\.\d{2}\b', text)   # line 44
amount = float(amount_match[0]) if amount_match else 0.0   # line 47 — takes the FIRST match
```

- It takes the **first** `\d+\.\d{2}` substring as the bill amount — on a real
  receipt that is almost always a line-item price or quantity, **not the total**.
- It extracts **no merchant** at all — the function returns only `(amount, date)`.
- The date regex (line 45) is format-blind and tries `%m/%d/%Y` before `%d/%m/%Y`,
  so it silently swaps day/month on DD/MM receipts.

**Measured on 100 real SROIE receipts** (`darentang/sroie` test split):

| Field | Regex accuracy |
|-------|---------------:|
| Amount | **15%** |
| Date | 49% |
| Merchant | **0%** (not extracted) |
| Exact match (amount **and** date) | **7%** |

Concrete failures (`results/samples/extraction_baseline_samples.json`):
- grabbed `3.50` (a line item) instead of the `436.20` total;
- grabbed `2.00` instead of `7.30`;
- read `09/02` as Feb 9 instead of Sept 2 (day/month swap).

> **The regex pulls the wrong number on 85% of real receipts.** This is the
> component to replace (Day 2: Donut / LayoutLMv3 / Vision-LLM comparison).

## 2. Multi-tenancy is broken — every user sees every user's money

There is no `user_id` column filter anywhere in the transaction path:

- `app.py:dashboard()` (lines 71–91) — `INSERT`, `DELETE WHERE id=%s`, and
  `SELECT * FROM transactions ORDER BY date DESC` all run against a **global**
  `transactions` table. Any logged-in user can read and delete **any** other
  user's transactions by row id.
- `invest.py:invest()` (lines 20–22) — `SELECT amount FROM transactions` sums
  **every user's** transactions into one `total_balance`, then recommends
  products off that shared aggregate.

This is the headline security defect. Day 5 fixes it (per-user filtering on every
query, keeping signatures) and Day 10 adds a regression test (`test_multitenancy.py`:
user A cannot see or delete user B's data).

## 3. The poppler path is hardcoded to one machine

`extract_bill.py:28`:
```python
poppler_path = r'C:\poppler-24.07.0\Library\bin\pdftotext.exe'
```
The extractor shells out to an absolute Windows path that exists only on the
author's machine — it cannot run anywhere else. Day 5 removes it (env var /
`pdfplumber`).

## 4. There is no expense categorizer at all

Transaction `description` is free text typed by the user; nothing categorizes it.
The "categories" a finance app needs (groceries / dining / transport / …) do not
exist in the codebase. This is the headline **feature add**.

**Baseline** on 600 synthetic transactions across 10 classes (seed `20260625`,
deliberate class imbalance, realistic raw-feed merchant strings):

| Baseline | Accuracy | Macro-F1 |
|----------|---------:|---------:|
| Majority class (`shopping`) | 16.2% | 0.028 |
| Keyword / rule matching | 63.8% | **0.675** |

Weakest keyword classes: `other` (F1 0.35, catch-all) and **`utilities` (F1 0.48)**.

> **Genuine insight:** keyword matching fails not on *rare* classes but on classes
> whose **literal category word never appears in the merchant string**. `utilities`
> has 51 samples (not rare) yet scores 0.48, because real utility billers — `PG&E`,
> `CON EDISON`, `COMCAST`, `AT&T` — never contain the word "utility." Meanwhile
> `income` (only 21 samples, genuinely rare) scores 0.73 because its merchants do
> carry literal cues (`PAYROLL`, `DEPOSIT`, `REFUND`, `DIVIDEND`). Keyword matching
> also mis-fires on **collisions**: `ENTERPRISE RENT-A-CAR` (transport) is labeled
> `rent` purely on the substring. A learned TF-IDF/embedding model (Day 3) is
> expected to beat this exactly where keywords are blind.

## 5. Other audited issues (tracked for later days)

- `invest.py` uses `mysql.connector` while the rest of the app uses `pymysql`
  (inconsistent driver; `mysql.connector` is imported but missing from
  `requirements.txt`). Day 5 consolidates onto `pymysql`.
- `invest()` recommendations are rule-matches on `min_investment <= total_balance`
  — not personalized, not risk-aware. Day 5 makes it per-user + risk-profile.
- No `src/`, `tests/`, `results/`, `data/` layout; `requirements.txt` is only
  `Flask / pymysql / werkzeug`. Day 5–10 add the production structure.

---

## Eval sets built today (`data/eval/`)

| File | Content | Source |
|------|---------|--------|
| `receipts.jsonl` | 100 receipts: reconstructed OCR text + GT amount/date/merchant | SROIE (`darentang/sroie`, public) |
| `transactions.csv` | 600 transactions labeled across 10 spending categories | Synthetic, seeded (`build_transactions.py`) — media-discipline compliant |

These are the fixed targets every Phase-2 strategy must beat.
See `results/baseline_metrics.json` and the charts in `results/`.
