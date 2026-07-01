# FinTrack Production Upgrade — Day 7 / 10

**Phase 5: Robustness + reach**
**Date:** 2026-07-01

---

## Resume gap progress

**Gap:** the Day-2/6 champion extractor and Day-3/6 categorizer were only ever tested
in the easy case — clean digital-PDF text (English) for extraction, in-distribution
merchant strings for categorization — and FinTrack had no natural-language analytics
at all. A resume line that says "receipt extraction / expense categorization" has to
survive phone photos, non-English receipts, and hard merchant strings, and a finance
app should answer "how much did I spend on food last month?"

**Today's contribution:** three robustness/reach tracks, each measured honestly:
1. **Image-receipt OCR robustness** — quantified how far field accuracy drops as
   capture quality degrades from clean scan → phone photo → faded → rotated.
2. **Active-learning loop** — closed the Day-6 no-keyword `model_failure` gap *via data*
   as predicted: hard-case accuracy **0.52 → 0.89** with uncertainty sampling vs **0.61**
   for random at the same label budget, with **zero** in-distribution regression.
3. **Multilingual/edge receipts** — shipped a multilingual gazetteer + a dot-date fix
   into the champion extractor: non-English amount **0.83 → 1.00**, date **0.58 → 1.00**,
   **0/60** firings on English (proven safe), and Day-5 integration parity still green.
4. **NL transaction query (RAG)** — a grounded parser: slot accuracy **1.00**, answer
   accuracy **1.00** on true labels, **100%** grounded (every answer cites real rows,
   0% hallucination).

---

## Files touched

| File | Change |
|------|--------|
| `src/extraction/extractor.py` | **Shipped:** multilingual TOTAL gazetteer appended to `TOTAL_KW` (German/ES/FR/IT), dot-separated day-first date support in `_norm_date` (`%d.%m.%Y`), multilingual date labels (`datum`/`fecha`/`data`) in `_rules_smart`. All additive; English priority preserved. |
| `data/eval/build_day7_pool.py` | New — builds a 242-row HARD active-learning pool from public brand names (no_kw / overlap), deterministic. |
| `data/eval/transactions_pool.csv` | New — the 242-row hard pool. |
| `results/run_day7_ocr_robustness.py` | New — render→degrade→OCR→extract→score across 5 capture profiles. |
| `results/run_day7_active_learning.py` | New — uncertainty-sampling AL loop vs random control. |
| `results/run_day7_multilingual_edge.py` | New — non-English before/after + English regression guard. |
| `results/run_day7_rag.py` | New — grounded NL-query parser + executor + eval. |
| `results/phase5_io.py` | New — shared idempotent long-format writer for `phase5_robustness.csv`. |

---

## Setup

- **Compute:** CPU, existing `.venv` (Python 3.11).
- **Datasets:** 100 SROIE receipts (`data/eval/receipts.jsonl`), 600 labeled transactions
  (`data/eval/transactions.csv`), 242 synthetic hard merchant strings
  (`data/eval/transactions_pool.csv`) — all public/synthetic per the media-discipline rule.
- **OCR engine note (honesty):** a real OCR engine (`rapidocr-onnxruntime`) could not be
  installed in the sandbox — the 15 MB wheel timed out repeatedly on the network. The OCR
  step therefore runs a **calibrated character-level OCR-noise model** (documented OCR
  confusion set O↔0/l↔1/S↔5/B↔8/rn↔m + token dropout, per-profile CER set to published
  Tesseract bands). The rendering, degradation, extractor, and scoring are all real; a live
  engine is a one-line swap (`_load_ocr()` already tries it first and labels which path ran).
  The reported numbers describe the *extractor's* sensitivity to text quality, which is the
  research question, not a claim about a specific engine's CER.

---

## Experiment 1 — Image-receipt OCR robustness

**Hypothesis:** the keyword-anchored extractor was tuned on clean PDF text; degraded
captures will hurt the *label-anchored* fields (date, amount) more than the token-overlap
field (merchant).

**Method:** render each of 60 SROIE receipts, apply 4 capture profiles, recover text,
run the shipped `extract_fields`, score amount/date/merchant with the exact Day-2 scorers.
`gold_text` = extractor on ground-truth text (the ceiling).

| Profile | amount | date | merchant |
|---------|:------:|:----:|:--------:|
| gold_text (ceiling) | 0.72 | 0.83 | 0.72 |
| clean_scan | 0.65 | 0.73 | 0.72 |
| phone_photo | 0.58 | 0.45 | 0.50 |
| faded | 0.33 | 0.33 | 0.38 |
| rotated | **0.35** | **0.22** | 0.27 |

**Interpretation:** the drop is not uniform. **Date collapses fastest** (0.83 → 0.22 under
rotation) because it depends on both an intact `date` label *and* an intact `dd/mm/yyyy`
digit string — OCR corrupts either and the field is lost. **Amount degrades more gracefully**
(0.72 → 0.35) because of the `max(amount)` fallback: even when the TOTAL anchor is corrupted,
the largest number on the receipt is often still the total. Merchant (token-overlap, ≥50%)
is the most forgiving at moderate degradation. Takeaway for production: a phone-photo path
should **deskew before OCR** (rotation is the single worst profile) and add a **confidence
gate** that routes low-quality captures to the LLM-Vision fallback (Day-2 ceiling 1.00).

Chart: `results/phase5_ocr_field_accuracy.png` · samples: `results/samples/phase5_ocr_samples.json`
· 5 degraded receipt images: `results/samples/day7_receipt_images/`.

---

## Experiment 2 — Active-learning loop (categorizer)

**Hypothesis (from Day-6):** the categorizer's hard-case failures are a *data* problem, not
a hyperparameter problem — labeling the hard rows fixes them, and uncertainty sampling should
find those rows faster than random.

**Method:** seed = 81 stratified easy in-dist rows; pool = 617 rows (in-dist + 171 hard);
fixed test = 71 hard + 73 in-dist. Two arms (active = query 32 lowest-confidence rows/round;
random = 32 random), 6 rounds, identical seed. Champion pipeline (TF-IDF word+char + LinearSVC)
refit each round.

| Arm | # labels | hard-case acc | overall macro-F1 | in-dist acc |
|-----|:-------:|:-------------:|:----------------:|:-----------:|
| seed only | 81 | 0.52 | 0.51 | 0.58 |
| **active (final)** | 273 | **0.89** | **0.90** | 0.97 |
| random (final) | 273 | 0.61 | 0.74 | 0.96 |

**Mechanism confirmed:** the active arm's queried batches were **46.9% hard** rows on average
vs **26.1%** for random — uncertainty sampling preferentially surfaces the exact no_kw/overlap
strings the Day-6 error analysis flagged. Result: **+28 points** of hard-case accuracy over
random at the *same* label budget, and macro-F1 0.90 vs 0.74, while in-distribution accuracy
holds at 0.97 (no easy-case regression). This is the Day-6 prediction ("the lever is data
difficulty, not hyperparameters") validated with a proper AL curve.

Chart: `results/phase5_active_learning.png` · curve: `results/phase5_active_learning.csv`
· hard rows AL pulled early: `results/samples/phase5_al_queried_hard.json`.

---

## Experiment 3 — Multilingual / edge receipts (shipped reach)

**Hypothesis:** the English-only anchors fail on non-English receipts; German is the hard case
(`Gesamtbetrag`/`Summe`/`Datum` share no English token).

**Method:** 12 synthetic DE/ES/FR/IT receipts with known total/date/merchant. `baseline` =
pinned pre-Day-7 English-only extractor; `+gazetteer` = the shipped multilingual version.

| Variant | amount | date | merchant |
|---------|:------:|:----:|:--------:|
| baseline (English-only) | 0.83 | 0.58 | 0.92 |
| **+gazetteer (shipped)** | **1.00** | **1.00** | 0.92 |

**Two findings, one a genuine bug:**
1. Latin-language totals (`Total TTC`, `Totale`, `Importe total`) largely worked already
   because they contain the substring `total`. **German was the real gap** and the gazetteer
   closes it (amount 0.83 → 1.00).
2. **Date accuracy was stuck at 0.58 not because of the label language but because European
   receipts print dot-separated day-first dates (`11.03.2024`) that the extractor's date
   parser *never tried* — it only handled `/` and `-`.** Adding `%d.%m.%Y` recovered every
   one (0.58 → 1.00). A silent parse failure, invisible until non-English data was tested.

**Regression guard:** the multilingual TOTAL tokens fired on **0/60** English receipts, and
`old == new` on all 100 English SROIE receipts (amount 0.770 / date 0.870 / merchant 0.770,
identical). Day-5 integration parity re-run: **ALL PARITY OK = True**, extraction amount 0.84,
`find_bill_details` delegation OK, multi-tenancy isolation intact. Safe to ship.

Detail: `results/phase5_multilingual.csv` · samples: `results/samples/phase5_multilingual_samples.json`.

---

## Experiment 4 — Natural-language transaction query (grounded RAG)

**Hypothesis:** a grounded retrieve-then-compute parser can answer analytics questions with
correct numbers and zero hallucination — and its accuracy is bounded by the categorizer, not
the parser.

**Method:** 22 NL queries, each with explicit gold slots (op / category / time-window) and a
gold answer computed from those slots on the data. Categories resolved via the Day-3 champion
categorizer. Two label sources: true labels, and **out-of-fold** cross-val categorizer
predictions (genuinely out-of-sample; resubstitution would be a meaningless 0% error since the
model was trained on this set).

| Metric | Value |
|--------|:-----:|
| slot accuracy (op/category/window parsed correctly) | 1.00 |
| answer accuracy — true labels | 1.00 |
| answer accuracy — **out-of-fold categorizer labels** | **0.77** |
| grounding rate (answer cites ≥1 real transaction) | 1.00 |
| categorizer out-of-fold error | 0.7% (4/600) |

**Genuine insight:** the parser is perfect and the execution is fully grounded (0%
hallucination — the opposite of asking an LLM to do the arithmetic from context), **but a mere
0.7% categorizer labeling error degraded answer accuracy to 77%.** Category totals are
*disproportionately* sensitive to misclassification: one transaction routed to the wrong
category shifts that category's total beyond the 1% tolerance. This is exactly why Experiment
2 matters — improving the categorizer's hard-case accuracy directly improves downstream
analytics reliability. Average F1 is not enough for analytics; you need per-class reliability
(and the AL loop / review queue that produces it).

Detail: `results/phase5_rag.csv` · answers: `results/samples/phase5_rag_answers.json`.

---

## Head-to-head comparison (running leaderboard)

| Component | Day introduced | Easy-case metric | Robustness / reach metric (Day 7) |
|-----------|:--------------:|------------------|-----------------------------------|
| Receipt extraction | Day 2/6 | amount-acc 0.84 (clean EN PDF) | phone-photo 0.58 · rotated 0.35 · **multilingual 1.00** (shipped) |
| Expense categorizer | Day 3/6 | in-dist macro-F1 0.975 | hard-case **0.52 → 0.89** via active learning |
| NL analytics (new) | Day 7 | — | slot 1.00 · answer 1.00 (true) / 0.77 (OOF categorizer) · grounding 1.00 |

---

## Key findings

1. **Rotation is the worst capture profile** for this extractor (date 0.83 → 0.22); deskew
   must come before OCR, and low-quality captures should route to the Vision fallback.
2. **Active learning closed the Day-6 gap via data, as predicted** — +28 pts hard-case
   accuracy over random at equal budget, because uncertainty sampling pulls ~1.8× more hard
   rows per batch. No in-distribution regression.
3. **A silent date-parser bug** (no dot-separated dates) was invisible until non-English data
   was tested — the label language was a red herring; the format was the real fault.
4. **Grounded RAG is only as good as the categorizer:** 0.7% labeling error → 23% of analytic
   answers wrong. Category aggregates amplify sparse misclassification. Average F1 ≠ analytics
   reliability.

## What didn't work (and why)

- **Fetching a real OCR engine** — `rapidocr-onnxruntime` timed out on the network (15 MB
  wheel, repeated resets). Fell back to a calibrated, clearly-labeled OCR-noise model; the
  extractor/degradation/scoring are real and a live engine is a one-line swap.
- **Multilingual date fix via labels alone did nothing** at first (date stayed 0.58) — the
  actual blocker was the missing dot-date format, not the label. Fixing the wrong layer first
  is itself the lesson.

## Sample outputs saved

- `results/phase5_robustness.csv` — combined long-format (all 4 tracks)
- `results/phase5_ocr.csv`, `phase5_active_learning.csv`, `phase5_multilingual.csv`, `phase5_rag.csv`
- Charts: `phase5_ocr_field_accuracy.png`, `phase5_active_learning.png`
- Samples: `phase5_ocr_samples.json`, `phase5_al_queried_hard.json`,
  `phase5_multilingual_samples.json`, `phase5_rag_answers.json`
- 5 degraded receipt images: `results/samples/day7_receipt_images/`

---

## Next session

**Day 8 — Phase 6: Frontier comparison + ablation.** Fresh held-out set (50 receipts + 100
transactions): each component vs Claude Opus 4.6 / GPT-5.4 zero-shot. Where small+specialized
wins (categorization rare classes, latency, on-device cost) vs where the LLM wins (zero-shot
extraction on novel/messy receipts). Categorizer ablation (keyword → +TF-IDF → +LightGBM →
+tuned → +threshold/AL). `results/frontier_comparison.csv` + `results/ablation.csv`. **[POST · PHASE-WRAP]**

## Code changes

Branch `sprint/day07-2026-07-01`; commits prefixed `sprint:` referencing each file. The
multilingual gazetteer + dot-date fix were shipped into `src/extraction/extractor.py` and
verified regression-neutral on English (old == new on 100 SROIE receipts) with Day-5
integration parity green. No metric was reported that a script did not compute; the OCR track
is explicitly labeled as a calibrated simulation because a live engine could not be fetched.
