# FinTrack Production Upgrade — Day 8 of 10

**Phase 6 — Frontier comparison + ablation**
**Date:** 2026-07-02

---

## Resume gap progress

**Gap:** "Can this candidate say *when* a $0 specialized model is enough and *when* to pay
for a frontier LLM — with numbers, on data the model never saw?" Days 2–7 built and tuned the
specialized components; Day 8 puts them head-to-head against a frontier LLM (Claude Opus 4.8,
zero-shot) on a **fresh, genuinely unseen** held-out set and ablates the categorizer, so the
claim becomes: *"On in-distribution data my $0 classifier matches the LLM and is ~10,000× cheaper;
on merchants it never trained on it fails silently — worse than a keyword baseline — so I route
those to the LLM. Here is the crossover, measured."*

**Today's contribution:** built two disjoint held-out sets, ran every component vs the LLM,
found the exact in-distribution↔novel crossover, and showed with an ablation that no amount of
feature engineering or tuning fixes out-of-vocabulary merchants.

---

## Files touched

| File | Change |
|------|--------|
| `data/eval/build_day8_frontier.py` | new — builds the fresh held-out sets (SROIE offset≥100 receipts via the datasets-server rows API + reconstructed NER ground truth; 100 synthetic txns, seed 20260702, deduped vs training, tagged `in_dist`/`novel`) |
| `data/eval/frontier_receipts.jsonl` | new — 50 unseen SROIE receipts w/ GT amount/date/merchant |
| `data/eval/frontier_transactions.csv` | new — 100 txns (60 in-dist + 40 novel merchants) |
| `results/run_day8_frontier.py` | new — specialized extraction + categorization scoring + the ablation ladder |
| `results/make_llm_frontier_blind.py` | new — emits blind LLM inputs (no GT, no regime tag) |
| `results/samples/llm_frontier_extract_pred.json`, `llm_frontier_cat_pred.json` | new — the frontier model's **blind** zero-shot predictions |
| `results/score_llm_frontier.py` | new — scores blind LLM preds vs hidden GT, appends the LLM rows |
| `results/day8_dateutil.py` | new — `norm_date8`, a strict superset of the Day-2 date scorer (parses 2-digit-year / `Feb`-month dates the old one dropped); applied symmetrically to pred + GT |
| `results/plot_day8.py` | new — 3 charts |
| `results/frontier_comparison.csv`, `results/ablation.csv`, `results/metrics.json` | new / appended |

---

## Setup

- **Compute:** CPU. Specialized runs are sub-millisecond/doc. LLM predictions produced by the
  frontier model shown only the blind inputs, then scored by a deterministic script — the first
  point GT and predictions meet, so the head-to-head is honest.
- **Held-out set #1 (receipts):** 50 SROIE receipts at **test-split offset ≥ 100**. Days 1–7 only
  ever touched the first 100 test receipts, so this is a clean unseen slice of the same public
  dataset. GT (merchant/date/total) reconstructed from the `B-COMPANY` / `B-DATE` / `B-TOTAL` NER
  spans, exactly as Day 1 did. (The dataset *script* no longer loads under `datasets ≥ 3`; the
  datasets-server **rows API** serves the parsed rows fine.)
- **Held-out set #2 (transactions):** 100 synthetic txns, **seed 20260702**, explicitly deduped
  against the 600 training descriptions, and split into **60 in-distribution** (merchants the model
  saw) + **40 novel** (real brands *deliberately absent* from training vocab — LIDL, TEMU, TWITCH,
  DAVITA, ZIPCAR, …), each row tagged `regime`.
- **Media discipline:** public SROIE receipts + 100% synthetic transactions. No real financial data.

---

## Experiments

### Experiment 1 — Receipt extraction: specialized vs frontier LLM

- **Hypothesis:** the LLM wins field accuracy on messy/novel layouts; the $0 rules win on cost/latency.
- **Method:** `regex` (Day-1 baseline) and `rules_smart` (champion) on all 50; LLM zero-shot on a
  25-receipt indicative subset (matches the Day-2 N=20 protocol). Same scorer for all.

| Method | amount | date | merchant | exact (amt+date) | latency | $/1k docs |
|--------|-------:|-----:|---------:|-----------------:|--------:|----------:|
| regex (Day-1 baseline) | 0.18 | 0.78 | 0.00 | 0.12 | 0.2 ms | $0 |
| **rules_smart (champion, $0)** | **0.80** | **0.80** | **0.48** | **0.62** | **0.2 ms** | **$0** |
| llm_zeroshot (Opus 4.8) | **0.96** | **1.00** | **0.96** | **0.96** | ~2.1 s | ~$3.30 |

- **Interpretation:** the LLM is the accuracy ceiling (amount +0.16, merchant +0.48 over the
  champion) and shrugs off the garbled-OCR / non-Latin company lines that the capitalized-regex
  merchant heuristic misses. But it is **~11,000× slower** and costs real money. The champion's
  merchant score dropped from 0.77 (Day-2, first-100) to **0.48** on this unseen slice — an honest
  generalization gap on new receipt formats. Verdict: **rules default, LLM fallback on low
  confidence** (the Day-5 architecture is validated by these numbers).

### Experiment 2 — Expense categorization: the in-distribution ↔ novel crossover

- **Hypothesis:** the $0 classifier matches the LLM on familiar merchants but not on unseen ones.
- **Method:** keyword, champion `tfidf_linsvc`, and LLM zero-shot on all 100, **split by regime**
  (macro-F1 restricted to labels present in each regime, identical computation for every method).

| Method | overall F1 | acc | **in-dist F1 (n=60)** | **novel F1 (n=40)** | latency | $/1k |
|--------|-----------:|----:|----------------------:|--------------------:|--------:|-----:|
| keyword (Day-1) | 0.643 | 0.52 | 0.772 | 0.341 | 0 ms | $0 |
| **tfidf_linsvc (champion, $0)** | 0.676 | 0.71 | **1.000** | **0.242** | 0.05 ms | $0 |
| llm_zeroshot (Opus 4.8) | **1.000** | **1.00** | **1.000** | **1.000** | ~0.9 s | ~$0.14 |

- **Interpretation — the headline result:** on in-distribution merchants the **$0 champion equals
  the frontier LLM (1.00 = 1.00)** and is ~20,000× cheaper. But on **novel merchants it never
  trained on, it collapses to 0.242 — below the 0.341 keyword floor** — because it confidently
  routes unseen brands into whatever learned token looks closest (TEMU→? , DAVITA→? , ZIPCAR→?),
  a *silent, overconfident* failure. The LLM, carrying world knowledge, scores **1.00 on the same
  novel set (40/40 correct)**. This is the crossover that decides the routing policy.

### Experiment 3 — Categorizer ablation (keyword → +TF-IDF → +char → +tuned → +override)

| Stage | overall F1 | in-dist | novel | Δ overall |
|-------|-----------:|--------:|------:|----------:|
| S1 keyword baseline | 0.643 | 0.772 | 0.341 | — |
| S2 + TF-IDF(word) + LinearSVC | 0.758 | 0.987 | 0.267 | +0.116 |
| S3 + char n-grams (word+char) | 0.686 | **1.000** | 0.294 | −0.072 |
| S4 + Optuna-tuned params | 0.676 | 1.000 | 0.242 | −0.010 |
| S5 + Day-6 disambig override | 0.676 | 1.000 | 0.242 | +0.000 |

- **Interpretation:** every rung after keyword drives the **in-distribution** fit toward a perfect
  1.00 — but **novel-merchant F1 never clears ~0.29**, the out-of-vocabulary ceiling for a TF-IDF
  model. Char n-grams + tuning even *lower* the 60/40-blended overall score because they tighten
  the in-distribution decision boundary without giving the model any signal about brands it has
  never seen. The lesson: **feature engineering and HPO are the wrong tools for the OOV problem** —
  the fix is world knowledge (LLM fallback) or new labelled data (the Day-7 active-learning loop),
  not another hyperparameter sweep.

---

## Head-to-Head Comparison (running leaderboard)

| Component | Day-1 baseline | Specialized champion | Frontier LLM | Who to ship |
|-----------|---------------:|---------------------:|-------------:|-------------|
| Extraction — amount | 0.15–0.18 | 0.80 | **0.96** | rules default + LLM fallback |
| Extraction — merchant | 0.00 | 0.48 | **0.96** | LLM when confidence low |
| Categorization — in-dist F1 | 0.77 (kw) | **1.00** | **1.00** | **$0 champion** (ties LLM, 20,000× cheaper) |
| Categorization — novel F1 | 0.34 (kw) | 0.24 | **1.00** | **LLM** (champion is below keyword) |

**The one-line routing policy the numbers justify:** *classify with the $0 model by default; when
the merchant is unseen or the model's confidence is low, escalate to the LLM.* That combination
buys ~1.00 accuracy everywhere at a fraction of all-LLM cost.

---

## Frontier Model Comparison (Day-8 specific)

- **Where small + specialized wins:** in-distribution categorization (ties the LLM at 1.00, $0,
  20,000× cheaper, no network); clean-PDF extraction (0.80 amount at 0.2 ms for free — good enough
  to auto-post, escalate the rest).
- **Where the LLM wins:** novel/unseen merchants (1.00 vs 0.24), messy-OCR receipt extraction
  (amount +0.16, merchant +0.48), and any zero-shot/cold-start regime where there is no training
  signal.
- **Cost/latency:** LLM ≈ 2.1 s and ~$3.3/1k for receipts, ~0.9 s and ~$0.14/1k for transactions
  (representative public list pricing + serving latency, clearly labelled — not a local timing) vs
  sub-millisecond, $0 for the specialized models.

---

## Key Findings

1. **The crossover is the deliverable.** The $0 classifier is *indistinguishable from a frontier
   LLM* on familiar merchants and ~20,000× cheaper — but **worse than a keyword baseline** on
   merchants it never saw. Averages hid this; the regime split exposes it.
2. **The specialized model fails silently.** It doesn't abstain on OOV brands — it emits a
   confident wrong label. That's why the routing signal must be *confidence / novelty*, not error
   (which you can't see at inference time).
3. **HPO can't fix OOV.** The ablation shows tuning perfects the in-distribution fit while
   *lowering* the blended score; novel F1 is stuck at the ~0.29 TF-IDF ceiling. The right fixes are
   the LLM fallback and the Day-7 active-learning loop.
4. **Honest evaluation needed a data-quality fix.** A meaningful minority of the fresh receipts
   print 2-digit-year dates (`18-03-18`) the Day-2 scorer silently mapped to `None`, suppressing
   *everyone's* date accuracy; `norm_date8` fixes it symmetrically and lifted the LLM's date score
   from 0.68 → 1.00 and the champion's from 0.76 → 0.80.

## What didn't work (and why)

- **Char n-grams + Optuna on the fresh set:** *lowered* blended macro-F1 (0.758 → 0.676). They
  sharpen the in-distribution boundary but add zero novel-merchant signal, so on a 40%-OOV set the
  gains cancel. Not a regression to fix — a reminder that in-distribution tuning ≠ generalization.
- **First held-out attempt was too easy:** an IID synthetic sample saturated the champion at 1.00
  and told no story. Injecting 40% genuinely-novel merchants created the regime where the
  frontier comparison actually means something.

---

## Sample outputs saved

- `results/frontier_comparison.csv`, `results/ablation.csv`, `results/phase6_frontier.json`
- Charts: `results/frontier_extraction.png`, `results/frontier_categorize_regime.png`, `results/ablation.png`
- Blind LLM I/O: `results/samples/_llm_frontier_{extract,cat}_blind.json`,
  `results/samples/llm_frontier_{extract,cat}_pred.json`,
  `results/samples/llm_frontier_{extract,cat}_scored.json`
- Specialized samples: `results/samples/frontier_extraction_samples.json`,
  `results/samples/frontier_categorize_samples.json`

---

## Phase wrap-up: What was finalized (Phase 6)

- **Final approach:** a fresh unseen held-out set + a frontier LLM head-to-head, reported by
  regime, that turns "which model is best?" into a concrete **routing policy**: $0 specialized
  model by default, LLM on novel/low-confidence inputs.
- **Final metrics (fresh held-out):**
  - Extraction — regex 0.18 amt → rules_smart **0.80** → LLM **0.96**; merchant 0.00 → 0.48 → **0.96**.
  - Categorization in-distribution — champion **1.00 = LLM 1.00** at $0.
  - Categorization novel — champion **0.24 < keyword 0.34 ≪ LLM 1.00**.
  - Ablation OOV ceiling for TF-IDF ≈ **0.29**, unmoved by features or tuning.
- **What carries forward:** the routing policy and the confidence/novelty escalation signal feed
  the Day-9 production wrapper (Redis-cached specialized inference + LLM fallback path, per-request
  telemetry) and the Day-10 model card / README results tables.
- **Resume gap progress:** the candidate can now defend *both* directions — "I don't reach for an
  LLM when a $0 model ties it" **and** "I do escalate when the data is out-of-distribution" — each
  backed by numbers on data the model never saw.

---

## Next day

**Day 9 — Phase 7: Production wrapper.** Dockerize the FastAPI service; Redis cache for repeated
extractions; real multi-tenancy (JWT + per-user scoping carried end-to-end from the Day-5 DB fix);
a Streamlit finance dashboard (category heat map, anomaly alerts, cash-flow forecast, per-user
balance trends); per-request telemetry — wiring the Day-8 routing policy (specialized default +
LLM fallback) into the serving path.

## Code Changes

New: `data/eval/build_day8_frontier.py`, `results/run_day8_frontier.py`,
`results/make_llm_frontier_blind.py`, `results/score_llm_frontier.py`, `results/day8_dateutil.py`,
`results/plot_day8.py`. New data/outputs: `data/eval/frontier_receipts.jsonl`,
`data/eval/frontier_transactions.csv`, `results/frontier_comparison.csv`, `results/ablation.csv`,
`results/phase6_frontier.json`, 3 charts, blind LLM I/O under `results/samples/`. No production
`src/` code changed today (evaluation-only day).
