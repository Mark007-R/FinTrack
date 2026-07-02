"""Day-8 Phase-6 — Frontier comparison + categorizer ablation (specialized side).

Runs on the FRESH held-out sets built by data/eval/build_day8_frontier.py:
  * frontier_receipts.jsonl    (50 SROIE receipts, test offset>=100, unseen)
  * frontier_transactions.csv  (100 synthetic txns, seed 20260702, deduped)

Two specialized components are scored end-to-end; the LLM zero-shot rows are
appended later by score_llm_frontier.py (blind predictions -> hidden GT) so this
file stays deterministic and API-free.

Outputs:
  results/frontier_comparison.csv   (extraction + categorization, specialized rows)
  results/ablation.csv              (keyword -> +tfidf -> +char -> +tuned -> +override)
  results/frontier_extraction.png, results/ablation.png
  results/samples/frontier_extraction_samples.json
  results/samples/frontier_categorize_samples.json
"""
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
SAMPLES = os.path.join(RESULTS, "samples")
EVAL = os.path.join(ROOT, "data", "eval")
sys.path.insert(0, RESULTS)
sys.path.insert(0, ROOT)
os.makedirs(SAMPLES, exist_ok=True)

from run_day2_extraction import parse_amount, norm_date, merchant_match  # noqa: E402,F401
from day8_dateutil import norm_date8  # noqa: E402
from src.extraction.extractor import ReceiptExtractor, _regex_fallback  # noqa: E402

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.pipeline import FeatureUnion, Pipeline  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402
from sklearn.metrics import f1_score, accuracy_score  # noqa: E402


# ===================================================================== EXTRACT
def load_receipts():
    rows = []
    for line in open(os.path.join(EVAL, "frontier_receipts.jsonl"), encoding="utf-8"):
        rows.append(json.loads(line))
    return rows


def score_extractor(name, fn, receipts):
    amt_ok = date_ok = merch_ok = both_ok = 0
    t0 = time.time()
    samples = []
    for r in receipts:
        pred = fn(r["text"])
        gt_amt = parse_amount(r["gt_total"])
        gt_date = norm_date8(r["gt_date"])
        a_ok = gt_amt is not None and pred.get("amount") is not None and abs(pred["amount"] - gt_amt) < 0.01
        d_ok = gt_date is not None and norm_date8(pred.get("date")) == gt_date
        m_ok = merchant_match(pred.get("merchant"), r["gt_merchant"])
        amt_ok += a_ok; date_ok += d_ok; merch_ok += m_ok; both_ok += (a_ok and d_ok)
        if len(samples) < 8:
            samples.append({"id": r["id"], "gt": {"amount": gt_amt, "date": gt_date, "merchant": r["gt_merchant"]},
                            "pred": {k: pred.get(k) for k in ("amount", "date", "merchant")},
                            "ok": {"amount": bool(a_ok), "date": bool(d_ok), "merchant": bool(m_ok)}})
    n = len(receipts)
    sec = time.time() - t0
    return {"component": "extraction", "method": name, "n": n,
            "amount_acc": round(amt_ok / n, 4), "date_acc": round(date_ok / n, 4),
            "merchant_acc": round(merch_ok / n, 4), "exact_amt_date": round(both_ok / n, 4),
            "sec_per_doc": round(sec / n, 5), "usd_per_1k_docs": 0.0}, samples


def run_extraction():
    receipts = load_receipts()
    extractor = ReceiptExtractor()
    rows, all_samples = [], {}
    # 1) old naive regex (Day-1 baseline)
    row, s = score_extractor("regex (Day-1 baseline)", _regex_fallback, receipts)
    rows.append(row); all_samples["regex"] = s
    # 2) champion rules_smart
    row, s = score_extractor("rules_smart (champion, local CPU)", extractor.extract, receipts)
    rows.append(row); all_samples["rules_smart"] = s
    json.dump(all_samples, open(os.path.join(SAMPLES, "frontier_extraction_samples.json"), "w"), indent=2)
    for r in rows:
        print(f"  [extract] {r['method']:38s} amt={r['amount_acc']:.2f} date={r['date_acc']:.2f} "
              f"merch={r['merchant_acc']:.2f} {r['sec_per_doc']*1000:.1f} ms/doc")
    return rows


# ================================================================== CATEGORIZE
def load_txn(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return [r["description"] for r in rows], [r["category"] for r in rows]


def load_txn_regime(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return ([r["description"] for r in rows], [r["category"] for r in rows],
            [r.get("regime", "in_dist") for r in rows])


def tfidf_union(char=True):
    feats = [("w", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True))]
    if char:
        feats.append(("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)))
    return FeatureUnion(feats)


# Day-1 keyword baseline (import the shipped fallback so it is the SAME rules)
from src.categorization.classifier import _keyword_predict, _disambig  # noqa: E402

# Optuna best params found on Day 6 (results/phase4_categorizer.json)
TUNED = dict(word_ngram_hi=3, word_min_df=1, char_lo=2, char_hi=6, char_min_df=3,
             sublinear=False, C=4.4321841874023535)


def tuned_union():
    return FeatureUnion([
        ("w", TfidfVectorizer(analyzer="word", ngram_range=(1, TUNED["word_ngram_hi"]),
                              min_df=TUNED["word_min_df"], sublinear_tf=TUNED["sublinear"])),
        ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(TUNED["char_lo"], TUNED["char_hi"]),
                              min_df=TUNED["char_min_df"], sublinear_tf=TUNED["sublinear"])),
    ])


def macro(labels, y, yp):
    return round(float(f1_score(y, yp, labels=labels, average="macro", zero_division=0)), 4)


def _regime_f1(labels, yte, yp, regimes, which):
    # macro-F1 restricted to labels PRESENT in this regime's ground truth, so a
    # 40-row subset isn't penalised for the 5 classes it happens not to contain.
    idx = [i for i, r in enumerate(regimes) if r == which]
    if not idx:
        return None
    yt = [yte[i] for i in idx]; yq = [yp[i] for i in idx]
    present = sorted(set(yt))
    return round(float(f1_score(yt, yq, labels=present, average="macro", zero_division=0)), 4), \
        round(float(accuracy_score(yt, yq)), 4), len(idx)


def run_categorize_and_ablation():
    Xtr, ytr = load_txn(os.path.join(EVAL, "transactions.csv"))                 # 600 training rows
    Xte, yte, regimes = load_txn_regime(os.path.join(EVAL, "frontier_transactions.csv"))  # 100 held-out
    labels = sorted(set(ytr) | set(yte))

    # ---------- ablation ladder (each trained on the 600, scored on fresh 100) ----------
    # Each stage records overall macro-F1 + the in_dist/novel split so the
    # ablation shows WHERE each rung helps (novel merchants are the hard regime).
    ladder = []

    def add(name, yp):
        ladder.append({"stage": name, "yp": yp,
                       "macro_f1": macro(labels, yte, yp),
                       "accuracy": round(float(accuracy_score(yte, yp)), 4),
                       "f1_in_dist": _regime_f1(labels, yte, yp, regimes, "in_dist")[0],
                       "f1_novel": _regime_f1(labels, yte, yp, regimes, "novel")[0]})

    # S1 keyword
    add("S1  keyword baseline", [_keyword_predict(x) for x in Xte])
    # S2 + TF-IDF(word) + LinearSVC
    p2 = Pipeline([("f", tfidf_union(char=False)), ("c", LinearSVC(class_weight="balanced", C=1.0))]).fit(Xtr, ytr)
    add("S2  + TF-IDF(word) + LinearSVC", list(p2.predict(Xte)))
    # S3 + char n-grams (word+char union) — Day-3 champion architecture
    p3 = Pipeline([("f", tfidf_union(char=True)), ("c", LinearSVC(class_weight="balanced", C=1.0))]).fit(Xtr, ytr)
    add("S3  + char n-grams (word+char)", list(p3.predict(Xte)))
    # S4 + Optuna-tuned hyperparameters (Day-6)
    p4 = Pipeline([("f", tuned_union()),
                   ("c", LinearSVC(class_weight="balanced", C=TUNED["C"]))]).fit(Xtr, ytr)
    yp4_raw = list(p4.predict(Xte))
    add("S4  + Optuna-tuned params", yp4_raw)
    # S5 + Day-6 disambiguation override (multi-word cross-category fix)
    add("S5  + Day-6 disambig override", [(_disambig(x) or p) for x, p in zip(Xte, yp4_raw)])

    base = ladder[0]["macro_f1"]; prev = base
    abl_rows = []
    for st in ladder:
        abl_rows.append({"stage": st["stage"], "macro_f1": st["macro_f1"], "accuracy": st["accuracy"],
                         "f1_in_dist": st["f1_in_dist"], "f1_novel": st["f1_novel"],
                         "delta_vs_prev": round(st["macro_f1"] - prev, 4),
                         "delta_vs_keyword": round(st["macro_f1"] - base, 4)})
        prev = st["macro_f1"]
    with open(os.path.join(RESULTS, "ablation.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "macro_f1", "accuracy", "f1_in_dist", "f1_novel",
                                          "delta_vs_prev", "delta_vs_keyword"])
        w.writeheader(); w.writerows(abl_rows)
    print("\n  [ablation] fresh held-out 100 (overall | in_dist | novel):")
    for r in abl_rows:
        print(f"    {r['stage']:34s} F1={r['macro_f1']:.3f}  in={r['f1_in_dist']:.3f}  "
              f"novel={r['f1_novel']:.3f}  dPrev={r['delta_vs_prev']:+.3f}")

    # ---------- frontier categorize rows: keyword vs shipped champion ----------
    # shipped champion = the tuned+disambig pipeline (S5) — the deployed classifier
    t0 = time.time(); kw_pred = [_keyword_predict(x) for x in Xte]; kw_sec = (time.time() - t0) / len(Xte)
    t0 = time.time(); champ_pred = [(_disambig(x) or p) for x, p in zip(Xte, p4.predict(Xte))]; ch_sec = (time.time() - t0) / len(Xte)
    kw_in = _regime_f1(labels, yte, kw_pred, regimes, "in_dist"); kw_nv = _regime_f1(labels, yte, kw_pred, regimes, "novel")
    ch_in = _regime_f1(labels, yte, champ_pred, regimes, "in_dist"); ch_nv = _regime_f1(labels, yte, champ_pred, regimes, "novel")
    cat_rows = [
        {"component": "categorization", "method": "keyword (Day-1 baseline)", "n": len(Xte),
         "macro_f1": ladder[0]["macro_f1"], "accuracy": round(float(accuracy_score(yte, kw_pred)), 4),
         "f1_in_dist": kw_in[0], "f1_novel": kw_nv[0],
         "sec_per_doc": round(kw_sec, 6), "usd_per_1k_docs": 0.0},
        {"component": "categorization", "method": "tfidf_linsvc (champion, local CPU)", "n": len(Xte),
         "macro_f1": ladder[-1]["macro_f1"], "accuracy": round(float(accuracy_score(yte, champ_pred)), 4),
         "f1_in_dist": ch_in[0], "f1_novel": ch_nv[0],
         "sec_per_doc": round(ch_sec, 6), "usd_per_1k_docs": 0.0},
    ]
    print(f"\n  [categorize] champion  overall={cat_rows[1]['macro_f1']:.3f}  "
          f"in_dist={ch_in[0]:.3f} (n={ch_in[2]})  novel={ch_nv[0]:.3f} (n={ch_nv[2]})")

    # per-class F1 for the champion (rare-class story)
    from sklearn.metrics import f1_score as f1s
    per_class = f1s(yte, champ_pred, labels=labels, average=None, zero_division=0)
    per_class_f1 = {c: round(float(v), 4) for c, v in zip(labels, per_class)}

    # samples: 12 champion predictions incl. any misses
    samp = []
    for x, gt, pr in zip(Xte, yte, champ_pred):
        if len(samp) < 12 or gt != pr:
            samp.append({"description": x, "gt": gt, "pred": pr, "ok": gt == pr})
        if len(samp) >= 20:
            break
    json.dump({"per_class_f1_champion": per_class_f1, "samples": samp},
              open(os.path.join(SAMPLES, "frontier_categorize_samples.json"), "w"), indent=2)
    print("\n  [categorize] champion per-class F1:", per_class_f1)
    return cat_rows, abl_rows, per_class_f1


# ======================================================================== MAIN
def write_frontier_csv(extract_rows, cat_rows):
    cols = ["component", "method", "n", "amount_acc", "date_acc", "merchant_acc",
            "exact_amt_date", "macro_f1", "accuracy", "f1_in_dist", "f1_novel",
            "sec_per_doc", "usd_per_1k_docs"]
    path = os.path.join(RESULTS, "frontier_comparison.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in extract_rows + cat_rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"\n  wrote {path} (specialized rows; LLM rows appended by score_llm_frontier.py)")


if __name__ == "__main__":
    print("== Day-8 Phase-6 frontier (specialized) ==")
    ex = run_extraction()
    cat, abl, pcf = run_categorize_and_ablation()
    write_frontier_csv(ex, cat)
    # summary json
    json.dump({"generated": "2026-07-02", "day": 8, "phase": "Phase 6 — frontier + ablation",
               "heldout": {"receipts": 50, "transactions": 100},
               "ablation": abl, "champion_per_class_f1": pcf},
              open(os.path.join(RESULTS, "phase6_frontier.json"), "w"), indent=2)
    print("done.")
