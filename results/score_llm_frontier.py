"""Score the Day-8 frontier LLM zero-shot predictions against HIDDEN ground truth
and append the two LLM rows to results/frontier_comparison.csv.

The predictions in samples/llm_frontier_{extract,cat}_pred.json were produced by
the frontier model shown ONLY the blind inputs (no GT, no regime). This scorer is
the first place GT and predictions meet — keeping the head-to-head honest.

Reported cost/latency for the LLM are representative figures (public API list
price + typical serving latency), clearly labelled — NOT a local CPU timing.
"""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
SAMPLES = os.path.join(RESULTS, "samples")
EVAL = os.path.join(ROOT, "data", "eval")
sys.path.insert(0, RESULTS)

from run_day2_extraction import parse_amount, norm_date, merchant_match  # noqa: E402,F401
from day8_dateutil import norm_date8  # noqa: E402
from sklearn.metrics import f1_score, accuracy_score  # noqa: E402

# Representative frontier pricing (public list, Opus-class): ~$15 / 1M output +
# ~$3 / 1M input tokens. A receipt ~ 900 in / 40 out tokens => ~$0.0033/doc;
# a txn ~ 40 in / 6 out => ~$0.00014/doc. Reported per 1k docs, clearly labelled.
LLM_USD_PER_1K_RECEIPT = 3.3
LLM_USD_PER_1K_TXN = 0.14
LLM_SEC_PER_RECEIPT = 2.1
LLM_SEC_PER_TXN = 0.9


def score_extraction():
    gt = {r["id"]: r
          for r in (json.loads(l) for l in open(os.path.join(EVAL, "frontier_receipts.jsonl"), encoding="utf-8"))}
    preds = json.load(open(os.path.join(SAMPLES, "llm_frontier_extract_pred.json")))["predictions"]
    n = len(preds)
    amt_ok = date_ok = merch_ok = both_ok = 0
    detail = []
    for p in preds:
        r = gt[p["id"]]
        gt_amt = parse_amount(r["gt_total"]); gt_date = norm_date8(r["gt_date"])
        a_ok = gt_amt is not None and abs(p["amount"] - gt_amt) < 0.01
        d_ok = gt_date is not None and norm_date8(p["date"]) == gt_date
        m_ok = merchant_match(p["merchant"], r["gt_merchant"])
        amt_ok += a_ok; date_ok += d_ok; merch_ok += m_ok; both_ok += (a_ok and d_ok)
        detail.append({"id": p["id"], "gt_amount": gt_amt, "pred_amount": p["amount"], "amount_ok": bool(a_ok),
                       "gt_date": gt_date, "pred_date": norm_date8(p["date"]), "date_ok": bool(d_ok),
                       "gt_merchant": r["gt_merchant"], "pred_merchant": p["merchant"], "merchant_ok": bool(m_ok)})
    row = {"component": "extraction", "method": "llm_zeroshot (Claude Opus 4.8, OCR text)", "n": n,
           "amount_acc": round(amt_ok / n, 4), "date_acc": round(date_ok / n, 4),
           "merchant_acc": round(merch_ok / n, 4), "exact_amt_date": round(both_ok / n, 4),
           "sec_per_doc": LLM_SEC_PER_RECEIPT, "usd_per_1k_docs": LLM_USD_PER_1K_RECEIPT}
    json.dump(detail, open(os.path.join(SAMPLES, "llm_frontier_extract_scored.json"), "w"), indent=2)
    print(f"[extract LLM n={n}] amount={row['amount_acc']:.2f} date={row['date_acc']:.2f} "
          f"merchant={row['merchant_acc']:.2f} exact={row['exact_amt_date']:.2f}")
    for d in detail:
        if not (d["amount_ok"] and d["date_ok"] and d["merchant_ok"]):
            print("   miss", d["id"], "| amt", d["amount_ok"], d["pred_amount"], "vs", d["gt_amount"],
                  "| date", d["date_ok"], "| merch", d["merchant_ok"])
    return row


def score_categorization():
    rows = list(csv.DictReader(open(os.path.join(EVAL, "frontier_transactions.csv"), encoding="utf-8")))
    gt = {int(r["id"]): r["category"] for r in rows}
    regime = {int(r["id"]): r.get("regime", "in_dist") for r in rows}
    labels = sorted(set(gt.values()))
    preds = json.load(open(os.path.join(SAMPLES, "llm_frontier_cat_pred.json")))["predictions"]
    ids = [p["id"] for p in preds]
    y = [gt[i] for i in ids]; yp = [p["category"] for p in preds]
    macro = round(float(f1_score(y, yp, labels=labels, average="macro", zero_division=0)), 4)
    acc = round(float(accuracy_score(y, yp)), 4)

    def reg(which):
        idx = [k for k, i in enumerate(ids) if regime[i] == which]
        yt = [y[k] for k in idx]; yq = [yp[k] for k in idx]
        present = sorted(set(yt))  # restrict to labels present in this regime (matches specialized scorer)
        return round(float(f1_score(yt, yq, labels=present, average="macro", zero_division=0)), 4), len(idx)

    f1_in, n_in = reg("in_dist"); f1_nv, n_nv = reg("novel")
    row = {"component": "categorization", "method": "llm_zeroshot (Claude Opus 4.8)", "n": len(ids),
           "macro_f1": macro, "accuracy": acc, "f1_in_dist": f1_in, "f1_novel": f1_nv,
           "sec_per_doc": LLM_SEC_PER_TXN, "usd_per_1k_docs": LLM_USD_PER_1K_TXN}
    # scored detail incl. misses
    detail = [{"id": i, "gt": gt[i], "pred": p, "regime": regime[i], "ok": gt[i] == p}
              for i, p in zip(ids, yp)]
    json.dump({"macro_f1": macro, "f1_in_dist": f1_in, "f1_novel": f1_nv, "detail": detail},
              open(os.path.join(SAMPLES, "llm_frontier_cat_scored.json"), "w"), indent=2)
    print(f"[categorize LLM n={len(ids)}] macroF1={macro} acc={acc} | in_dist={f1_in} (n={n_in}) "
          f"novel={f1_nv} (n={n_nv})")
    print("   misses:", [f"{d['id']}:{d['gt']}!={d['pred']}({d['regime']})" for d in detail if not d["ok"]])
    return row


def append_rows(rows):
    path = os.path.join(RESULTS, "frontier_comparison.csv")
    existing = list(csv.DictReader(open(path, encoding="utf-8")))
    cols = existing[0].keys() if existing else rows[0].keys()
    cols = ["component", "method", "n", "amount_acc", "date_acc", "merchant_acc",
            "exact_amt_date", "macro_f1", "accuracy", "f1_in_dist", "f1_novel",
            "sec_per_doc", "usd_per_1k_docs"]
    existing = [e for e in existing if not e["method"].startswith("llm_zeroshot")]
    existing += [{c: r.get(c, "") for c in cols} for r in rows]
    # keep component grouping order: extraction rows then categorization rows
    order = {"extraction": 0, "categorization": 1}
    existing.sort(key=lambda e: (order.get(e["component"], 9), "llm" in e["method"]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for e in existing:
            w.writerow({c: e.get(c, "") for c in cols})
    print(f"\nappended {len(rows)} LLM rows -> {path}")


if __name__ == "__main__":
    print("== Day-8 frontier LLM scoring (blind preds -> hidden GT) ==")
    ex = score_extraction()
    ca = score_categorization()
    append_rows([ex, ca])
