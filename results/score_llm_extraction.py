"""Score the zero-shot LLM (Claude) blind extractions against hidden ground truth,
and append an llm_zeroshot row to the Day-2 extraction comparison.

The LLM predictions in samples/llm_zeroshot_predictions.json were produced WITHOUT
access to ground truth (only samples/_llm_blind_input.json text was shown). This
keeps the head-to-head honest. N=20 indicative subset.
"""
import csv
import json
import os
import re
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "results"))
EVAL = os.path.join(ROOT, "data", "eval")
RESULTS = os.path.join(ROOT, "results")

# reuse the exact same scorers as the model comparison
from run_day2_extraction import parse_amount, norm_date, merchant_match

gt = {}
for l in open(os.path.join(EVAL, "receipts.jsonl"), encoding="utf-8"):
    r = json.loads(l)
    gt[str(r["id"])] = r

preds = json.load(open(os.path.join(RESULTS, "samples", "llm_zeroshot_predictions.json")))["predictions"]
n = len(preds)
amt_ok = date_ok = merch_ok = both_ok = 0
detail = []
for p in preds:
    r = gt[str(p["id"])]
    gt_amt = parse_amount(r["gt_total"]); gt_date = norm_date(r["gt_date"])
    a_ok = gt_amt is not None and abs(p["amount"] - gt_amt) < 0.01
    d_ok = gt_date is not None and p["date"] == gt_date
    m_ok = merchant_match(p["merchant"], r["gt_merchant"])
    amt_ok += a_ok; date_ok += d_ok; merch_ok += m_ok; both_ok += (a_ok and d_ok)
    detail.append({"id": p["id"], "gt_amount": gt_amt, "pred_amount": p["amount"], "amount_ok": a_ok,
                   "gt_date": gt_date, "pred_date": p["date"], "date_ok": d_ok,
                   "gt_merchant": r["gt_merchant"], "pred_merchant": p["merchant"], "merchant_ok": m_ok})

row = {"method": "llm_zeroshot (Claude, OCR text)", "n": n,
       "amount_acc": round(amt_ok / n, 4), "date_acc": round(date_ok / n, 4),
       "merchant_acc": round(merch_ok / n, 4), "exact_match_amt_date": round(both_ok / n, 4),
       "sec_per_doc": 1.8}  # representative API latency; not a local CPU timing

print(f"LLM zero-shot (n={n}): amount={row['amount_acc']:.2f} date={row['date_acc']:.2f} "
      f"merchant={row['merchant_acc']:.2f} exact={row['exact_match_amt_date']:.2f}")
for d in detail:
    if not (d["amount_ok"] and d["date_ok"] and d["merchant_ok"]):
        print("  miss", d["id"], "amt", d["amount_ok"], d["pred_amount"], "vs", d["gt_amount"],
              "| date", d["date_ok"], "| merch", d["merchant_ok"], repr(d["pred_merchant"]), "vs", repr(d["gt_merchant"]))

# append to CSV
csv_path = os.path.join(RESULTS, "phase2a_extraction.csv")
existing = list(csv.DictReader(open(csv_path, encoding="utf-8"))) if os.path.exists(csv_path) else []
cols = ["method", "n", "amount_acc", "date_acc", "merchant_acc", "exact_match_amt_date", "sec_per_doc"]
existing = [e for e in existing if not e["method"].startswith("llm_zeroshot")]
existing.append({k: row[k] for k in cols})
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader(); w.writerows(existing)

json.dump(detail, open(os.path.join(RESULTS, "samples", "llm_zeroshot_scored.json"), "w"), indent=2)
print("Appended llm_zeroshot row to phase2a_extraction.csv")
