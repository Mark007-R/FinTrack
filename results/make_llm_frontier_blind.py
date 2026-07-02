"""Emit BLIND inputs for the Day-8 frontier LLM zero-shot head-to-head.

The LLM (frontier model) is shown ONLY these files — never the ground truth —
so the head-to-head is honest. Predictions go into a sibling *_pred.json; the
deterministic scorer (score_llm_frontier.py) then scores them against the hidden
GT in the eval files.

  * extraction: 25-receipt indicative subset (matches the Day-2 N=20 protocol)
  * categorization: ALL 100 transactions (regime crossover needs both regimes)
"""
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "data", "eval")
SAMPLES = os.path.join(ROOT, "results", "samples")
os.makedirs(SAMPLES, exist_ok=True)

# ---- extraction blind (first 25 receipts) ----
rec = [json.loads(l) for l in open(os.path.join(EVAL, "frontier_receipts.jsonl"), encoding="utf-8")]
ext_blind = [{"id": r["id"], "text": r["text"]} for r in rec[:25]]
json.dump({"task": "Extract amount (grand total, GST-inclusive), date, and merchant name from each receipt.",
           "items": ext_blind},
          open(os.path.join(SAMPLES, "_llm_frontier_extract_blind.json"), "w"), indent=2)
print("wrote extract blind:", len(ext_blind))

# ---- categorization blind (all 100, regime hidden) ----
txn = list(csv.DictReader(open(os.path.join(EVAL, "frontier_transactions.csv"), encoding="utf-8")))
cats = ["groceries", "dining", "transport", "utilities", "rent", "entertainment",
        "health", "shopping", "income", "other"]
cat_blind = [{"id": int(r["id"]), "description": r["description"]} for r in txn]
json.dump({"task": "Classify each bank/card transaction description into exactly one category.",
           "categories": cats, "items": cat_blind},
          open(os.path.join(SAMPLES, "_llm_frontier_cat_blind.json"), "w"), indent=2)
print("wrote cat blind:", len(cat_blind))
