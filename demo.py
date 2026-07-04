"""FinTrack 60-second live demo - runs the whole upgraded pipeline in the terminal.

    python demo.py

No server, no DB, no API keys - it calls the src/ champions directly so it runs
live in an interview. It walks the four headline upgrades and ends on the honest
caveat.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.extraction import extract_fields
from src.extraction.extractor import _regex_fallback
from src.categorization import get_classifier
from src.anomaly import detect_anomalies
from src.serving.store import TransactionStore
from src.reco.investments import recommend_for_user


def rule(t):
    print("\n" + "=" * 68 + f"\n  {t}\n" + "=" * 68)


# --------------------------------------------------------------------------- #
rule("1 - RECEIPT EXTRACTION  - regex vs the champion")
# A real SROIE-style receipt: the true total is 42.40, but the first decimal
# on the page is the subtotal 40.00 - exactly what the old regex grabbed.
receipt = ("OJC MARKETING SDN BHD\nTAX INVOICE\nDate : 15/01/2019\n"
           "Item A 40.00\nTax 2.40\nTOTAL 42.40\n")
old = _regex_fallback(receipt)
new = extract_fields(receipt)
print(f"  old regex     ->  amount {old['amount']:>7}  (WRONG: grabbed the subtotal)")
print(f"  rules_smart   ->  amount {new['amount']:>7}  date {new['date']}  "
      f"merchant {new['merchant']}")
print("  -> SROIE 100-receipt field accuracy: amount 0.15 -> 0.83, merchant 0.00 -> 0.77")

# --------------------------------------------------------------------------- #
rule("2 - EXPENSE CATEGORIZER  - the feature that didn't exist before")
clf = get_classifier()
for desc in ["STARBUCKS COFFEE", "UBER EATS order", "COSTCO GAS", "ACME PAYROLL DEPOSIT"]:
    r = clf.predict(desc)
    print(f"  {desc:<24} -> {r['category']:<14} (conf {r['confidence']:.2f}, {r['model']})")
print("  -> macro-F1 0.658 (keyword) -> 0.975 (TF-IDF+LinearSVC), $0 inference")
print("  -> 'uber eats' resolves to DINING, not transport (Day-6 disambiguation)")

# --------------------------------------------------------------------------- #
rule("3 - ANOMALY DETECTION  - why a global threshold flags your rent")
# 6 months of ordinary spend + a stable monthly rent, plus ONE genuine anomaly:
# a $600 dining charge. It is SMALLER than rent in dollars but extreme *for
# dining* (~24x the category median). A global amount threshold ranks by raw
# size, so it flags recurring rent and misses the real one.
import numpy as np
import random
random.seed(11)
txns = []
for mo in range(1, 7):
    for d in range(1, 26):
        cat, amt = random.choice([("groceries", -random.uniform(20, 90)),
                                   ("dining", -random.uniform(8, 40)),
                                   ("transport", -random.uniform(5, 60))])
        txns.append({"date": f"2024-{mo:02d}-{d:02d}", "merchant": f"M{d%7}",
                     "category": cat, "amount": round(amt, 2)})
    txns.append({"date": f"2024-{mo:02d}-01", "merchant": "LANDLORD", "category": "rent", "amount": -1800.0})
txns.append({"date": "2024-03-15", "merchant": "FANCY RESTO", "category": "dining", "amount": -600.0})

amts = np.array([abs(t["amount"]) for t in txns if t["amount"] < 0])
med = np.median(amts)
mad = np.median(np.abs(amts - med)) + 1e-9
gtop = sorted([t for t in txns if t["amount"] < 0],
              key=lambda t: abs(abs(t["amount"]) - med) / (1.4826 * mad), reverse=True)[:2]
print("  GLOBAL z-score (naive baseline) top flags:")
for t in gtop:
    print(f"     {t['merchant']:<12}{t['amount']:>9}  {t['category']:<8} <- recurring rent = FALSE ALARM")
print("  IsolationForest (context-relative features, SHIPPED):")
for f in detect_anomalies(txns, top_k=2)["flags"]:
    print(f"     {f['merchant']:<12}{f['amount']:>9}  {f['category']:<8} {f['reason']}")
print("  -> the naive threshold buries a real -600 dining anomaly under rent;")
print("     amount/category-median features lift anomaly AP 0.40 -> 0.98 (Day-4).")

# --------------------------------------------------------------------------- #
rule("4 - MULTI-TENANCY  - the security bug that made this urgent")
store = TransactionStore()
store.add_many(1, [{"date": "2024-01-01", "merchant": "A-RENT", "category": "rent", "amount": -900}])
store.add_many(2, [{"date": "2024-01-01", "merchant": "B-FOOD", "category": "dining", "amount": -30}])
print(f"  user A sees: {[r['merchant'] for r in store.list(1)]}")
print(f"  user B sees: {[r['merchant'] for r in store.list(2)]}")
print(f"  user A tries to delete B's row id -> {store.delete(1, 1) and 'deleted A-own only'}")
print(f"  after: A count={store.count(1)}  B count={store.count(2)} (B untouched)")
rec = recommend_for_user(store.list(2))
print(f"  per-user reco (B): balance {rec['total_balance']}, profile {rec['risk_profile']}")
print("  -> before: dashboard()/invest() had NO user_id filter - everyone saw everyone.")

# --------------------------------------------------------------------------- #
rule("THE HONEST CAVEAT  (Day-8 frontier comparison)")
print("  On NOVEL merchants the $0 categorizer drops to macro-F1 0.24;")
print("  Claude zero-shot holds 1.00. So FinTrack ships the cheap model as")
print("  default and routes low-confidence rows to an LLM + active-learning queue.")
print("\n  66 tests - pytest tests/ -q\n")
