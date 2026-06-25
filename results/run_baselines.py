"""Day-1 baselines for FinTrack upgrade.

(1) Receipt extraction: the ACTUAL find_bill_details regex from extract_bill.py
    measured on 100 SROIE receipts (amount / date / merchant field accuracy).
(2) Expense categorization: majority-class + keyword baselines on 600 synthetic
    transactions (macro-F1 + per-class F1).

Outputs:
    results/baseline_metrics.json
    results/samples/extraction_baseline_samples.json
    results/samples/categorization_baseline_samples.json
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from extract_bill import find_bill_details  # the real regex under audit

EVAL = os.path.join(ROOT, "data", "eval")
RESULTS = os.path.join(ROOT, "results")
SAMPLES = os.path.join(RESULTS, "samples")
os.makedirs(SAMPLES, exist_ok=True)


# ----------------------- helpers -----------------------
def parse_amount(s):
    """Extract a US-style decimal amount from a ground-truth total string."""
    m = re.findall(r"\d[\d,]*\.\d{2}", str(s).replace(" ", ""))
    if not m:
        m2 = re.findall(r"\d[\d,]*", str(s))
        return float(m2[-1].replace(",", "")) if m2 else None
    return float(m[-1].replace(",", ""))


def norm_date(s):
    s = str(s).strip()
    for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%y",
                "%d %b %Y", "%d %B %Y"):
        try:
            from datetime import datetime
            return datetime.strptime(s.split()[0] if " " in s else s, fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    m = re.search(r"\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}", s)
    if m:
        for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y-%m-%d"):
            try:
                from datetime import datetime
                return datetime.strptime(m.group(), fmt).strftime("%Y-%m-%d")
            except Exception:
                continue
    return None


# ----------------------- (1) extraction baseline -----------------------
def eval_extraction():
    rows = [json.loads(l) for l in open(os.path.join(EVAL, "receipts.jsonl"), encoding="utf-8")]
    amt_ok = date_ok = merch_ok = both_ok = 0
    samples = []
    for r in rows:
        pred_amt, pred_date = find_bill_details(r["text"])
        gt_amt = parse_amount(r["gt_total"])
        gt_date = norm_date(r["gt_date"])
        a_ok = gt_amt is not None and abs(abs(pred_amt) - gt_amt) < 0.01
        d_ok = gt_date is not None and pred_date == gt_date
        # the regex extracts NO merchant -> always a miss
        m_ok = False
        amt_ok += a_ok
        date_ok += d_ok
        merch_ok += m_ok
        both_ok += (a_ok and d_ok)
        if len(samples) < 10:
            samples.append({"id": r["id"], "gt_amount": gt_amt, "pred_amount": abs(pred_amt),
                            "amount_ok": a_ok, "gt_date": gt_date, "pred_date": pred_date,
                            "date_ok": d_ok, "gt_merchant": r["gt_merchant"],
                            "pred_merchant": None, "merchant_ok": m_ok})
    n = len(rows)
    json.dump(samples, open(os.path.join(SAMPLES, "extraction_baseline_samples.json"), "w"), indent=2)
    return {
        "dataset": "SROIE (darentang/sroie) test split, 100 receipts",
        "n": n,
        "method": "find_bill_details regex (extract_bill.py:43) — first \\d+\\.\\d{2} match",
        "amount_accuracy": round(amt_ok / n, 4),
        "date_accuracy": round(date_ok / n, 4),
        "merchant_accuracy": round(merch_ok / n, 4),
        "exact_match_amount_and_date": round(both_ok / n, 4),
        "note": "Regex returns only (amount, date); it has NO merchant field, so merchant accuracy is 0 by construction.",
    }


# ----------------------- (2) categorization baselines -----------------------
KEYWORDS = {
    "groceries": ["grocer", "market", "mkt", "food", "walmart", "costco", "aldi", "kroger", "safeway"],
    "dining": ["dining", "restaurant", "cafe", "pizza", "coffee", "starbucks", "mcdonald", "eats", "grubhub", "doordash"],
    "transport": ["transport", "uber", "lyft", "gas", "fuel", "oil", "parking", "air", "taxi", "metro", "subway"],
    "utilities": ["utilit", "electric", "energy", "water", "gas co", "comcast", "verizon", "at&t", "mobile", "internet"],
    "rent": ["rent", "lease", "apartment", "apt", "property", "landlord", "residential"],
    "entertainment": ["entertain", "netflix", "spotify", "hulu", "cinema", "theatre", "games", "xbox", "disney", "music"],
    "health": ["health", "pharmacy", "medical", "clinic", "dental", "dr ", "lab", "rx", "diagnostic"],
    "shopping": ["shop", "amazon", "target", "store", "best buy", "ikea", "nike", "macy", "ebay", "apple"],
    "income": ["payroll", "deposit", "salary", "refund", "interest", "dividend", "payout", "income"],
    "other": [],
}


def keyword_predict(desc):
    d = desc.lower()
    for cat, kws in KEYWORDS.items():
        for kw in kws:
            if kw in d:
                return cat
    return "other"


def macro_f1(y_true, y_pred, labels):
    f1s = {}
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s[c] = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0
    return round(sum(f1s.values()) / len(labels), 4), f1s


def eval_categorization():
    import csv
    rows = list(csv.DictReader(open(os.path.join(EVAL, "transactions.csv"), encoding="utf-8")))
    y_true = [r["category"] for r in rows]
    labels = sorted(set(y_true))
    acc = lambda yp: round(sum(1 for t, p in zip(y_true, yp) if t == p) / len(y_true), 4)

    majority = Counter(y_true).most_common(1)[0][0]
    y_majority = [majority] * len(rows)
    y_keyword = [keyword_predict(r["description"]) for r in rows]

    mf1_maj, perclass_maj = macro_f1(y_true, y_majority, labels)
    mf1_kw, perclass_kw = macro_f1(y_true, y_keyword, labels)

    samples = []
    for r, p in list(zip(rows, y_keyword))[:120]:
        if len(samples) < 10 and r["category"] != p:
            samples.append({"description": r["description"], "true": r["category"], "keyword_pred": p})
    json.dump(samples, open(os.path.join(SAMPLES, "categorization_baseline_samples.json"), "w"), indent=2)

    return {
        "dataset": "synthetic transactions (data/eval/transactions.csv, seed 20260625)",
        "n": len(rows),
        "n_classes": len(labels),
        "class_distribution": dict(Counter(y_true)),
        "majority_class": {"label": majority, "accuracy": acc(y_majority), "macro_f1": mf1_maj},
        "keyword": {"accuracy": acc(y_keyword), "macro_f1": mf1_kw, "per_class_f1": perclass_kw},
        "per_class_f1_majority": perclass_maj,
    }


def main():
    out = {
        "generated": "2026-06-25",
        "day": 1,
        "phase": "Phase 1 — audit + eval sets + baseline",
        "receipt_extraction_baseline": eval_extraction(),
        "expense_categorization_baseline": eval_categorization(),
    }
    path = os.path.join(RESULTS, "baseline_metrics.json")
    json.dump(out, open(path, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print("\nSAVED", path)


if __name__ == "__main__":
    main()
