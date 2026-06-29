"""Anomaly + recurring + duplicate detection — Day-4 Phase-2c champions.

Day-4 bake-off (3,054-txn synthetic stream, 20 injected anomalies):

    robust z-score (global, MAD)   AP 0.400   <- baseline; flags your own rent
    STL residual (daily aggregate) AP 0.256
    IsolationForest (multivariate) AP 0.979   <- champion (this module)

The genuine insight: a *global amount threshold flags fixed costs like rent as
fraud*. The fix is context-relative features (amount / category-median), which is
why IsolationForest jumps 0.40 -> 0.98. Duplicate charges are normal-sized, so
they need a SEPARATE merchant+amount+time rule (amount detectors caught 0/6).
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import numpy as np


def _parse_date(s):
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(str(s)[:10], fmt)
        except Exception:
            continue
    return None


def _norm_merchant(m):
    s = "".join(ch for ch in str(m).upper() if ch.isalnum() or ch == " ")
    return " ".join(s.split()[:3])


def detect_anomalies(transactions: list[dict], top_k: int | None = None) -> dict:
    """IsolationForest on context-relative features over a user's outflows.

    transactions: list of {date, merchant, category, amount}. amount<0 = spend.
    Returns flags sorted by descending anomaly score.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    out = [t for t in transactions if float(t.get("amount", 0)) < 0]
    if len(out) < 8:
        return {"n_transactions": len(transactions), "n_flagged": 0, "flags": []}

    abs_amt = np.array([abs(float(t["amount"])) for t in out])
    cats = [t.get("category", "other") for t in out]
    dates = [_parse_date(t.get("date")) for t in out]

    # category-median magnitude => context-relative scale (the key feature)
    cat_sum, cat_cnt = defaultdict(float), defaultdict(int)
    by_cat = defaultdict(list)
    for a, c in zip(abs_amt, cats):
        by_cat[c].append(a)
    cat_med = {c: float(np.median(v)) for c, v in by_cat.items()}

    feat = np.column_stack([
        np.log1p(abs_amt),
        abs_amt / np.array([max(cat_med.get(c, 1e-9), 1e-9) for c in cats]),
        np.array([d.day if d else 15 for d in dates]),
        np.array([d.weekday() if d else 0 for d in dates]),
    ])
    Xs = StandardScaler().fit_transform(feat)
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=42)
    iso.fit(Xs)
    score = -iso.decision_function(Xs)  # higher = more anomalous

    # native threshold: IsolationForest's own boundary (score>0 == below avg path)
    flagged_mask = iso.predict(Xs) == -1
    order = np.argsort(-score)
    flags = []
    for i in order:
        a = abs_amt[i]
        med = cat_med.get(cats[i], a)
        ratio = a / max(med, 1e-9)
        is_anom = bool(flagged_mask[i])
        reason = (f"{ratio:.1f}x the {cats[i]} median (${med:.0f})"
                  if ratio >= 1.5 else "unusual timing/amount pattern")
        flags.append({
            "date": out[i].get("date"), "merchant": out[i].get("merchant", ""),
            "category": cats[i], "amount": float(out[i]["amount"]),
            "score": round(float(score[i]), 4), "is_anomaly": is_anom,
            "reason": reason,
        })
    if top_k:
        flags = flags[:top_k]
    else:
        flags = [f for f in flags if f["is_anomaly"]] or flags[:5]
    return {"n_transactions": len(transactions),
            "n_flagged": sum(1 for f in flags if f["is_anomaly"]), "flags": flags}


def find_recurring_groups(transactions: list[dict]) -> list[dict]:
    """Cadence + amount-stability clustering (Day-4 champion, F1 0.994)."""
    by_m = defaultdict(list)
    for t in transactions:
        by_m[_norm_merchant(t.get("merchant", ""))].append(t)
    groups = []
    for m, items in by_m.items():
        if len(items) < 3 or not m:
            continue
        items = sorted(items, key=lambda t: _parse_date(t.get("date")) or datetime.min)
        ds = [_parse_date(t.get("date")) for t in items]
        gaps = [(ds[i] - ds[i - 1]).days for i in range(1, len(ds)) if ds[i] and ds[i - 1]]
        if len(gaps) < 2:
            continue
        med_gap = float(np.median(gaps)); gap_std = float(np.std(gaps))
        amts = np.array([abs(float(t["amount"])) for t in items])
        cv = float(np.std(amts) / (np.mean(amts) + 1e-9))
        cadence = (5 <= med_gap <= 9) or (12 <= med_gap <= 16) or (26 <= med_gap <= 35)
        regular = gap_std <= max(4.0, 0.35 * med_gap)
        if cadence and regular and cv < 0.15:
            groups.append({"merchant": m, "n": len(items),
                           "median_gap_days": round(med_gap, 1),
                           "amount_cv": round(cv, 3),
                           "mean_amount": round(float(amts.mean()), 2)})
    return groups


def find_duplicate_charges(transactions: list[dict]) -> list[dict]:
    """Same merchant + ~same amount within <=2 days (Day-4 separate detector)."""
    by_m = defaultdict(list)
    for t in transactions:
        by_m[_norm_merchant(t.get("merchant", ""))].append(t)
    dups = []
    for m, items in by_m.items():
        items = sorted(items, key=lambda t: _parse_date(t.get("date")) or datetime.min)
        for i in range(1, len(items)):
            d0, d1 = _parse_date(items[i - 1].get("date")), _parse_date(items[i].get("date"))
            if not d0 or not d1:
                continue
            a0, a1 = abs(float(items[i - 1]["amount"])), abs(float(items[i]["amount"]))
            if (d1 - d0).days <= 2 and abs(a1 - a0) <= max(0.01, 0.01 * a1):
                dups.append({"merchant": m, "date": items[i].get("date"),
                             "amount": float(items[i]["amount"]),
                             "prior_date": items[i - 1].get("date")})
    return dups
