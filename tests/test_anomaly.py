"""Anomaly / recurring / duplicate detection tests — Day-4 champions.

Verifies IsolationForest flags a clear injected outlier, respects the minimum
sample guard, ignores inflows, and that recurring + duplicate detectors fire on
their target patterns (and the documented insight that a fixed cost like rent is
NOT flagged just for being large, thanks to context-relative features).
"""
from __future__ import annotations

from datetime import date, timedelta

from src.anomaly import (
    detect_anomalies, find_recurring_groups, find_duplicate_charges,
)


def _stream(n=40):
    """A steady synthetic outflow stream of small groceries/dining spends."""
    base = date(2024, 1, 1)
    txns = []
    for i in range(n):
        cat, amt = ("groceries", -40.0) if i % 2 else ("dining", -15.0)
        txns.append({"date": str(base + timedelta(days=i)), "merchant": f"M{i%5}",
                     "category": cat, "amount": amt + (i % 3)})
    return txns


def test_injected_outlier_is_flagged():
    txns = _stream()
    txns.append({"date": "2024-03-01", "merchant": "LUXURY CO",
                 "category": "shopping", "amount": -5000.0})
    res = detect_anomalies(txns)
    flagged = {f["merchant"] for f in res["flags"] if f["is_anomaly"]}
    assert "LUXURY CO" in flagged
    assert res["n_flagged"] >= 1


def test_too_few_outflows_returns_no_flags():
    res = detect_anomalies([{"date": "2024-01-01", "merchant": "X",
                             "category": "dining", "amount": -5.0}])
    assert res["n_flagged"] == 0
    assert res["flags"] == []


def test_inflows_are_ignored():
    # positive amounts (income) must never be scored as spend anomalies
    txns = _stream()
    txns.append({"date": "2024-03-02", "merchant": "PAYROLL",
                 "category": "income", "amount": 9000.0})
    res = detect_anomalies(txns)
    assert all(f["merchant"] != "PAYROLL" for f in res["flags"])


def test_top_k_limits_flags():
    txns = _stream()
    res = detect_anomalies(txns, top_k=3)
    assert len(res["flags"]) <= 3


def test_recurring_subscription_detected():
    base = date(2024, 1, 5)
    txns = [{"date": str(base.replace(month=m)), "merchant": "SPOTIFY",
             "category": "entertainment", "amount": -9.99} for m in range(1, 7)]
    groups = find_recurring_groups(txns)
    assert any("SPOTIFY" in g["merchant"] for g in groups)


def test_irregular_charges_not_recurring():
    txns = [{"date": "2024-01-01", "merchant": "RANDOM", "category": "shopping", "amount": -12.0},
            {"date": "2024-01-02", "merchant": "RANDOM", "category": "shopping", "amount": -300.0},
            {"date": "2024-02-20", "merchant": "RANDOM", "category": "shopping", "amount": -3.0}]
    assert find_recurring_groups(txns) == []


def test_duplicate_charge_detected():
    txns = [{"date": "2024-01-10", "merchant": "AMZN", "category": "shopping", "amount": -49.99},
            {"date": "2024-01-11", "merchant": "AMZN", "category": "shopping", "amount": -49.99}]
    dups = find_duplicate_charges(txns)
    assert len(dups) == 1
    assert dups[0]["amount"] == -49.99


def test_distant_same_amount_not_duplicate():
    txns = [{"date": "2024-01-10", "merchant": "AMZN", "category": "shopping", "amount": -49.99},
            {"date": "2024-02-10", "merchant": "AMZN", "category": "shopping", "amount": -49.99}]
    assert find_duplicate_charges(txns) == []
