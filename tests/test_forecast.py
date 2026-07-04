"""Cash-flow forecast tests — Day-4 champion (Prophet) + seasonal-naive fallback.

Verifies the insufficient-history guard, the horizon contract, non-negativity,
and that the free seasonal-naive fallback is used when <24 months of history.
"""
from __future__ import annotations

from datetime import date

from src.forecast import forecast_cashflow


def _monthly(n_months, amount=-100.0, start=(2023, 1)):
    """One spend transaction per month for n_months."""
    y, m = start
    txns = []
    for _ in range(n_months):
        txns.append({"date": str(date(y, m, 15)), "merchant": "X",
                     "category": "groceries", "amount": amount})
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return txns


def test_insufficient_history_guard():
    res = forecast_cashflow(_monthly(2))
    assert res["method"] == "insufficient_history"
    assert res["forecast"] == []


def test_forecast_returns_horizon_length():
    res = forecast_cashflow(_monthly(8), horizon_months=3)
    assert len(res["forecast"]) == 3
    assert all(set(f) == {"month", "predicted_spend"} for f in res["forecast"])


def test_forecast_values_non_negative():
    res = forecast_cashflow(_monthly(10))
    assert all(f["predicted_spend"] >= 0 for f in res["forecast"])


def test_seasonal_naive_used_under_24_months():
    res = forecast_cashflow(_monthly(10))
    assert res["method"] == "seasonal_naive_lag12"


def test_last_actual_reported():
    res = forecast_cashflow(_monthly(6, amount=-250.0))
    assert res["last_actual"] == 250.0
    assert res["history_months"] == 6


def test_forecast_month_labels_advance():
    res = forecast_cashflow(_monthly(6, start=(2023, 1)), horizon_months=2)
    months = [f["month"] for f in res["forecast"]]
    assert months == ["2023-07", "2023-08"]
