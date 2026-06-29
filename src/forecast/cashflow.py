"""Cash-flow forecast — Day-4 Phase-2c champion (Prophet) + cheap fallbacks.

Day-4 bake-off (next-month total spend, expanding walk-forward):

    naive (lag-1)          MAPE 26.25%   <- baseline
    seasonal-naive (lag-12)MAPE 18.91%   <- strong free fallback
    LightGBM (lagged feat) MAPE 17.08%
    Prophet                MAPE 15.79%   <- champion (this module)

Insight: Prophet beats free seasonal-naive by only ~3 MAPE points, so the cheap
baseline ships first and Prophet is used when >=24 months of history exist.
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


def _monthly_outflow(transactions: list[dict]):
    """Aggregate signed transactions into a sorted monthly total-spend series."""
    bucket = defaultdict(float)
    for t in transactions:
        amt = float(t.get("amount", 0))
        if amt >= 0:
            continue
        d = _parse_date(t.get("date"))
        if not d:
            continue
        bucket[(d.year, d.month)] += abs(amt)
    months = sorted(bucket)
    series = [bucket[m] for m in months]
    return months, series


def _next_month(y, m):
    return (y + 1, 1) if m == 12 else (y, m + 1)


def forecast_cashflow(transactions: list[dict], horizon_months: int = 1) -> dict:
    months, series = _monthly_outflow(transactions)
    if len(series) < 3:
        return {"method": "insufficient_history", "history_months": len(series),
                "forecast": [], "last_actual": series[-1] if series else None}

    y = np.array(series, dtype=float)
    method = "seasonal_naive_lag12"

    def seasonal_naive(hist, steps):
        preds = []
        ext = list(hist)
        for _ in range(steps):
            preds.append(ext[-12] if len(ext) >= 12 else ext[-1])
            ext.append(preds[-1])
        return preds

    preds = None
    if len(series) >= 24:
        try:
            from prophet import Prophet
            ds = [datetime(yr, mo, 1) for (yr, mo) in months]
            import pandas as pd
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                        daily_seasonality=False)
            m.fit(pd.DataFrame({"ds": ds, "y": y}))
            future_months = []
            cy, cm = months[-1]
            for _ in range(horizon_months):
                cy, cm = _next_month(cy, cm)
                future_months.append(datetime(cy, cm, 1))
            fc = m.predict(pd.DataFrame({"ds": future_months}))["yhat"].tolist()
            preds = [max(0.0, float(v)) for v in fc]
            method = "prophet"
        except Exception:
            preds = None

    if preds is None:
        preds = seasonal_naive(list(y), horizon_months)

    fm = []
    cy, cm = months[-1]
    for v in preds:
        cy, cm = _next_month(cy, cm)
        fm.append({"month": f"{cy:04d}-{cm:02d}", "predicted_spend": round(float(v), 2)})

    return {"method": method, "history_months": len(series),
            "forecast": fm, "last_actual": round(float(y[-1]), 2)}
