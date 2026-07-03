"""Dashboard data + figure layer (Day-9).

Pure, importable functions that turn transaction lists and API responses into the
DataFrames and matplotlib figures the Streamlit app renders. Kept separate from
the Streamlit UI so the whole data path is unit-testable headlessly (the Day-9
harness imports this module directly — no Streamlit runtime required).

Media discipline: the demo stream is synthetic (seeded), never real financial
data. In the running app these transactions come from the JWT-scoped
`GET /transactions` endpoint instead.
"""
from __future__ import annotations

import random
from collections import defaultdict
from datetime import date, timedelta

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CATEGORIES = ["groceries", "dining", "transport", "utilities", "rent",
              "entertainment", "health", "shopping"]

# representative merchants per category (synthetic; mirrors the categorizer's vocab)
_MERCHANTS = {
    "groceries": ["WALMART", "COSTCO", "KROGER", "ALDI", "SAFEWAY"],
    "dining": ["STARBUCKS", "CHIPOTLE", "DOORDASH", "MCDONALDS", "UBER EATS"],
    "transport": ["UBER", "SHELL FUEL", "LYFT", "METRO TRANSIT", "PARKING"],
    "utilities": ["PG&E", "COMCAST", "VERIZON", "AT&T", "CITY WATER"],
    "rent": ["GREENBRIAR APARTMENTS"],
    "entertainment": ["NETFLIX", "SPOTIFY", "HULU", "DISNEY+", "STEAM GAMES"],
    "health": ["CVS PHARMACY", "QUEST DIAGNOSTIC", "DENTAL CLINIC"],
    "shopping": ["AMAZON", "TARGET", "BEST BUY", "NIKE", "IKEA"],
}


def synthetic_user_stream(seed: int = 7, months: int = 8) -> list[dict]:
    """A seeded, synthetic per-user transaction stream (income + spend + 1 anomaly)."""
    rng = random.Random(seed)
    start = date(2025, 11, 1) - timedelta(days=30 * months)
    txns: list[dict] = []
    for m in range(months):
        month_start = date(start.year + (start.month - 1 + m) // 12,
                           (start.month - 1 + m) % 12 + 1, 1)
        # monthly income
        txns.append({"date": month_start.isoformat(), "merchant": "ACME PAYROLL",
                     "category": "income", "amount": round(rng.uniform(4200, 4800), 2)})
        # fixed rent
        txns.append({"date": (month_start + timedelta(days=1)).isoformat(),
                     "merchant": "GREENBRIAR APARTMENTS", "category": "rent",
                     "amount": -1650.0})
        # variable spend
        for _ in range(rng.randint(18, 26)):
            cat = rng.choice(CATEGORIES)
            merch = rng.choice(_MERCHANTS[cat])
            base = {"groceries": 70, "dining": 28, "transport": 22, "utilities": 120,
                    "rent": 1650, "entertainment": 15, "health": 45, "shopping": 60}[cat]
            amt = -round(abs(rng.gauss(base, base * 0.4)) + 1, 2)
            day = month_start + timedelta(days=rng.randint(2, 27))
            txns.append({"date": day.isoformat(), "merchant": merch,
                         "category": cat, "amount": amt})
    # one injected anomaly (large electronics splurge)
    txns.append({"date": (start + timedelta(days=30 * (months - 1) + 12)).isoformat(),
                 "merchant": "BEST BUY", "category": "shopping", "amount": -2399.0})
    return txns


# --------------------------------------------------------------------------- #
# DataFrames
# --------------------------------------------------------------------------- #
def _month_key(d: str) -> str:
    return str(d)[:7]


def category_month_matrix(transactions: list[dict]) -> pd.DataFrame:
    """month x category spend matrix (absolute outflow), for the heat map."""
    agg: dict = defaultdict(lambda: defaultdict(float))
    for t in transactions:
        amt = float(t["amount"])
        if amt >= 0:
            continue  # spend only
        cat = t.get("category", "other")
        agg[_month_key(t["date"])][cat] += -amt
    months = sorted(agg.keys())
    cats = CATEGORIES
    mat = pd.DataFrame([[round(agg[mn].get(c, 0.0), 2) for c in cats] for mn in months],
                       index=months, columns=cats)
    return mat


def balance_trend(transactions: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(transactions)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["balance"] = df["amount"].astype(float).cumsum()
    return df[["date", "amount", "balance"]].reset_index(drop=True)


def anomaly_alert_table(anomaly_response: dict) -> pd.DataFrame:
    flags = [f for f in anomaly_response.get("flags", []) if f.get("is_anomaly")]
    if not flags:
        return pd.DataFrame(columns=["date", "merchant", "category", "amount", "score", "reason"])
    df = pd.DataFrame(flags)
    keep = [c for c in ["date", "merchant", "category", "amount", "score", "reason"] if c in df.columns]
    return df[keep].sort_values("score", ascending=False).reset_index(drop=True)


def forecast_table(forecast_response: dict) -> pd.DataFrame:
    fc = forecast_response.get("forecast", [])
    return pd.DataFrame(fc) if fc else pd.DataFrame(columns=["month", "predicted_spend"])


# --------------------------------------------------------------------------- #
# Figures (matplotlib, Agg — testable / embeddable)
# --------------------------------------------------------------------------- #
def fig_category_heatmap(matrix: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    data = matrix.values.astype(float)
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_title("Monthly spend by category ($)", fontsize=11, fontweight="bold")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center",
                    fontsize=6, color="black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="$ spent")
    fig.tight_layout()
    return fig


def fig_balance_trend(trend: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(trend["date"], trend["balance"], color="#1f77b4", lw=1.8)
    ax.fill_between(trend["date"], trend["balance"], alpha=0.15, color="#1f77b4")
    ax.axhline(0, color="grey", lw=0.7, ls="--")
    ax.set_title("Running balance (per-user)", fontsize=11, fontweight="bold")
    ax.set_ylabel("$")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def fig_cashflow_forecast(trend: pd.DataFrame, forecast_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    # historical monthly outflow
    hist = trend.copy()
    hist["month"] = hist["date"].dt.strftime("%Y-%m")
    spend = hist[hist["amount"] < 0].groupby("month")["amount"].sum().abs()
    ax.bar(range(len(spend)), spend.values, color="#8888cc", label="actual spend")
    if not forecast_df.empty and "predicted_spend" in forecast_df:
        n = len(spend)
        ax.bar(range(n, n + len(forecast_df)), forecast_df["predicted_spend"].values,
               color="#d62728", alpha=0.85, label="forecast")
    ax.set_title("Cash-flow: monthly spend + next-month forecast", fontsize=11, fontweight="bold")
    ax.set_ylabel("$ outflow")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
