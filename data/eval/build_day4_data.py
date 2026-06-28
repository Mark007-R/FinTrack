"""Reproducible generator for the Day-4 anomaly / recurring / cash-flow eval set.

MEDIA DISCIPLINE: 100% synthetic. No real personal financial data is used.

This extends the Day-1 single-month categorization set into a 36-month DAILY
spending stream for ONE user, so the three Day-4 components each have ground
truth:

  * cash-flow forecast  -> a long enough monthly-spend series (36 points) with
    real yearly seasonality (Nov/Dec holiday bump, summer-travel bump) and a
    mild inflation trend, so seasonal-naive / Prophet / gradient-boosted can be
    compared on held-out months.
  * recurring-bill detection -> a fixed set of subscriptions billed monthly at a
    stable amount (NETFLIX, SPOTIFY, RENT, gym, phone, SaaS) + a biweekly
    payroll inflow, each tagged is_recurring=1 with a recurring_group.
  * anomaly detection -> 20 injected amount-outliers tagged is_anomaly=1, plus a
    handful of duplicate charges (is_duplicate=1) that are *not* amount outliers
    -- on purpose, to show amount-based anomaly detectors miss duplicate-charge
    fraud that the recurring/duplicate detector catches.

Run:  python data/eval/build_day4_data.py  ->  data/eval/day4_transactions.csv
"""
import csv
import os
from datetime import date, timedelta

import numpy as np

SEED = 20260628
rng = np.random.default_rng(SEED)

START = date(2023, 1, 1)
MONTHS = 36  # 2023-01 .. 2025-12

# ---- discretionary (non-recurring) merchants per category --------------------
MERCHANTS = {
    "groceries": ["WALMART SUPERCENTER", "TRADER JOE'S", "WHOLE FOODS MKT", "SAFEWAY",
                  "KROGER", "ALDI", "COSTCO WHSE", "PUBLIX"],
    "dining": ["MCDONALD'S", "CHIPOTLE", "DOORDASH", "UBER EATS", "OLIVE GARDEN",
               "PANERA BREAD", "GRUBHUB", "SQ *TAQUERIA"],
    "transport": ["SHELL OIL", "UBER TRIP", "LYFT RIDE", "CHEVRON", "EXXONMOBIL",
                  "MTA SUBWAY", "76 GAS", "PARKING METER"],
    "health": ["CVS/PHARMACY", "WALGREENS", "QUEST DIAGNOSTICS", "LABCORP",
               "ONE MEDICAL", "RITE AID"],
    "shopping": ["AMAZON.COM", "TARGET", "BEST BUY", "IKEA", "NIKE STORE",
                 "HOME DEPOT", "ETSY", "APPLE STORE"],
    "entertainment": ["AMC THEATRES", "TICKETMASTER", "REGAL CINEMAS", "STEAM GAMES"],
    "other": ["ATM WITHDRAWAL", "VENMO PAYMENT", "PAYPAL TRANSFER", "USPS POSTAGE"],
}
# typical amount range per discretionary category (outflow magnitudes)
AMT_RANGE = {
    "groceries": (8, 180), "dining": (5, 90), "transport": (3, 120),
    "health": (10, 300), "shopping": (8, 400), "entertainment": (8, 110),
    "other": (5, 250),
}
CAT_WEIGHTS = {"groceries": 22, "dining": 20, "transport": 16, "shopping": 15,
               "health": 9, "entertainment": 9, "other": 9}

# ---- recurring subscriptions (monthly, stable amount) ------------------------
# (merchant, category, amount-outflow, billing day-of-month, recurring_group)
SUBSCRIPTIONS = [
    ("RENT - GREYSTAR PROPERTY", "rent", 1850.00, 1, "rent"),
    ("VERIZON WIRELESS", "utilities", 80.00, 5, "phone"),
    ("CON EDISON", "utilities", 120.00, 8, "power"),      # mild noise added below
    ("NETFLIX", "entertainment", 15.99, 12, "netflix"),
    ("SPOTIFY", "entertainment", 10.99, 14, "spotify"),
    ("PLANET FITNESS", "health", 24.99, 18, "gym"),
    ("ADOBE CREATIVE CLOUD", "shopping", 52.99, 22, "adobe"),
]
PAYROLL_AMT = 3200.00  # biweekly inflow (recurring income)


def seasonal_factor(d: date) -> float:
    """Yearly seasonality on discretionary spend volume."""
    f = 1.0
    if d.month in (11, 12):
        f *= 1.35          # holiday shopping bump
    if d.month in (6, 7, 8):
        f *= 1.15          # summer travel bump
    if d.month in (1, 2):
        f *= 0.90          # post-holiday belt-tightening
    return f


def month_index(d: date) -> int:
    return (d.year - START.year) * 12 + (d.month - START.month)


def main():
    rows = []
    rid = 0

    # 1) discretionary daily transactions ------------------------------------
    cats = list(CAT_WEIGHTS)
    wts = np.array([CAT_WEIGHTS[c] for c in cats], float)
    wts /= wts.sum()
    end = date(START.year + (START.month - 1 + MONTHS - 1) // 12,
               (START.month - 1 + MONTHS - 1) % 12 + 1, 28)
    d = START
    while d <= end:
        trend = 1.0 + 0.0035 * month_index(d)          # ~0.35%/mo inflation drift
        lam = 2.2 * seasonal_factor(d) * trend          # avg discretionary txns/day
        n_today = rng.poisson(lam)
        for _ in range(n_today):
            cat = rng.choice(cats, p=wts)
            merch = rng.choice(MERCHANTS[cat])
            lo, hi = AMT_RANGE[cat]
            amt = -round(float(rng.uniform(lo, hi)) * trend, 2)
            rows.append([d.isoformat(), str(merch), cat, amt, 0, "", 0, 0, ""])
        d += timedelta(days=1)

    # 2) recurring subscriptions (monthly) -----------------------------------
    for m in range(MONTHS):
        y = START.year + (START.month - 1 + m) // 12
        mo = (START.month - 1 + m) % 12 + 1
        for merch, cat, amt, dom, grp in SUBSCRIPTIONS:
            # con edison varies a little (real utility bills do); others are flat
            noise = rng.uniform(-12, 18) if grp == "power" else rng.uniform(-0.0, 0.0)
            day = min(dom, 28)
            rows.append([date(y, mo, day).isoformat(), merch, cat,
                         -round(amt + noise, 2), 1, grp, 0, 0, ""])

    # 3) biweekly payroll inflow (recurring income) --------------------------
    pd = date(START.year, START.month, 3)
    while pd <= end:
        rows.append([pd.isoformat(), "PAYROLL DEPOSIT - ACME CORP", "income",
                     round(PAYROLL_AMT, 2), 1, "payroll", 0, 0, ""])
        pd += timedelta(days=14)

    # 4) inject 20 amount-anomalies ------------------------------------------
    # big, clearly-out-of-distribution outflows spread across the timeline
    anomaly_specs = [
        ("BEST BUY",            "shopping",   -2399.00, "big_electronics"),
        ("APPLE STORE",         "shopping",   -1899.00, "big_electronics"),
        ("HOME DEPOT",          "shopping",   -1450.00, "big_purchase"),
        ("DELTA AIR",           "transport",  -1280.00, "expensive_travel"),
        ("EXPEDIA TRAVEL",      "transport",  -1640.00, "expensive_travel"),
        ("QUEST DIAGNOSTICS",   "health",     -1750.00, "large_medical"),
        ("ONE MEDICAL",         "health",      -980.00, "large_medical"),
        ("OLIVE GARDEN",        "dining",      -742.00, "huge_dining"),
        ("DOORDASH",            "dining",      -515.00, "huge_dining"),
        ("WHOLE FOODS MKT",     "groceries",   -690.00, "huge_grocery"),
        ("COSTCO WHSE",         "groceries",   -880.00, "huge_grocery"),
        ("TICKETMASTER",        "entertainment", -1320.00, "concert_splurge"),
        ("AMAZON.COM",          "shopping",   -2050.00, "big_electronics"),
        ("NIKE STORE",          "shopping",    -915.00, "big_purchase"),
        ("SHELL OIL",           "transport",   -430.00, "fuel_outlier"),
        ("CVS/PHARMACY",        "health",      -610.00, "large_medical"),
        ("IKEA",                "shopping",   -1730.00, "big_purchase"),
        ("WIRE TRANSFER OUT",   "other",      -3200.00, "large_transfer"),
        ("ATM WITHDRAWAL",      "other",       -800.00, "cash_outlier"),
        ("TARGET",              "shopping",   -1180.00, "big_purchase"),
    ]
    anom_months = rng.choice(range(MONTHS), size=len(anomaly_specs), replace=True)
    for (merch, cat, amt, kind), m in zip(anomaly_specs, anom_months):
        y = START.year + (START.month - 1 + m) // 12
        mo = (START.month - 1 + m) % 12 + 1
        day = int(rng.integers(1, 28))
        rows.append([date(y, mo, day).isoformat(), merch, cat, amt, 0, "", 1, 0, kind])

    # 5) inject 6 duplicate charges (NOT amount-outliers) --------------------
    # exact-amount repeats of a subscription/normal charge within 0-2 days --
    # these are fraud/billing errors a *recurring/duplicate* detector should
    # catch but an amount-based anomaly detector will miss (genuine insight).
    dup_specs = [
        ("NETFLIX", "entertainment", -15.99, date(2023, 6, 12)),
        ("SPOTIFY", "entertainment", -10.99, date(2024, 2, 14)),
        ("DOORDASH", "dining", -42.50, date(2024, 9, 7)),
        ("VERIZON WIRELESS", "utilities", -80.00, date(2025, 1, 5)),
        ("ADOBE CREATIVE CLOUD", "shopping", -52.99, date(2025, 5, 22)),
        ("UBER TRIP", "transport", -28.30, date(2025, 8, 19)),
    ]
    for merch, cat, amt, dd in dup_specs:
        rows.append([(dd + timedelta(days=int(rng.integers(0, 3)))).isoformat(),
                     merch, cat, amt, 0, "", 0, 1, "duplicate_charge"])

    # sort by date, assign ids -----------------------------------------------
    rows.sort(key=lambda r: r[0])
    out = os.path.join(os.path.dirname(__file__), "day4_transactions.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "date", "merchant", "category", "amount",
                    "is_recurring", "recurring_group", "is_anomaly",
                    "is_duplicate", "anomaly_type"])
        for r in rows:
            d_, merch, cat, amt, isrec, grp, isan, isdup, kind = r
            w.writerow([rid, d_, merch, cat, amt, isrec, grp, isan, isdup, kind])
            rid += 1

    n = len(rows)
    print(f"wrote {out}")
    print(f"  rows               = {n}")
    print(f"  recurring (subs)   = {sum(r[4] for r in rows)}")
    print(f"  injected anomalies = {sum(r[6] for r in rows)}")
    print(f"  duplicate charges  = {sum(r[7] for r in rows)}")
    print(f"  date span          = {rows[0][0]} .. {rows[-1][0]}")


if __name__ == "__main__":
    main()
