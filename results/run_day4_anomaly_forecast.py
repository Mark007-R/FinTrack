"""Day 4 / Phase 2c -- Anomaly + recurring-bill detection + cash-flow forecast.

Runs three independent bake-offs on the synthetic 36-month spending stream
(data/eval/day4_transactions.csv) and writes one master comparison table plus
charts, sample outputs, and an append to results/metrics.json.

  1. ANOMALY DETECTION    IsolationForest vs robust z-score (MAD) vs STL-residual
                          -> AP, Precision@20, Recall@20, F1@native vs 20 injected
  2. RECURRING + DUPLICATE  amount/period clustering for subscriptions; same-merchant
                          same-amount within <=2d for duplicate charges
                          -> precision/recall/F1 vs ground truth
  3. CASH-FLOW FORECAST   seasonal-naive (lag12) vs naive (lag1) vs Prophet vs
                          LightGBM (lagged features), expanding walk-forward
                          -> MAPE / MAE on held-out months

All numbers are computed, never hand-typed. MEDIA DISCIPLINE: synthetic data only.
"""
import json
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL
import lightgbm as lgb

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "eval", "day4_transactions.csv")
SAMPLES = os.path.join(HERE, "samples")
os.makedirs(SAMPLES, exist_ok=True)
RNG = np.random.default_rng(20260628)

rows = []  # master comparison rows: dict(task, method, metric -> value)


def f1_at(y_true, scores, thr):
    pred = (scores >= thr).astype(int)
    return f1_score(y_true, pred, zero_division=0), pred


# =============================================================================
# load
# =============================================================================
df = pd.read_csv(DATA, parse_dates=["date"])
df["abs_amount"] = df["amount"].abs()
out = df[df["amount"] < 0].copy().reset_index(drop=True)   # outflows = spend
print(f"loaded {len(df)} txns ({len(out)} outflows); "
      f"{df.is_anomaly.sum()} anomalies, {df.is_duplicate.sum()} duplicates, "
      f"{df.is_recurring.sum()} recurring")

# =============================================================================
# 1) ANOMALY DETECTION  (transaction-level, on outflows)
# =============================================================================
y_anom = out["is_anomaly"].values
K = int(y_anom.sum())  # 20

# category median magnitude (robust scale reference)
cat_med = out.groupby("category")["abs_amount"].transform("median")

# --- method A: robust modified z-score, global on amount (Iglewicz-Hoaglin) ---
# The classic univariate anomaly baseline: flag by how far the spend magnitude
# sits from the median in MAD units. (We also checked a within-category variant;
# it degenerates because fixed recurring bills create near-zero-variance groups,
# so every $120 utility bill in an $80-flat group scores higher than a $2,399
# splurge -- documented in the report. Global is the fair, standard baseline.)
_x = out["abs_amount"].values
_med = np.median(_x)
_mad = np.median(np.abs(_x - _med)) or 1e-9
score_z = np.abs(0.6745 * (_x - _med) / _mad)

# --- method B: IsolationForest on engineered features ------------------------
feat = pd.DataFrame({
    "log_amt": np.log1p(out["abs_amount"].values),
    "amt_over_catmed": out["abs_amount"].values / np.maximum(cat_med.values, 1e-9),
    "dom": out["date"].dt.day.values,
    "dow": out["date"].dt.dayofweek.values,
})
Xs = StandardScaler().fit_transform(feat.values)
iso = IsolationForest(n_estimators=300, contamination="auto", random_state=42)
iso.fit(Xs)
score_iso = -iso.decision_function(Xs)  # higher = more anomalous

# --- method C: STL residual on the daily-spend series ------------------------
daily = (out.set_index("date")["abs_amount"].resample("D").sum().asfreq("D", fill_value=0.0))
stl = STL(daily, period=7, robust=True).fit()
resid = stl.resid
rmed, rmad = np.median(resid), (np.median(np.abs(resid - np.median(resid))) or 1e-9)
resid_z = 0.6745 * (resid - rmed) / rmad
day_score = resid_z.reindex(out["date"].dt.normalize().values).values
day_score = np.nan_to_num(np.abs(day_score), nan=0.0)
# attribute the day's anomaly to its single largest outflow (spike -> line item)
out["_dnorm"] = out["date"].dt.normalize()
is_day_max = out.groupby("_dnorm")["abs_amount"].transform("max").values == out["abs_amount"].values
score_stl = day_score * is_day_max.astype(float)

methods = {
    "robust_zscore_MAD": (score_z, 3.5),
    "isolation_forest": (score_iso, np.quantile(score_iso, 1 - K / len(out))),
    "stl_residual": (score_stl, 3.5),
}
anom_results = {}
for name, (sc, thr) in methods.items():
    ap = average_precision_score(y_anom, sc)
    order = np.argsort(-sc)
    topk = np.zeros_like(y_anom); topk[order[:K]] = 1
    p_at_k = precision_score(y_anom, topk, zero_division=0)
    r_at_k = recall_score(y_anom, topk, zero_division=0)
    f1_nat, pred_nat = f1_at(y_anom, sc, thr)
    n_flag = int(pred_nat.sum())
    anom_results[name] = dict(AP=ap, P_at_20=p_at_k, R_at_20=r_at_k,
                              F1_native=f1_nat, n_flagged=n_flag)
    rows.append(dict(task="anomaly", method=name, AP=round(ap, 3),
                     P_at_20=round(p_at_k, 3), R_at_20=round(r_at_k, 3),
                     F1_native=round(f1_nat, 3), n_flagged=n_flag))
    print(f"  [anomaly] {name:20s} AP={ap:.3f} P@20={p_at_k:.3f} "
          f"R@20={r_at_k:.3f} F1nat={f1_nat:.3f} (flagged {n_flag})")

# do the amount-based detectors catch the duplicate charges? (expected: no)
dup_mask = out["is_duplicate"].values == 1
for name, (sc, thr) in methods.items():
    pred = (sc >= thr)
    caught = int((pred & dup_mask).sum())
    anom_results[name]["duplicates_caught"] = caught
_dup_summary = ", ".join(f"{k}:{v['duplicates_caught']}" for k, v in anom_results.items())
print(f"  [anomaly] duplicate charges among outflows: {int(dup_mask.sum())} "
      f"-> caught by amount-based detectors: {_dup_summary}")

# anomaly samples
order_z = np.argsort(-score_iso)[:12]
anom_samples = []
for i in order_z:
    r = out.iloc[i]
    anom_samples.append(dict(date=str(r["date"].date()), merchant=r["merchant"],
                             category=r["category"], amount=float(r["amount"]),
                             is_true_anomaly=int(r["is_anomaly"]),
                             iso_score=round(float(score_iso[i]), 4),
                             modz=round(float(score_z[i]), 2)))
json.dump(anom_samples, open(os.path.join(SAMPLES, "phase2c_anomaly_samples.json"), "w"), indent=2)

# =============================================================================
# 2) RECURRING + DUPLICATE detection
# =============================================================================
def norm_merchant(m):
    s = "".join(ch for ch in str(m).upper() if ch.isalnum() or ch == " ")
    return " ".join(s.split()[:3])

df["m"] = df["merchant"].map(norm_merchant)
recur_pred = np.zeros(len(df), dtype=int)
detected_groups = []
for m, g in df.groupby("m"):
    if len(g) < 3:
        continue
    gg = g.sort_values("date")
    gaps = gg["date"].diff().dt.days.dropna().values
    if len(gaps) < 2:
        continue
    med_gap = np.median(gaps)
    gap_std = np.std(gaps)
    amts = gg["abs_amount"].values
    cv = np.std(amts) / (np.mean(amts) + 1e-9)
    # recurring if a regular cadence (weekly/biweekly/monthly) + stable amount
    cadence = (5 <= med_gap <= 9) or (12 <= med_gap <= 16) or (26 <= med_gap <= 35)
    regular = gap_std <= max(4.0, 0.35 * med_gap)
    stable = cv < 0.15
    if cadence and regular and stable:
        recur_pred[g.index] = 1
        detected_groups.append(dict(merchant=m, n=len(g),
                                    median_gap_days=round(float(med_gap), 1),
                                    amount_cv=round(float(cv), 3),
                                    mean_amount=round(float(amts.mean()), 2)))

y_rec = df["is_recurring"].values
rec_p = precision_score(y_rec, recur_pred, zero_division=0)
rec_r = recall_score(y_rec, recur_pred, zero_division=0)
rec_f1 = f1_score(y_rec, recur_pred, zero_division=0)
rows.append(dict(task="recurring", method="merchant_cadence_amount_cluster",
                 precision=round(rec_p, 3), recall=round(rec_r, 3),
                 f1=round(rec_f1, 3), n_flagged=int(recur_pred.sum()),
                 groups_found=len(detected_groups)))
print(f"  [recurring] P={rec_p:.3f} R={rec_r:.3f} F1={rec_f1:.3f} "
      f"({len(detected_groups)} subscription groups, {int(recur_pred.sum())} txns flagged)")

# --- duplicate charges: same merchant + ~same amount within <=2 days ----------
dup_pred = np.zeros(len(df), dtype=int)
dff = df.sort_values(["m", "date"]).reset_index()
for m, g in dff.groupby("m"):
    g = g.sort_values("date")
    idx = g["index"].values
    dates = g["date"].values
    amts = g["abs_amount"].values
    for i in range(1, len(g)):
        dgap = (dates[i] - dates[i - 1]) / np.timedelta64(1, "D")
        if dgap <= 2 and abs(amts[i] - amts[i - 1]) <= max(0.01, 0.01 * amts[i]):
            dup_pred[idx[i]] = 1  # flag the second (duplicate) charge
y_dup = df["is_duplicate"].values
dup_p = precision_score(y_dup, dup_pred, zero_division=0)
dup_r = recall_score(y_dup, dup_pred, zero_division=0)
dup_f1 = f1_score(y_dup, dup_pred, zero_division=0)
rows.append(dict(task="duplicate", method="same_merchant_amount_within_2d",
                 precision=round(dup_p, 3), recall=round(dup_r, 3),
                 f1=round(dup_f1, 3), n_flagged=int(dup_pred.sum())))
print(f"  [duplicate] P={dup_p:.3f} R={dup_r:.3f} F1={dup_f1:.3f} "
      f"({int(dup_pred.sum())} flagged, {int(y_dup.sum())} true)")

json.dump(dict(detected_subscription_groups=detected_groups),
          open(os.path.join(SAMPLES, "phase2c_recurring_samples.json"), "w"), indent=2)

# =============================================================================
# 3) CASH-FLOW FORECAST (monthly total outflow), expanding walk-forward
# =============================================================================
ms = (out.set_index("date")["abs_amount"].resample("MS").sum())
ms = ms.asfreq("MS")
y = ms.values.astype(float)
idx = ms.index
n = len(y)
TEST = 6  # last 6 months held out
test_start = n - TEST


def mape(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return float(np.mean(np.abs((a - p) / a)) * 100)


def mae(a, p):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(p, float))))


def make_feats(series):
    s = pd.Series(series)
    F = pd.DataFrame({
        "lag1": s.shift(1), "lag2": s.shift(2), "lag3": s.shift(3),
        "lag12": s.shift(12), "roll3": s.shift(1).rolling(3).mean(),
        "roll6": s.shift(1).rolling(6).mean(),
    })
    mo = idx.month.values
    F["sin"] = np.sin(2 * np.pi * mo / 12)
    F["cos"] = np.cos(2 * np.pi * mo / 12)
    return F


preds = {"seasonal_naive_lag12": [], "naive_lag1": [], "prophet": [], "lightgbm": []}
actuals = []
feats_all = make_feats(y)

# Prophet import once
try:
    from prophet import Prophet
    HAVE_PROPHET = True
except Exception:
    HAVE_PROPHET = False

for t in range(test_start, n):
    actuals.append(y[t])
    preds["seasonal_naive_lag12"].append(y[t - 12])
    preds["naive_lag1"].append(y[t - 1])

    # prophet: fit on history up to t
    if HAVE_PROPHET:
        hist = pd.DataFrame({"ds": idx[:t], "y": y[:t]})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False)
        m.fit(hist)
        fut = pd.DataFrame({"ds": [idx[t]]})
        preds["prophet"].append(float(m.predict(fut)["yhat"].iloc[0]))
    else:
        preds["prophet"].append(y[t - 12])

    # lightgbm: train on rows [12 .. t-1] with valid features, predict row t
    Xtr = feats_all.iloc[12:t].values
    ytr = y[12:t]
    Xte = feats_all.iloc[t:t + 1].values
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                              max_depth=3, num_leaves=15, min_child_samples=5,
                              subsample=0.9, random_state=42, verbose=-1)
    model.fit(Xtr, ytr)
    preds["lightgbm"].append(float(model.predict(Xte)[0]))

fc_results = {}
for name, p in preds.items():
    if name == "prophet" and not HAVE_PROPHET:
        continue
    fc_results[name] = dict(MAPE=mape(actuals, p), MAE=mae(actuals, p))
    rows.append(dict(task="forecast", method=name,
                     MAPE=round(fc_results[name]["MAPE"], 2),
                     MAE=round(fc_results[name]["MAE"], 1)))
    print(f"  [forecast] {name:22s} MAPE={fc_results[name]['MAPE']:6.2f}%  "
          f"MAE={fc_results[name]['MAE']:.0f}")

json.dump(dict(months=[str(d.date()) for d in idx[test_start:]],
               actual=[round(float(a), 2) for a in actuals],
               predictions={k: [round(float(x), 2) for x in v]
                            for k, v in preds.items()
                            if not (k == "prophet" and not HAVE_PROPHET)}),
          open(os.path.join(SAMPLES, "phase2c_forecast_samples.json"), "w"), indent=2)

# =============================================================================
# write master CSV + charts + metrics.json append
# =============================================================================
master = pd.DataFrame(rows)
master_path = os.path.join(HERE, "phase2c_anomaly_forecast.csv")
master.to_csv(master_path, index=False)
print(f"\nwrote {master_path}")

# chart 1: anomaly precision/recall@20 + AP
fig, ax = plt.subplots(figsize=(8, 4.5))
names = list(anom_results)
x = np.arange(len(names)); w = 0.25
ax.bar(x - w, [anom_results[n]["AP"] for n in names], w, label="AP")
ax.bar(x, [anom_results[n]["P_at_20"] for n in names], w, label="Precision@20")
ax.bar(x + w, [anom_results[n]["R_at_20"] for n in names], w, label="Recall@20")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=12, ha="right")
ax.set_ylim(0, 1.05); ax.set_ylabel("score")
ax.set_title("Anomaly detection (20 injected amount-outliers, n=%d outflows)" % len(out))
ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(HERE, "phase2c_anomaly_detection.png"), dpi=110)
plt.close(fig)

# chart 2: anomaly timeline (daily spend with true anomalies marked)
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(daily.index, daily.values, lw=0.7, color="#3b6", label="daily spend")
ta = out[out["is_anomaly"] == 1]
ax.scatter(ta["date"], ta["abs_amount"], color="crimson", zorder=5, s=28,
           label="injected anomaly")
ax.set_title("Daily spend stream with injected anomalies (36 months)")
ax.set_ylabel("USD/day"); ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(HERE, "phase2c_anomaly_timeline.png"), dpi=110)
plt.close(fig)

# chart 3: cash-flow forecast on held-out months
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(idx, y, color="#333", lw=1.3, label="actual monthly spend")
months_test = idx[test_start:]
colors = {"seasonal_naive_lag12": "#888", "naive_lag1": "#c90",
          "prophet": "#06c", "lightgbm": "#c0392b"}
for name, p in preds.items():
    if name == "prophet" and not HAVE_PROPHET:
        continue
    ax.plot(months_test, p, "o--", color=colors.get(name), lw=1.4, ms=4,
            label=f"{name} (MAPE {fc_results[name]['MAPE']:.1f}%)")
ax.axvspan(months_test[0], months_test[-1], color="grey", alpha=0.08)
ax.set_title("Cash-flow forecast: next-month total spend (expanding walk-forward)")
ax.set_ylabel("USD/month"); ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(os.path.join(HERE, "phase2c_cashflow_forecast.png"), dpi=110)
plt.close(fig)
print("wrote 3 charts")

# append to metrics.json (existing file is a list of per-day blocks)
mpath = os.path.join(HERE, "metrics.json")
allm = json.load(open(mpath)) if os.path.exists(mpath) else []
if isinstance(allm, dict):
    allm = [allm]
day4_block = {
    "day": "day4_phase2c",
    "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "dataset": {"path": "data/eval/day4_transactions.csv", "n_txns": int(len(df)),
                "n_outflows": int(len(out)), "n_anomalies": int(df.is_anomaly.sum()),
                "n_duplicates": int(df.is_duplicate.sum()),
                "n_recurring": int(df.is_recurring.sum()),
                "months": int(n), "test_months": TEST},
    "anomaly": anom_results,
    "recurring": dict(precision=rec_p, recall=rec_r, f1=rec_f1,
                      groups_found=len(detected_groups),
                      n_flagged=int(recur_pred.sum())),
    "duplicate": dict(precision=dup_p, recall=dup_r, f1=dup_f1,
                      n_flagged=int(dup_pred.sum()), n_true=int(y_dup.sum())),
    "forecast": fc_results,
}
allm = [b for b in allm if not (isinstance(b, dict) and b.get("day") == "day4_phase2c")]
allm.append(day4_block)
json.dump(allm, open(mpath, "w"), indent=2)
print(f"appended day4_phase2c to {mpath}")
print("\nDONE.")
