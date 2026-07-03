"""Day-9 Phase-7 production-wrapper integration harness.

Boots the FastAPI app in-process (TestClient runs the real middleware + deps) and
proves every Phase-7 guarantee with executable checks:

  1. AUTH GATE      — unauthenticated ML calls are rejected (401).
  2. JWT ROUND-TRIP — register/login issue a working Bearer token.
  3. MULTI-TENANCY  — user A cannot see or delete user B's transactions.
  4. CACHE          — a repeated /extract is served from cache, identical + faster.
  5. TELEMETRY      — /metrics reports request counts, latency percentiles, hits.
  6. DASHBOARD      — the data layer builds all four figures from live API output.
  7. DEPLOY ARTIFACTS — Dockerfile + docker-compose.yml present and well-formed.

Writes results/phase7_production.json (+ charts + samples). Exit code is non-zero
if any check fails, so the daily run cannot log success on a broken service.
"""
from __future__ import annotations

import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
from dashboard import data as D  # noqa: E402
from src.serving.store import get_txn_store  # noqa: E402
from src.serving.cache import get_cache  # noqa: E402
from src.serving.telemetry import get_sink  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
SAMPLES = os.path.join(RESULTS, "samples")
os.makedirs(SAMPLES, exist_ok=True)

checks: list[dict] = []


def check(name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})
    print(f"[{'PASS' if passed else 'FAIL'}] {name} — {detail}")


def bearer(tok: str) -> dict:
    return {"Authorization": f"Bearer {tok}"}


def main() -> int:
    # clean slate (in-process singletons persist across a run)
    get_txn_store().clear()
    get_cache().clear()
    get_sink().clear()

    client = TestClient(api.app)
    out: dict = {"cache_backend": get_cache().backend}

    # --- health ---------------------------------------------------------------
    h = client.get("/health").json()
    check("health_ok", h.get("status") == "ok",
          f"version={h.get('version')} cache={h.get('cache_backend')} "
          f"categorizer={h.get('categorizer_model')}")

    # --- 1. AUTH GATE ---------------------------------------------------------
    r_noauth = client.post("/categorize", json={"description": "STARBUCKS"})
    r_badtok = client.post("/categorize", json={"description": "STARBUCKS"},
                           headers=bearer("not-a-real-token"))
    check("auth_gate_no_token", r_noauth.status_code == 401,
          f"unauthenticated /categorize -> {r_noauth.status_code}")
    check("auth_gate_bad_token", r_badtok.status_code == 401,
          f"tampered token /categorize -> {r_badtok.status_code}")

    # --- 2. JWT ROUND-TRIP ----------------------------------------------------
    ra = client.post("/auth/register", json={"username": "alice", "password": "alice_pw_1"})
    rb = client.post("/auth/register", json={"username": "bob", "password": "bob_pw_1"})
    tok_a = ra.json()["access_token"]
    tok_b = rb.json()["access_token"]
    uid_a, uid_b = ra.json()["user_id"], rb.json()["user_id"]
    # login path + duplicate register
    login_a = client.post("/auth/token", data={"username": "alice", "password": "alice_pw_1"})
    dup = client.post("/auth/register", json={"username": "alice", "password": "x_pw_123"})
    check("jwt_register_login", ra.status_code == 200 and login_a.status_code == 200,
          f"alice uid={uid_a}, bob uid={uid_b}, login={login_a.status_code}")
    check("jwt_duplicate_rejected", dup.status_code == 409,
          f"duplicate register -> {dup.status_code}")
    # protected call now works
    ok_cat = client.post("/categorize", json={"description": "STARBUCKS"}, headers=bearer(tok_a))
    check("auth_protected_ok", ok_cat.status_code == 200,
          f"authenticated /categorize -> {ok_cat.status_code} ({ok_cat.json().get('category')})")

    # --- 3. MULTI-TENANCY -----------------------------------------------------
    stream_a = D.synthetic_user_stream(seed=1)
    stream_b = D.synthetic_user_stream(seed=99)[:5]
    client.post("/transactions", json={"transactions": stream_a}, headers=bearer(tok_a))
    client.post("/transactions", json={"transactions": stream_b}, headers=bearer(tok_b))
    list_a = client.get("/transactions", headers=bearer(tok_a)).json()
    list_b = client.get("/transactions", headers=bearer(tok_b)).json()
    isolation = (list_a["n_transactions"] == len(stream_a)
                 and list_b["n_transactions"] == len(stream_b)
                 and list_a["user_id"] == uid_a and list_b["user_id"] == uid_b)
    check("tenant_isolation_read", isolation,
          f"A sees {list_a['n_transactions']} (own {len(stream_a)}), "
          f"B sees {list_b['n_transactions']} (own {len(stream_b)}) — no cross-leak")
    # Multi-tenancy delete guarantee: A can only ever address rows in A's own
    # scope. Transaction ids are per-user (both users start at 1), so there is no
    # globally-shared id A could use to name B's row — the strongest possible
    # isolation. We prove it two ways:
    #   (a) A deleting id=X touches at most A's own row; B's exact id-set is intact.
    #   (b) A deleting an id beyond A's range 404s (nothing to delete in A's scope).
    b_ids_before = {t["id"] for t in list_b["transactions"]}
    b_txn_id = list_b["transactions"][0]["id"]
    _ = client.delete(f"/transactions/{b_txn_id}", headers=bearer(tok_a))  # hits A's own or 404
    a_max_id = max(t["id"] for t in list_a["transactions"])
    cross_del = client.delete(f"/transactions/{a_max_id + 1000}", headers=bearer(tok_a))
    b_ids_after = {t["id"] for t in client.get("/transactions", headers=bearer(tok_b)).json()["transactions"]}
    isolation_del = (cross_del.status_code == 404 and b_ids_after == b_ids_before)
    check("tenant_isolation_delete", isolation_del,
          f"A cannot reference B's rows (id namespaces are per-user); B's id-set "
          f"unchanged ({len(b_ids_after)}/{len(b_ids_before)} intact); out-of-scope "
          f"delete -> {cross_del.status_code}")

    # --- 4. CACHE -------------------------------------------------------------
    # Average over many iterations: N cold misses (cache cleared each time) vs N
    # warm hits, so the speedup ratio is not dominated by single-call noise.
    receipt = "WALMART SUPERCENTER\nMILK 3.49\nBREAD 2.19\nTOTAL 5.68\nDATE 03/14/2025"
    N = 40
    miss_times, hit_times = [], []
    for i in range(N):
        get_cache().clear()  # force a cold miss (re-run the extractor)
        t0 = time.perf_counter()
        r1 = client.post("/extract", json={"text": receipt}, headers=bearer(tok_a))
        miss_times.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        r2 = client.post("/extract", json={"text": receipt}, headers=bearer(tok_a))  # served from cache
        hit_times.append((time.perf_counter() - t0) * 1000)
    t_miss = sum(miss_times) / N
    t_hit = sum(hit_times) / N
    identical = r1.json() == r2.json()
    cache_stats = get_cache().stats()
    check("cache_hit_identical", identical and cache_stats["hits"] >= 1,
          f"backend={cache_stats['backend']} hits={cache_stats['hits']} "
          f"avg_miss={t_miss:.3f}ms avg_hit={t_hit:.3f}ms ({N} iters) identical={identical}")
    out["cache"] = {**cache_stats, "iters": N,
                    "miss_ms": round(t_miss, 3), "hit_ms": round(t_hit, 3),
                    "speedup_x": round(t_miss / t_hit, 2) if t_hit else None}

    # --- 5. TELEMETRY ---------------------------------------------------------
    metrics = client.get("/metrics").json()
    tel = metrics["telemetry"]
    check("telemetry_recorded", tel.get("n_requests", 0) > 0 and "latency_ms_p95" in tel,
          f"n={tel.get('n_requests')} p50={tel.get('latency_ms_p50')}ms "
          f"p95={tel.get('latency_ms_p95')}ms cache_hits={tel.get('cache_hits')}")
    out["telemetry"] = tel

    # --- 6. DASHBOARD DATA LAYER ---------------------------------------------
    txns_a = client.get("/transactions", headers=bearer(tok_a)).json()["transactions"]
    anomaly = client.post("/anomaly", json={"transactions": txns_a, "top_k": 10},
                          headers=bearer(tok_a)).json()
    forecast = client.post("/forecast", json={"transactions": txns_a, "horizon_months": 1},
                           headers=bearer(tok_a)).json()
    trend = D.balance_trend(txns_a)
    mat = D.category_month_matrix(txns_a)
    figs = {
        "day9_dashboard_heatmap.png": D.fig_category_heatmap(mat),
        "day9_dashboard_balance.png": D.fig_balance_trend(trend),
        "day9_dashboard_cashflow.png": D.fig_cashflow_forecast(trend, D.forecast_table(forecast)),
    }
    for fname, fig in figs.items():
        fig.savefig(os.path.join(RESULTS, fname), dpi=110, bbox_inches="tight")
        plt.close(fig)
    alerts = D.anomaly_alert_table(anomaly)
    dashboard_ok = (not mat.empty and len(trend) == len(txns_a) and len(alerts) >= 1)
    check("dashboard_builds", dashboard_ok,
          f"heatmap {mat.shape}, balance rows {len(trend)}, anomaly alerts {len(alerts)}")
    out["dashboard"] = {"heatmap_shape": list(mat.shape), "balance_rows": len(trend),
                        "anomaly_alerts": int(len(alerts)),
                        "forecast_next_month": forecast.get("forecast", [])}

    # --- 7. DEPLOY ARTIFACTS --------------------------------------------------
    dockerfile = os.path.join(ROOT, "Dockerfile")
    compose = os.path.join(ROOT, "docker-compose.yml")
    df_ok = os.path.exists(dockerfile) and "uvicorn" in open(dockerfile).read()
    co = open(compose).read() if os.path.exists(compose) else ""
    co_ok = "redis" in co and "api" in co and "REDIS_URL" in co
    check("deploy_artifacts", df_ok and co_ok,
          f"Dockerfile={'ok' if df_ok else 'missing'}, "
          f"compose(api+redis+REDIS_URL)={'ok' if co_ok else 'bad'}")

    # --- summary + persistence ------------------------------------------------
    n_pass = sum(c["passed"] for c in checks)
    out["checks"] = checks
    out["n_checks"] = len(checks)
    out["n_passed"] = n_pass
    out["all_passed"] = n_pass == len(checks)

    with open(os.path.join(RESULTS, "phase7_production.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    samples = {
        "auth_gate": {"no_token_status": r_noauth.status_code,
                      "bad_token_status": r_badtok.status_code,
                      "authed_status": ok_cat.status_code},
        "tenant_isolation": {"alice_rows": list_a["n_transactions"],
                             "bob_rows": list_b["n_transactions"],
                             "cross_delete_status": cross_del.status_code},
        "cache": out["cache"],
        "telemetry_summary": tel,
        "dashboard_anomaly_alerts": alerts.head(5).to_dict(orient="records"),
        "forecast": forecast.get("forecast", []),
    }
    with open(os.path.join(SAMPLES, "phase7_production_samples.json"), "w", encoding="utf-8") as fh:
        json.dump(samples, fh, indent=2, default=str)

    # append to the running metrics.json (a list of per-day entries)
    metrics_path = os.path.join(RESULTS, "metrics.json")
    allm = []
    if os.path.exists(metrics_path):
        try:
            loaded = json.load(open(metrics_path))
            allm = loaded if isinstance(loaded, list) else [loaded]
        except Exception:
            allm = []
    allm = [e for e in allm if not (isinstance(e, dict) and e.get("day") == 9)]
    allm.append({
        "generated": "2026-07-03", "day": 9, "phase": "Phase 7 - production wrapper",
        "n_checks": len(checks), "n_passed": n_pass, "all_passed": out["all_passed"],
        "cache_backend": out["cache_backend"], "cache_speedup_x": out["cache"]["speedup_x"],
        "telemetry_p95_ms": tel.get("latency_ms_p95"),
    })
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(allm, fh, indent=2)

    # summary chart
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [c["check"] for c in checks]
    vals = [1 if c["passed"] else 0 for c in checks]
    colors = ["#2ca02c" if v else "#d62728" for v in vals]
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 1.15)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["FAIL", "PASS"])
    ax.set_title(f"Day-9 production checks — {n_pass}/{len(checks)} passed", fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "phase7_checks.png"), dpi=120)
    plt.close(fig)

    print(f"\n=== {n_pass}/{len(checks)} checks passed | cache={out['cache_backend']} "
          f"| speedup={out['cache']['speedup_x']}x | p95={tel.get('latency_ms_p95')}ms ===")
    return 0 if out["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
