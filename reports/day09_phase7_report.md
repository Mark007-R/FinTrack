# FinTrack Production Upgrade — Day 09 (Phase 7: Production wrapper)

**Date:** 2026-07-03 · **Day 09 of 10** · Project A (FinTrack)

---

## Resume gap progress

**Gap:** Days 1–8 produced a validated ML service (`api.py`, five endpoints on the
`src/` champions) — but it was single-tenant, stateless, uncached, unobservable,
and undeployable. Any caller could hit any endpoint with no identity, the same
receipt re-ran the model every time, there was no per-request visibility, and
there was no container to ship.

**Today's contribution:** wrapped the Day-5 service into a **deployable,
multi-tenant, cached, observable** production service — JWT auth with per-user
scoping enforced end-to-end (the API-layer analogue of the Day-5 SQL `user_id`
fix), a Redis result cache with an in-process fallback, per-request telemetry, a
Streamlit analytics dashboard that is a *client of the API*, and a Docker/Compose
stack. **12/12 executable production checks pass**, and the image was **built and
run live** — the full `docker compose` stack (api + redis) came up healthy with
`cache_backend=redis`.

---

## Files touched

| File | Change |
|------|--------|
| `src/serving/auth.py` (new) | JWT issue/verify (HS256, python-jose), pbkdf2_sha256 password hashing, in-memory user directory, `get_current_user` FastAPI dependency |
| `src/serving/store.py` (new) | per-user `TransactionStore` — every read/write keyed by the authenticated `user_id`; no cross-tenant addressing exists |
| `src/serving/cache.py` (new) | `ResultCache` — Redis backend when `REDIS_URL` is reachable, bounded in-process LRU otherwise; SHA-1 namespaced keys, hit/miss stats |
| `src/serving/telemetry.py` (new) | ASGI `TelemetryMiddleware` → per-request record (latency/status/user/cache-hit) to a ring + `logs/telemetry.jsonl`; `/metrics` summary |
| `api.py` (rewrite, v0.5.0→0.9.0) | `/auth/register`, `/auth/token`; `/transactions` POST/GET/DELETE (scoped); JWT dependency on all ML endpoints; cache on `/extract`+`/categorize`; `/metrics`; telemetry middleware |
| `src/schemas.py` | +`RegisterRequest`, `TokenResponse`, `AddTransactionsRequest`, `TransactionRow`, `TransactionsResponse`; relaxed `Anomaly/ForecastRequest.transactions` to optional (fall back to stored) |
| `dashboard/data.py` (new) | headless-testable data + matplotlib figure layer (category matrix, balance trend, cash-flow, anomaly table) |
| `dashboard/app.py` (new) | Streamlit dashboard — JWT login → seed → 4 ML-backed views, all via the API |
| `Dockerfile`, `docker-compose.yml`, `.dockerignore` (new) | slim non-root image + api+redis stack with healthchecks |
| `requirements-api.txt` | +python-multipart, +redis (auth/serving deps) |
| `results/run_day9_production.py` (new) | the integration harness below |
| `.gitignore` | +`logs/`, +`mlruns/` |

Production/ML champion code from Days 1–8 was **not** modified — Day 9 wraps it.

---

## Setup

- **Compute:** CPU. In-process FastAPI `TestClient` (runs the real ASGI app +
  middleware + dependencies) for the harness; live Docker Desktop (Linux engine)
  for the container/compose proof.
- **Data:** synthetic, seeded per-user transaction streams (media discipline — no
  real financial data). Two demo tenants (alice/bob).
- **Components exercised:** all five Day 2–4 champions, behind auth + cache +
  telemetry.

---

## Experiments (executable production checks)

Each is a hard assertion in `results/run_day9_production.py`; the run exits
non-zero if any fails (so the daily run cannot log success on a broken service).

### 1 — Auth gate
- **Hypothesis:** ML endpoints reject unauthenticated / tampered requests.
- **Method:** call `/categorize` with no token and with a garbage token.
- **Result:** no-token → **401**, bad-token → **401**, valid-token → **200**.
- **Interpretation:** the JWT dependency is the single choke point that makes
  every protected endpoint tenant-scoped.

### 2 — JWT round-trip
- Register + login issue a working Bearer token; duplicate register → **409**.
  alice=uid 1, bob=uid 2, login → 200.

### 3 — Multi-tenancy (the headline)
| Check | Result |
|-------|--------|
| A reads only A's rows | A sees 196/196 own, B sees 5/5 own — **no cross-leak** |
| A cannot address B's rows | txn ids are **per-user** (both start at 1) → no shared id names B's row; B's id-set **5/5 intact** after A's delete attempts; out-of-scope delete → **404** |

**Interpretation:** identity comes only from the token, never the request body,
so cross-tenant access is structurally impossible — the API-layer form of the
Day-5 `WHERE user_id = %s` fix.

### 4 — Redis cache
| Metric | In-process (harness) | Redis (live compose) |
|--------|---------------------:|---------------------:|
| backend | `memory` (fallback) | **`redis`** |
| repeated `/extract` identical | ✅ | ✅ |
| avg miss (40 iters) | 12.19 ms | — |
| avg hit (40 iters) | 10.18 ms | — |
| speedup | **1.2×** | (hit_rate 0.5 confirmed) |

### 5 — Telemetry
- 95 requests recorded; **p50 3.2 ms / p95 22.7 ms**, 40 cache-hits tracked,
  0 errors; per-path breakdown available at `/metrics`.

### 6 — Dashboard data layer
- From live API output: category heat-map matrix **(8 months × 8 categories)**,
  balance-trend **195 rows**, **10 anomaly alerts** — all four figures render
  headless (`results/day9_dashboard_*.png`).

### 7 — Deploy artifacts (proven live, not just present)
- `docker build` → image built; `docker run` → `/health` **200** in ~2 s,
  auth gate 401, authed categorize `NETFLIX→entertainment`.
- `docker compose up` → **api + redis both healthy**, `/health` reports
  `cache_backend=redis`, `/metrics` shows a real Redis cache hit
  (`hits:1, misses:1, hit_rate:0.5`). Evidence:
  `results/samples/day9_docker_compose_evidence.txt`.

---

## Head-to-Head / running leaderboard (component state entering Day 10)

| Component | Old production (pre-sprint) | Shipped state (Day 9) |
|-----------|-----------------------------|-----------------------|
| Bill extraction | regex, amount 0.15, no merchant | rules_smart + LLM-fallback routing, amount 0.83 |
| Expense categorization | did not exist | TF-IDF+LinearSVC joblib, in-dist F1 1.00 / hard-case 0.89 |
| Anomaly / recurring / forecast | did not exist | IsoForest AP 0.98 / recurring F1 0.99 / Prophet 15.8% MAPE |
| Investment reco | cross-user aggregate (bug) | per-user risk-profiled |
| **Multi-tenancy** | **A sees B (bug)** | **JWT-scoped; A cannot address B — proven** |
| **Serving** | **Flask only, no ML API** | **async FastAPI + Redis cache + telemetry, Dockerized** |

---

## Key findings

1. **A cache is only as valuable as the op it skips.** In-process, the rules
   extractor is ~12 ms, so caching buys just **1.2×** — an honest, unimpressive
   number. The cache earns its keep on the **Day-8 LLM-fallback path** (novel
   receipts route to an LLM at ~1.8 s + API cost): there a hit turns a paid
   1,800 ms call into a free sub-ms lookup. And **Redis beats the in-process
   cache not on latency but on *sharing*** — it's the only backend that survives
   a restart and is shared across replicas/workers, which is exactly what a
   multi-process uvicorn deployment needs.
2. **Per-user id namespaces make cross-tenant access unnameable.** Because each
   tenant's transaction ids start at 1, there is no global id A could submit to
   reach B's row — isolation is structural, not a runtime check that could be
   forgotten. The regression assertion proves B's id-set is untouched by A.
3. **Telemetry surfaced a cold-start tail immediately:** p50 3 ms but
   `latency_ms_max` 1,434 ms on the compose run — the first request pays model +
   import warm-up. That single number is the argument for a warm-up ping in the
   container start-period (already in the healthcheck).

## What didn't work (and why)

- **Standalone `docker run` reports `cache_backend=memory`**, not redis — correct
  behaviour: no `REDIS_URL` is set outside compose, so the service transparently
  falls back rather than crashing. The Redis path is only exercised (and is
  proven) under `docker compose`, where `REDIS_URL=redis://redis:6379/0` is
  injected. This graceful degradation is the design, not a gap.
- **The Docker Desktop Linux engine dropped mid-session** after the first build
  attempt; it was restarted and the build/run/compose proof completed live. Noted
  for reproducibility.

---

## Sample outputs saved

- `results/phase7_production.json` — all 12 checks + cache/telemetry summary
- `results/phase7_checks.png` — pass/fail chart (12/12)
- `results/day9_dashboard_{heatmap,balance,cashflow}.png` — dashboard figures
- `results/samples/phase7_production_samples.json` — auth/isolation/cache/forecast samples
- `results/samples/day9_docker_compose_evidence.txt` — live compose `/health` + `/metrics`
- `results/metrics.json` — appended `day 9 / Phase 7` entry

---

## Next day

**Day 10 (Phase 8 — PROJECT COMPLETE):** 40+ pytest tests
(`test_extraction`, `test_categorization`, `test_anomaly`, `test_forecast`,
`test_multitenancy` [regression: A cannot see/delete B], `test_auth`, `test_api`
— golden path + edge cases), README rewrite with results tables + architecture
diagram, `docs/MODEL_CARD.md` for the categorizer, and the 60-second demo.
Today's `run_day9_production.py` checks become the seed for `test_auth.py` +
`test_api.py` + `test_multitenancy.py`.

---

## Code changes

New `src/serving/` package (auth, store, cache, telemetry); `api.py` rewritten to
v0.9.0 (auth + scoping + cache + telemetry + `/metrics`); `dashboard/` Streamlit
client + headless data layer; `Dockerfile` + `docker-compose.yml` + `.dockerignore`;
`src/schemas.py` extended; `requirements-api.txt` + `.gitignore` updated. No Day
1–8 ML champion code modified.
