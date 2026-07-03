"""FinTrack ML API — async FastAPI service.

Day-5 stood up the five ML endpoints; **Day-9 (Phase 7) production-wraps them**:

  * **JWT multi-tenancy** — `/auth/register` + `/auth/token` hand out a signed
    token; every ML endpoint depends on it and is scoped to the caller. No
    endpoint accepts a caller-supplied user id, so user A can never read/mutate
    user B's data (the API-layer analogue of the Day-5 SQL `user_id` fix).
  * **Per-user transaction store** — `/transactions` (POST/GET/DELETE); the
    analytics endpoints run on the caller's stored transactions when the body is
    omitted.
  * **Redis cache** (in-process fallback) on the deterministic `/extract` and
    `/categorize` endpoints — repeated inputs skip the model.
  * **Per-request telemetry** — latency / status / user / cache-hit to a ring +
    `logs/telemetry.jsonl`, summarised at `/metrics`.

Endpoint map (all ML endpoints require `Authorization: Bearer <token>`):

    POST /auth/register  {username,password}      -> token
    POST /auth/token     (OAuth2 form)            -> token
    POST /transactions   {transactions}           -> stored count
    GET  /transactions                            -> the caller's rows only
    POST /extract        {text}                   -> {amount,date,merchant}   (cached)
    POST /categorize     {description}            -> {category,confidence}    (cached)
    POST /anomaly        [{transactions}|stored]  -> flags + recurring + duplicates
    POST /forecast       [{transactions}|stored]  -> next-month cash-flow
    POST /recommend      [{transactions}|stored]  -> risk-profiled options
    GET  /metrics                                 -> telemetry + cache summary

Run:  uvicorn api:app --port 8000     (or python api.py)
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm

from src.schemas import (
    ExtractTextRequest, ExtractionResult,
    CategorizeRequest, CategorizeResult,
    AnomalyRequest, AnomalyResult,
    ForecastRequest, ForecastResult,
    RecommendRequest, RecommendResult,
    RegisterRequest, TokenResponse,
    AddTransactionsRequest, TransactionsResponse,
)
from src.extraction import extract_fields
from src.categorization import get_classifier
from src.anomaly import detect_anomalies, find_recurring_groups, find_duplicate_charges
from src.forecast import forecast_cashflow
from src.reco.investments import recommend_for_user

from src.serving.auth import (
    CurrentUser, create_access_token, get_current_user, get_user_store,
)
from src.serving.store import get_txn_store
from src.serving.cache import get_cache
from src.serving.telemetry import TelemetryMiddleware, get_sink

app = FastAPI(
    title="FinTrack ML API",
    version="0.9.0",
    description="Multi-tenant receipt extraction, expense categorization, anomaly "
                "detection, cash-flow forecasting, and per-user investment "
                "recommendation — JWT-scoped, cached, and telemetered.",
)
app.add_middleware(TelemetryMiddleware)


# --------------------------------------------------------------------------- #
# health / metrics
# --------------------------------------------------------------------------- #
@app.get("/health")
async def health():
    clf = get_classifier()
    return {"status": "ok", "version": app.version,
            "categorizer_loaded": clf.available(),
            "categorizer_model": clf.model_id,
            "cache_backend": get_cache().backend}


@app.get("/metrics")
async def metrics():
    return {"telemetry": get_sink().summary(), "cache": get_cache().stats()}


# --------------------------------------------------------------------------- #
# auth
# --------------------------------------------------------------------------- #
@app.post("/auth/register", response_model=TokenResponse)
async def register(req: RegisterRequest):
    store = get_user_store()
    try:
        rec = store.create(req.username, req.password)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="Username already registered.")
    token = create_access_token(rec["user_id"], rec["username"])
    return TokenResponse(access_token=token, user_id=rec["user_id"], username=rec["username"])


@app.post("/auth/token", response_model=TokenResponse)
async def token(form: OAuth2PasswordRequestForm = Depends()):
    rec = get_user_store().verify(form.username, form.password)
    if not rec:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Incorrect username or password.",
                            headers={"WWW-Authenticate": "Bearer"})
    tok = create_access_token(rec["user_id"], rec["username"])
    return TokenResponse(access_token=tok, user_id=rec["user_id"], username=rec["username"])


# --------------------------------------------------------------------------- #
# per-user transactions (server-side scoping — the Day-5 fix, at the API layer)
# --------------------------------------------------------------------------- #
@app.post("/transactions", response_model=TransactionsResponse)
async def add_transactions(req: AddTransactionsRequest, request: Request,
                           user: CurrentUser = Depends(get_current_user)):
    request.state.user_id = user.user_id
    store = get_txn_store()
    store.add_many(user.user_id, [t.model_dump() for t in req.transactions])
    rows = store.list(user.user_id)
    return TransactionsResponse(user_id=user.user_id, n_transactions=len(rows),
                                transactions=rows)


@app.get("/transactions", response_model=TransactionsResponse)
async def list_transactions(request: Request,
                            user: CurrentUser = Depends(get_current_user)):
    request.state.user_id = user.user_id
    rows = get_txn_store().list(user.user_id)   # ONLY this user's rows
    return TransactionsResponse(user_id=user.user_id, n_transactions=len(rows),
                                transactions=rows)


@app.delete("/transactions/{txn_id}")
async def delete_transaction(txn_id: int, request: Request,
                             user: CurrentUser = Depends(get_current_user)):
    request.state.user_id = user.user_id
    ok = get_txn_store().delete(user.user_id, txn_id)   # scoped delete
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Transaction not found for this user.")
    return {"deleted": txn_id, "user_id": user.user_id}


# --------------------------------------------------------------------------- #
# ML endpoints (JWT-protected; deterministic ones are cached)
# --------------------------------------------------------------------------- #
@app.post("/extract", response_model=ExtractionResult)
async def extract(req: ExtractTextRequest, request: Request,
                  user: CurrentUser = Depends(get_current_user)):
    request.state.user_id = user.user_id
    cache = get_cache()
    cached = cache.get("extract", req.text)
    if cached is not None:
        request.state.cache_hit = True
        return cached
    result = extract_fields(req.text)
    cache.set("extract", req.text, result)
    return result


@app.post("/categorize", response_model=CategorizeResult)
async def categorize(req: CategorizeRequest, request: Request,
                     user: CurrentUser = Depends(get_current_user)):
    request.state.user_id = user.user_id
    cache = get_cache()
    cached = cache.get("categorize", req.description)
    if cached is not None:
        request.state.cache_hit = True
        return cached
    result = get_classifier().predict(req.description)
    cache.set("categorize", req.description, result)
    return result


def _resolve_txns(body_txns, user: CurrentUser) -> list[dict]:
    """Body transactions if supplied, else the caller's stored transactions.

    Either way the data is scoped to the authenticated user — there is no path
    to another tenant's rows.
    """
    if body_txns:
        return [t.model_dump() for t in body_txns]
    return get_txn_store().list(user.user_id)


@app.post("/anomaly", response_model=AnomalyResult)
async def anomaly(req: AnomalyRequest, request: Request,
                  user: CurrentUser = Depends(get_current_user)):
    request.state.user_id = user.user_id
    txns = _resolve_txns(req.transactions, user)
    res = detect_anomalies(txns, top_k=req.top_k)
    res["recurring_groups"] = find_recurring_groups(txns)
    res["duplicate_charges"] = find_duplicate_charges(txns)
    return res


@app.post("/forecast", response_model=ForecastResult)
async def forecast(req: ForecastRequest, request: Request,
                   user: CurrentUser = Depends(get_current_user)):
    request.state.user_id = user.user_id
    txns = _resolve_txns(req.transactions, user)
    return forecast_cashflow(txns, horizon_months=req.horizon_months)


@app.post("/recommend", response_model=RecommendResult)
async def recommend(req: RecommendRequest, request: Request,
                    user: CurrentUser = Depends(get_current_user)):
    request.state.user_id = user.user_id
    txns = _resolve_txns(req.transactions, user)
    return recommend_for_user(txns, total_balance=req.total_balance)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
