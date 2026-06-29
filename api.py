"""FinTrack ML API — async FastAPI service (Day-5 Phase-3).

Separate from the Flask app (which stays on :5000). This service exposes the
five Day 2-4 champion components behind Pydantic-validated JSON endpoints:

    POST /extract     receipt text  -> {amount, date, merchant}      (rules_smart champion)
    POST /categorize  description   -> {category, confidence}        (TF-IDF + LinearSVC champion)
    POST /anomaly     transactions  -> anomaly flags + recurring + duplicates (IsolationForest)
    POST /forecast    transactions  -> next-month cash-flow forecast (Prophet + fallback)
    POST /recommend   transactions  -> risk-profiled investment options (per-user)

Run:  uvicorn api:app --port 8000     (or python api.py)
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI

from src.schemas import (
    ExtractTextRequest, ExtractionResult,
    CategorizeRequest, CategorizeResult,
    AnomalyRequest, AnomalyResult,
    ForecastRequest, ForecastResult,
    RecommendRequest, RecommendResult,
)
from src.extraction import extract_fields
from src.categorization import get_classifier
from src.anomaly import detect_anomalies, find_recurring_groups, find_duplicate_charges
from src.forecast import forecast_cashflow
from src.reco.investments import recommend_for_user

app = FastAPI(
    title="FinTrack ML API",
    version="0.5.0",
    description="Receipt extraction, expense categorization, anomaly detection, "
                "cash-flow forecasting, and per-user investment recommendation.",
)


@app.get("/health")
async def health():
    clf = get_classifier()
    return {"status": "ok", "categorizer_loaded": clf.available(),
            "categorizer_model": clf.model_id}


@app.post("/extract", response_model=ExtractionResult)
async def extract(req: ExtractTextRequest):
    return extract_fields(req.text)


@app.post("/categorize", response_model=CategorizeResult)
async def categorize(req: CategorizeRequest):
    return get_classifier().predict(req.description)


@app.post("/anomaly", response_model=AnomalyResult)
async def anomaly(req: AnomalyRequest):
    txns = [t.model_dump() for t in req.transactions]
    res = detect_anomalies(txns, top_k=req.top_k)
    res["recurring_groups"] = find_recurring_groups(txns)
    res["duplicate_charges"] = find_duplicate_charges(txns)
    return res


@app.post("/forecast", response_model=ForecastResult)
async def forecast(req: ForecastRequest):
    txns = [t.model_dump() for t in req.transactions]
    return forecast_cashflow(txns, horizon_months=req.horizon_months)


@app.post("/recommend", response_model=RecommendResult)
async def recommend(req: RecommendRequest):
    txns = [t.model_dump() for t in req.transactions]
    return recommend_for_user(txns, total_balance=req.total_balance)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
