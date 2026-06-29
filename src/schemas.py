"""Pydantic v2 schemas shared by the FastAPI service and the src/ components.

Every API request/response is validated against one of these models, so the
contract is explicit and self-documenting (FastAPI renders them in /docs).
"""
from __future__ import annotations

from datetime import date as _date
from typing import Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
class ExtractionResult(BaseModel):
    """Field-level result of receipt/bill extraction."""
    amount: float = Field(..., description="Bill total as a positive magnitude (USD).")
    date: Optional[str] = Field(None, description="ISO-8601 (YYYY-MM-DD) bill date, or null.")
    merchant: Optional[str] = Field(None, description="Merchant/company name, or null.")
    method: str = Field(..., description="Which extractor produced the result (rules_smart|regex_fallback).")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Heuristic confidence in [0,1].")


class ExtractTextRequest(BaseModel):
    text: str = Field(..., description="OCR/pdf-extracted receipt text.")


# --------------------------------------------------------------------------- #
# categorization
# --------------------------------------------------------------------------- #
class CategorizeRequest(BaseModel):
    description: str = Field(..., min_length=1, description="Transaction description / merchant string.")


class CategorizeResult(BaseModel):
    description: str
    category: str = Field(..., description="Predicted spending category.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    model: str = Field(..., description="Model id that produced the label.")


# --------------------------------------------------------------------------- #
# anomaly
# --------------------------------------------------------------------------- #
class Txn(BaseModel):
    date: str = Field(..., description="ISO date of the transaction.")
    merchant: str = ""
    category: str = "other"
    amount: float = Field(..., description="Signed amount; negative = outflow/spend.")


class AnomalyRequest(BaseModel):
    transactions: list[Txn] = Field(..., min_length=1)
    top_k: Optional[int] = Field(None, description="If set, only return the top-k most anomalous.")


class AnomalyFlag(BaseModel):
    date: str
    merchant: str
    category: str
    amount: float
    score: float = Field(..., description="Anomaly score (higher = more anomalous).")
    is_anomaly: bool
    reason: str


class AnomalyResult(BaseModel):
    n_transactions: int
    n_flagged: int
    flags: list[AnomalyFlag]
    recurring_groups: list[dict]
    duplicate_charges: list[dict]


# --------------------------------------------------------------------------- #
# forecast
# --------------------------------------------------------------------------- #
class ForecastRequest(BaseModel):
    transactions: list[Txn] = Field(..., min_length=1)
    horizon_months: int = Field(1, ge=1, le=12)


class ForecastResult(BaseModel):
    method: str
    history_months: int
    forecast: list[dict] = Field(..., description="List of {month, predicted_spend}.")
    last_actual: Optional[float] = None


# --------------------------------------------------------------------------- #
# recommendation
# --------------------------------------------------------------------------- #
class RecommendRequest(BaseModel):
    transactions: list[Txn] = Field(default_factory=list,
                                     description="The user's own transactions (user-scoped).")
    total_balance: Optional[float] = Field(None, description="Override balance instead of deriving it.")


class InvestmentOption(BaseModel):
    name: str
    type: str
    min_investment: float
    expected_return_pct: float
    risk: str
    suitability: float = Field(..., ge=0.0, le=1.0, description="Per-user fit score in [0,1].")


class RecommendResult(BaseModel):
    total_balance: float
    risk_profile: str
    risk_score: float
    options: list[InvestmentOption]
