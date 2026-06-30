"""Expense categorizer — Day-3 Phase-2b champion (TF-IDF word+char + LinearSVC).

Day-3 bake-off (test split, 10 classes):

    keyword (old)              macro-F1 0.658
    TF-IDF(word) + LightGBM    0.116   (the documented trap)
    TF-IDF(word+char)+LightGBM 0.850
    SBERT + LightGBM           0.939
    TF-IDF(word+char)+LinearSVC 0.975  <-- champion: 0.08s fit, $0 inference
    DistilBERT fine-tune       0.994   (ceiling, 72s train)

LinearSVC has no predict_proba, so confidence is the normalised decision margin
(softmax over class scores). The model artifact lives at
`models/expense_classifier.joblib` (git-ignored); train it with
`python -m src.categorization.train`.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(ROOT, "models", "expense_classifier.joblib")

# Fallback keyword rules (the Day-1 baseline) so the component degrades gracefully
# if the trained artifact is missing (e.g. a fresh clone before `train`).
KEYWORDS = {
    "groceries": ["grocer", "market", "mkt", "food", "walmart", "costco", "aldi", "kroger", "safeway"],
    "dining": ["dining", "restaurant", "cafe", "pizza", "coffee", "starbucks", "mcdonald", "eats", "grubhub", "doordash"],
    "transport": ["transport", "uber", "lyft", "gas", "fuel", "oil", "parking", "air", "taxi", "metro", "subway"],
    "utilities": ["utilit", "electric", "energy", "water", "gas co", "comcast", "verizon", "at&t", "mobile", "internet"],
    "rent": ["rent", "lease", "apartment", "apt", "property", "landlord", "residential"],
    "entertainment": ["entertain", "netflix", "spotify", "hulu", "cinema", "theatre", "games", "xbox", "disney", "music"],
    "health": ["health", "pharmacy", "medical", "clinic", "dental", "dr ", "lab", "rx", "diagnostic"],
    "shopping": ["shop", "amazon", "target", "store", "best buy", "ikea", "nike", "macy", "ebay", "apple"],
    "income": ["payroll", "deposit", "salary", "refund", "interest", "dividend", "payout", "income"],
    "other": [],
}


def _keyword_predict(desc: str) -> str:
    d = desc.lower()
    for cat, kws in KEYWORDS.items():
        for kw in kws:
            if kw in d:
                return cat
    return "other"


# Day-6 fix: high-precision multi-word disambiguation for cross-category merchant
# strings where a single brand token would mislead the model (e.g. "uber eats" is
# dining, not transport). Error analysis on the Day-6 stress set found
# multi_category_overlap was a distinct failure mode; this layer lifted overlap-row
# accuracy 0.375 -> 0.75 with ZERO regression on the in-distribution set (the rules
# are multi-word, so they cannot fire on ordinary single-brand rows).
DISAMBIG = [
    ("uber eats", "dining"),
    ("amazon fresh", "groceries"),
    ("amazon grocery", "groceries"),
    ("apple music", "entertainment"),
    ("costco gas", "transport"),
    ("walmart pharmacy", "health"),
]


def _disambig(desc: str):
    d = desc.lower()
    for phrase, cat in DISAMBIG:
        if phrase in d:
            return cat
    return None


class ExpenseClassifier:
    """Loads the trained pipeline; falls back to keyword rules if absent."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.pipeline = None
        self.classes_: Optional[list[str]] = None
        self.model_id = "keyword_fallback"
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.model_path):
            import joblib
            blob = joblib.load(self.model_path)
            self.pipeline = blob["pipeline"]
            self.classes_ = list(blob["classes"])
            self.model_id = blob.get("model_id", "tfidf_linsvc")

    def available(self) -> bool:
        return self.pipeline is not None

    def predict(self, description: str) -> dict:
        return self.predict_batch([description])[0]

    def predict_batch(self, descriptions: list[str]) -> list[dict]:
        if not self.available():
            out = []
            for d in descriptions:
                ov = _disambig(d)
                out.append({"description": d, "category": ov or _keyword_predict(d),
                            "confidence": 0.9 if ov else 0.4, "model": self.model_id})
            return out
        # LinearSVC -> use decision_function margins, softmax-normalised for a [0,1] score
        scores = self.pipeline.decision_function(descriptions)
        scores = np.atleast_2d(scores)
        preds = self.pipeline.predict(descriptions)
        out = []
        for d, row, p in zip(descriptions, scores, preds):
            ex = np.exp(row - np.max(row))
            soft = ex / ex.sum()
            ov = _disambig(d)  # Day-6 high-precision override (multi-word phrases)
            out.append({"description": d, "category": ov or str(p),
                        "confidence": 0.95 if ov else round(float(soft.max()), 4),
                        "model": self.model_id + ("+disambig" if ov else "")})
        return out


@lru_cache(maxsize=1)
def get_classifier() -> ExpenseClassifier:
    """Process-wide singleton (model loads once)."""
    return ExpenseClassifier()
