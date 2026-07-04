"""Expense categorizer tests — Day-3 champion (TF-IDF word+char + LinearSVC).

Verifies the trained artifact loads, single/batch prediction shapes, the Day-6
multi-word disambiguation layer, confidence bounds, and graceful behaviour on
unseen merchants.
"""
from __future__ import annotations

import pytest

from src.categorization import get_classifier
from src.categorization.classifier import ExpenseClassifier, KEYWORDS

VALID = set(KEYWORDS)  # the 10 spending categories


@pytest.fixture(scope="module")
def clf():
    return get_classifier()


def test_model_artifact_loads(clf):
    assert clf.available() is True
    assert clf.model_id.startswith("tfidf")


def test_predict_returns_expected_shape(clf):
    out = clf.predict("STARBUCKS COFFEE #123")
    assert set(out) >= {"description", "category", "confidence", "model"}
    assert out["category"] in VALID


def test_known_merchants_map_to_expected_category(clf):
    assert clf.predict("STARBUCKS")["category"] == "dining"
    assert clf.predict("UBER TRIP")["category"] == "transport"
    assert clf.predict("NETFLIX.COM")["category"] == "entertainment"


def test_disambiguation_uber_eats_is_dining_not_transport(clf):
    out = clf.predict("UBER EATS order")
    assert out["category"] == "dining"           # multi-word override beats 'uber'
    assert "disambig" in out["model"]


def test_disambiguation_costco_gas_is_transport(clf):
    assert clf.predict("COSTCO GAS station")["category"] == "transport"


def test_confidence_within_unit_interval(clf):
    out = clf.predict("WHOLE FOODS MARKET")
    assert 0.0 <= out["confidence"] <= 1.0


def test_batch_length_matches_input(clf):
    descs = ["STARBUCKS", "UBER", "NETFLIX", "WHOLE FOODS", "SHELL GAS"]
    out = clf.predict_batch(descs)
    assert len(out) == len(descs)
    assert [o["description"] for o in out] == descs


def test_unseen_merchant_returns_valid_category(clf):
    out = clf.predict("ZZQ UNKNOWN VENDOR 8841")
    assert out["category"] in VALID              # never crashes / never blank


def test_income_detected(clf):
    assert clf.predict("ACME CORP PAYROLL DEPOSIT")["category"] == "income"


def test_keyword_fallback_when_model_missing():
    # a classifier pointed at a nonexistent artifact degrades to keyword rules
    fallback = ExpenseClassifier(model_path="/nonexistent/model.joblib")
    assert fallback.available() is False
    assert fallback.predict("STARBUCKS")["category"] == "dining"
