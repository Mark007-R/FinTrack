"""FinTrack production ML package (Day-5 Phase-3 integration).

Wires the Day 2-4 bake-off champions into importable, signature-stable components:

    src.extraction.extractor   -- receipt field extraction (rules_smart champion + regex fallback)
    src.categorization.classifier -- expense categorizer (TF-IDF + LinearSVC champion)
    src.anomaly.detector       -- IsolationForest anomaly + recurring/duplicate detection
    src.forecast.cashflow      -- next-month cash-flow forecast (Prophet + seasonal-naive fallback)
    src.reco.investments       -- per-user, risk-profiled investment recommendation

These power both the existing Flask app (via signature-preserving delegation) and the
new async FastAPI service in api.py.
"""
__all__ = ["extraction", "categorization", "anomaly", "forecast", "reco", "schemas"]
