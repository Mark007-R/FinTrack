"""FastAPI service tests (Day-5 endpoints + Day-9 production wrap).

Golden paths and edge cases for the async ML API:
  * health / auth / protected-endpoint gating
  * extract + categorize golden paths (+ empty-PDF, non-receipt, foreign-currency,
    unseen-merchant edge cases)
  * anomaly / forecast / recommend on the caller's stored transactions
  * HTTP-level multi-tenancy regression: user B cannot read or delete user A's rows
"""
from __future__ import annotations

import itertools

import pytest
from fastapi.testclient import TestClient

import api
from src.serving.store import get_txn_store

_counter = itertools.count(1)


@pytest.fixture
def client():
    get_txn_store().clear()
    return TestClient(api.app)


def _register(client) -> tuple[str, dict]:
    """Register a uniquely-named user and return (username, auth-header)."""
    name = f"user_{next(_counter)}"
    r = client.post("/auth/register", json={"username": name, "password": "pw123456"})
    assert r.status_code == 200, r.text
    tok = r.json()["access_token"]
    return name, {"Authorization": f"Bearer {tok}"}


# --------------------------------------------------------------------------- #
# health + auth gating
# --------------------------------------------------------------------------- #
def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["categorizer_loaded"] is True


def test_protected_endpoint_requires_token(client):
    r = client.post("/categorize", json={"description": "STARBUCKS"})
    assert r.status_code == 401


def test_register_then_duplicate_conflict(client):
    name, _ = _register(client)
    r = client.post("/auth/register", json={"username": name, "password": "pw123456"})
    assert r.status_code == 409


def test_bad_token_rejected(client):
    r = client.post("/categorize", json={"description": "X"},
                    headers={"Authorization": "Bearer garbage"})
    assert r.status_code == 401


# --------------------------------------------------------------------------- #
# categorize + extract golden paths and edge cases
# --------------------------------------------------------------------------- #
def test_categorize_golden_path(client):
    _, hdr = _register(client)
    r = client.post("/categorize", json={"description": "STARBUCKS COFFEE"}, headers=hdr)
    assert r.status_code == 200
    assert r.json()["category"] == "dining"


def test_categorize_unseen_merchant_returns_valid(client):
    _, hdr = _register(client)
    r = client.post("/categorize", json={"description": "ZQX VENDOR 9910"}, headers=hdr)
    assert r.status_code == 200
    assert r.json()["category"]           # non-empty, valid label


def test_extract_golden_path(client):
    _, hdr = _register(client)
    r = client.post("/extract", json={"text": "SHOP\nTotal 42.40\nDate 01/02/2024\n"}, headers=hdr)
    assert r.status_code == 200
    assert r.json()["amount"] == 42.40


def test_extract_empty_pdf_edge(client):
    _, hdr = _register(client)
    r = client.post("/extract", json={"text": ""}, headers=hdr)
    assert r.status_code == 200
    assert r.json()["amount"] == 0.0
    assert r.json()["method"] == "empty"


def test_extract_non_receipt_edge(client):
    _, hdr = _register(client)
    r = client.post("/extract", json={"text": "Thank you for your enquiry."}, headers=hdr)
    assert r.status_code == 200               # must not 500 on a non-receipt


def test_extract_foreign_currency_edge(client):
    _, hdr = _register(client)
    r = client.post("/extract",
                    json={"text": "LADEN\nDatum 11.03.2024\nGesamtbetrag 19.90\n"}, headers=hdr)
    assert r.status_code == 200
    assert r.json()["amount"] == 19.90
    assert r.json()["date"] == "2024-03-11"


def test_categorize_cache_hit_second_call(client):
    _, hdr = _register(client)
    body = {"description": "NETFLIX MONTHLY"}
    client.post("/categorize", json=body, headers=hdr)
    client.post("/categorize", json=body, headers=hdr)
    stats = client.get("/metrics").json()["cache"]
    assert stats.get("hits", 0) >= 1


# --------------------------------------------------------------------------- #
# stored-transaction analytics
# --------------------------------------------------------------------------- #
def test_transactions_roundtrip_and_recommend(client, user_a_txns):
    _, hdr = _register(client)
    r = client.post("/transactions", json={"transactions": user_a_txns}, headers=hdr)
    assert r.status_code == 200
    assert r.json()["n_transactions"] == len(user_a_txns)
    rec = client.post("/recommend", json={}, headers=hdr)
    assert rec.status_code == 200
    assert rec.json()["risk_profile"] in {"conservative", "balanced", "aggressive"}
    assert len(rec.json()["options"]) >= 1


# --------------------------------------------------------------------------- #
# HTTP-level multi-tenancy regression (the audited bug)
# --------------------------------------------------------------------------- #
def test_user_b_cannot_see_user_a_transactions(client, user_a_txns):
    _, hdr_a = _register(client)
    _, hdr_b = _register(client)
    client.post("/transactions", json={"transactions": user_a_txns}, headers=hdr_a)
    # B lists -> must see zero of A's rows
    rb = client.get("/transactions", headers=hdr_b)
    assert rb.status_code == 200
    assert rb.json()["n_transactions"] == 0


def test_user_b_cannot_delete_user_a_transaction(client, user_a_txns):
    _, hdr_a = _register(client)
    _, hdr_b = _register(client)
    client.post("/transactions", json={"transactions": user_a_txns}, headers=hdr_a)
    a_rows = client.get("/transactions", headers=hdr_a).json()["transactions"]
    victim_id = a_rows[0]["id"]
    # B attempts to delete A's row id -> 404 (not found in B's scope)
    rd = client.delete(f"/transactions/{victim_id}", headers=hdr_b)
    assert rd.status_code == 404
    # A's data is fully intact
    assert client.get("/transactions", headers=hdr_a).json()["n_transactions"] == len(user_a_txns)
