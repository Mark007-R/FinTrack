"""Multi-tenancy regression tests (Day-5 DB fix carried to the API store layer).

THE audited bug: `dashboard()` and `invest()` ran against a global `transactions`
table with no `user_id` filter, so every user could see and DELETE every other
user's rows. These tests lock in the fix at the store layer: user A can never
read, delete, or aggregate user B's transactions. (The API-level HTTP regression
lives in test_api.py.)
"""
from __future__ import annotations

import pytest

from src.serving.store import TransactionStore
from src.reco.investments import recommend_for_user


@pytest.fixture
def store():
    s = TransactionStore()
    s.add_many(1, [{"date": "2024-01-01", "merchant": "A-ONLY",
                    "category": "dining", "amount": -10.0}])
    s.add_many(2, [{"date": "2024-01-01", "merchant": "B-ONLY", "category": "rent", "amount": -900.0},
                   {"date": "2024-01-02", "merchant": "B-TWO", "category": "shopping", "amount": -50.0}])
    return s


def test_user_a_sees_only_own_rows(store):
    a_rows = store.list(1)
    assert len(a_rows) == 1
    assert all(r["merchant"] == "A-ONLY" for r in a_rows)
    assert all(r["merchant"] not in {"B-ONLY", "B-TWO"} for r in a_rows)


def test_user_b_sees_only_own_rows(store):
    assert store.count(2) == 2
    assert {r["merchant"] for r in store.list(2)} == {"B-ONLY", "B-TWO"}


def test_user_a_cannot_delete_user_b_row(store):
    # user A tries to delete id=1, but scoped to A that id is A's own single row;
    # B's rows (also id 1 & 2 in B's namespace) are untouched.
    b_before = store.count(2)
    store.delete(1, 1)               # deletes A's row only
    assert store.count(1) == 0
    assert store.count(2) == b_before  # B fully intact


def test_delete_is_scoped_by_user(store):
    # even asking to delete B's id from A's scope must not remove B's data
    assert store.delete(1, 2) is False   # A has no id=2
    assert store.count(2) == 2


def test_counts_are_isolated(store):
    assert store.count(1) == 1
    assert store.count(2) == 2
    assert store.count(999) == 0         # unknown user -> nothing


def test_investment_balance_is_per_user(store):
    # the audited invest() summed ALL users' money; recommend_for_user must only
    # ever see the transactions handed to it (one tenant's rows).
    a = recommend_for_user(store.list(1))
    b = recommend_for_user(store.list(2))
    assert a["total_balance"] == -10.0     # A's own rows only
    assert b["total_balance"] == -950.0    # B's own rows only
    assert a["total_balance"] != b["total_balance"]


def test_add_returns_per_user_sequential_ids(store):
    row = store.add(1, {"date": "2024-02-01", "merchant": "A-TWO",
                        "category": "dining", "amount": -5.0})
    assert row["id"] == 2                  # A's second row -> id 2, independent of B
