"""JWT auth + password-hashing tests (Day-9 multi-tenancy foundation).

Verifies password hashing (never plaintext), credential verification, duplicate
registration, JWT round-trip, and that a tampered/garbage token is rejected by
the `get_current_user` dependency with a 401.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from src.serving.auth import (
    UserStore, create_access_token, get_current_user,
)


@pytest.fixture
def users():
    return UserStore()


def test_password_is_hashed_not_plaintext(users):
    rec = users.create("alice", "s3cret-pw")
    assert rec["password_hash"] != "s3cret-pw"
    assert rec["password_hash"].startswith("$pbkdf2-sha256$")


def test_verify_correct_password(users):
    users.create("bob", "hunter2pw")
    assert users.verify("bob", "hunter2pw") is not None


def test_verify_wrong_password_returns_none(users):
    users.create("carol", "correct-pw")
    assert users.verify("carol", "wrong-pw") is None


def test_verify_unknown_user_returns_none(users):
    assert users.verify("nobody", "whatever") is None


def test_duplicate_registration_raises(users):
    users.create("dave", "pw123456")
    with pytest.raises(ValueError):
        users.create("dave", "another-pw")


def test_ids_are_sequential(users):
    a = users.create("u1", "pw123456")
    b = users.create("u2", "pw123456")
    assert b["user_id"] == a["user_id"] + 1


def test_token_roundtrip_yields_user_identity():
    tok = create_access_token(user_id=42, username="erin")
    current = get_current_user(token=tok)
    assert current.user_id == 42
    assert current.username == "erin"


def test_tampered_token_rejected():
    tok = create_access_token(user_id=7, username="mallory")
    with pytest.raises(HTTPException) as exc:
        get_current_user(token=tok + "tampered")
    assert exc.value.status_code == 401


def test_garbage_token_rejected():
    with pytest.raises(HTTPException) as exc:
        get_current_user(token="not-a-jwt")
    assert exc.value.status_code == 401
