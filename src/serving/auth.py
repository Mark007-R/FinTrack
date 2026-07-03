"""JWT authentication + password hashing for real multi-tenancy (Day-9).

The audited Flask app had no per-user scoping (Day-5 fixed the SQL; this carries
the same guarantee into the async API). Identity is a signed JWT:

  * `POST /auth/register` / `POST /auth/token` hand out a token for a user.
  * every ML endpoint depends on `get_current_user`, which decodes the token and
    returns the caller's user id.
  * NO endpoint accepts a caller-supplied user id, so the token *is* the scope —
    user A can never address user B's data.

Password hashing uses passlib's pbkdf2_sha256 (pure-python: no native bcrypt
build needed, so the image stays slim and the tests run anywhere). The JWT secret
comes from `FINTRACK_JWT_SECRET`; a dev default is used if unset (a real deploy
sets the env var — noted in the report and README).
"""
from __future__ import annotations

import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
JWT_SECRET = os.getenv("FINTRACK_JWT_SECRET", "dev-secret-change-me-in-prod")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_TTL_MIN = int(os.getenv("FINTRACK_JWT_TTL_MIN", "60"))

_pwd = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token", auto_error=True)


# --------------------------------------------------------------------------- #
# a tiny in-memory user directory (demo). A real deploy swaps this for the MySQL
# `users1` table — the interface (get/create by username) is all the API needs.
# --------------------------------------------------------------------------- #
class UserStore:
    def __init__(self) -> None:
        self._by_name: dict[str, dict] = {}
        self._next_id = 1
        self._lock = threading.Lock()

    def create(self, username: str, password: str) -> dict:
        with self._lock:
            if username in self._by_name:
                raise ValueError("username already registered")
            uid = self._next_id
            self._next_id += 1
            rec = {"user_id": uid, "username": username,
                   "password_hash": _pwd.hash(password)}
            self._by_name[username] = rec
            return rec

    def verify(self, username: str, password: str) -> Optional[dict]:
        rec = self._by_name.get(username)
        if not rec or not _pwd.verify(password, rec["password_hash"]):
            return None
        return rec

    def exists(self, username: str) -> bool:
        return username in self._by_name


_users = UserStore()


def get_user_store() -> UserStore:
    return _users


# --------------------------------------------------------------------------- #
# token issue / verify
# --------------------------------------------------------------------------- #
def create_access_token(user_id: int, username: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "username": username,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=ACCESS_TOKEN_TTL_MIN)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


class CurrentUser:
    __slots__ = ("user_id", "username")

    def __init__(self, user_id: int, username: str):
        self.user_id = user_id
        self.username = username


def get_current_user(token: str = Depends(oauth2_scheme)) -> CurrentUser:
    """FastAPI dependency: decode the Bearer token into the caller's identity.

    Raises 401 on a missing/expired/tampered token — this is the single choke
    point that makes every protected endpoint tenant-scoped.
    """
    cred_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired authentication token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        sub = payload.get("sub")
        if sub is None:
            raise cred_exc
        return CurrentUser(user_id=int(sub), username=payload.get("username", ""))
    except (JWTError, ValueError):
        raise cred_exc
