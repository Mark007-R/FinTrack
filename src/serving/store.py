"""Per-user transaction repository — the API-layer analogue of the Day-5 DB fix.

Every read/write is keyed by the authenticated `user_id` that the endpoint gets
from the JWT (never from the request body), so tenant isolation is enforced by
construction: there is no code path by which user A can address user B's rows.

This is an in-memory store (the demo/test substrate); the same interface maps
directly onto the MySQL `transactions` table with a `WHERE user_id = %s` clause,
which is exactly what `app.py:dashboard()` already does for the Flask side.
"""
from __future__ import annotations

import threading
from collections import defaultdict


class TransactionStore:
    def __init__(self) -> None:
        self._by_user: dict[int, list[dict]] = defaultdict(list)
        self._seq_by_user: dict[int, int] = defaultdict(int)
        self._lock = threading.Lock()

    def add(self, user_id: int, txn: dict) -> dict:
        with self._lock:
            self._seq_by_user[user_id] += 1
            row = {"id": self._seq_by_user[user_id], **txn}
            self._by_user[user_id].append(row)
            return row

    def add_many(self, user_id: int, txns: list[dict]) -> int:
        for t in txns:
            self.add(user_id, t)
        return len(txns)

    def list(self, user_id: int) -> list[dict]:
        # a copy, sorted newest-first (mirrors the Flask ORDER BY date DESC)
        rows = list(self._by_user.get(user_id, []))
        rows.sort(key=lambda r: str(r.get("date", "")), reverse=True)
        return rows

    def delete(self, user_id: int, txn_id: int) -> bool:
        with self._lock:
            rows = self._by_user.get(user_id, [])
            for i, r in enumerate(rows):
                if r.get("id") == txn_id:
                    rows.pop(i)
                    return True
            return False

    def count(self, user_id: int) -> int:
        return len(self._by_user.get(user_id, []))

    def clear(self) -> None:
        with self._lock:
            self._by_user.clear()
            self._seq_by_user.clear()


_store = TransactionStore()


def get_txn_store() -> TransactionStore:
    return _store
