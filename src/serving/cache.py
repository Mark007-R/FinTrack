"""Result cache for the deterministic ML endpoints (Day-9).

`/extract` and `/categorize` are pure functions of their input text, so the same
receipt or merchant string re-submitted should not re-run the model. This module
provides a tiny key/value cache with two backends chosen at startup:

  * **Redis** when `REDIS_URL` is set AND the server is reachable (the docker-
    compose deployment). Values are JSON blobs with a TTL.
  * **in-process LRU dict** otherwise, so the service (and the test harness) runs
    identically with no Redis available. `backend` reports which is active.

Keys are namespaced + SHA-1 hashed so arbitrary text is safe as a key and two
endpoints never collide. `stats()` exposes hits/misses for the /metrics endpoint
and the Day-9 speedup measurement.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict
from typing import Any, Optional

DEFAULT_TTL = int(os.getenv("FINTRACK_CACHE_TTL", "3600"))
_MAX_LOCAL = 2048


def _key(namespace: str, raw: str) -> str:
    h = hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()
    return f"fintrack:{namespace}:{h}"


class _LocalLRU:
    """Bounded in-process fallback (TTL ignored — process-lifetime only)."""

    backend = "memory"

    def __init__(self) -> None:
        self._d: "OrderedDict[str, str]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
            return None

    def setex(self, key: str, ttl: int, val: str) -> None:
        with self._lock:
            self._d[key] = val
            self._d.move_to_end(key)
            while len(self._d) > _MAX_LOCAL:
                self._d.popitem(last=False)

    def flushdb(self) -> None:
        with self._lock:
            self._d.clear()


class ResultCache:
    def __init__(self, ttl: int = DEFAULT_TTL) -> None:
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
        self._backend = self._make_backend()
        self.backend = getattr(self._backend, "backend", "redis")

    def _make_backend(self):
        url = os.getenv("REDIS_URL")
        if url:
            try:
                import redis  # type: ignore
                client = redis.Redis.from_url(url, decode_responses=True,
                                              socket_connect_timeout=1)
                client.ping()  # fail fast to the local fallback if unreachable
                return client
            except Exception:
                pass  # no server -> in-process cache; service still runs
        return _LocalLRU()

    def get(self, namespace: str, raw: str) -> Optional[Any]:
        val = self._backend.get(_key(namespace, raw))
        if val is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(val)

    def set(self, namespace: str, raw: str, value: Any) -> None:
        self._backend.setex(_key(namespace, raw), self.ttl,
                            json.dumps(value, default=str))

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"backend": self.backend, "hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0}

    def clear(self) -> None:
        try:
            self._backend.flushdb()
        except Exception:
            pass
        self.hits = 0
        self.misses = 0


_cache: Optional[ResultCache] = None


def get_cache() -> ResultCache:
    global _cache
    if _cache is None:
        _cache = ResultCache()
    return _cache
