"""Per-request telemetry (Day-9).

An ASGI middleware records one structured record per request — method, path,
status, latency, the authenticated user (if any), and whether a cache hit served
it — to both a bounded in-memory ring (for the `/metrics` endpoint) and an
append-only JSONL file (`logs/telemetry.jsonl`, git-ignored) for offline analysis.

Endpoints annotate the request with cache-hit / user via `request.state` so the
middleware can pick them up without each handler re-logging.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(ROOT, "logs")
LOG_PATH = os.path.join(LOG_DIR, "telemetry.jsonl")
_RING_MAX = 5000


class TelemetrySink:
    def __init__(self, path: str = LOG_PATH) -> None:
        self.path = path
        self.ring: "deque[dict]" = deque(maxlen=_RING_MAX)
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def record(self, rec: dict) -> None:
        with self._lock:
            self.ring.append(rec)
            try:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
            except Exception:
                pass  # telemetry must never break the request path

    def summary(self) -> dict:
        with self._lock:
            recs = list(self.ring)
        if not recs:
            return {"n_requests": 0}
        lat = sorted(r["latency_ms"] for r in recs)
        n = len(lat)
        by_path: dict[str, int] = {}
        errors = 0
        cache_hits = 0
        for r in recs:
            by_path[r["path"]] = by_path.get(r["path"], 0) + 1
            if r["status"] >= 400:
                errors += 1
            if r.get("cache_hit"):
                cache_hits += 1
        return {
            "n_requests": n,
            "errors": errors,
            "cache_hits": cache_hits,
            "latency_ms_p50": lat[int(0.50 * (n - 1))],
            "latency_ms_p95": lat[int(0.95 * (n - 1))],
            "latency_ms_max": lat[-1],
            "requests_by_path": by_path,
        }

    def clear(self) -> None:
        with self._lock:
            self.ring.clear()


_sink: Optional[TelemetrySink] = None


def get_sink() -> TelemetrySink:
    global _sink
    if _sink is None:
        _sink = TelemetrySink()
    return _sink


class TelemetryMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        request.state.cache_hit = False
        request.state.user_id = None
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - start) * 1000, 3)
        response.headers["X-Process-Time-Ms"] = str(latency_ms)
        get_sink().record({
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": latency_ms,
            "user_id": getattr(request.state, "user_id", None),
            "cache_hit": bool(getattr(request.state, "cache_hit", False)),
        })
        return response
