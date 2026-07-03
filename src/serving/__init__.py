"""Day-9 Phase-7 production-serving layer for the FinTrack ML API.

Four small, independently testable pieces that turn the Day-5 stateless service
into a multi-tenant production service:

    auth.py       JWT issue/verify + password hashing + get_current_user dependency
    store.py      per-user transaction repository (server-side scoping)
    cache.py      Redis cache with an in-process fallback (repeated-call speedup)
    telemetry.py  per-request telemetry (latency, status, user, cache-hit)

The Day-5 DB fix (every transaction query filtered by user_id) is carried into
the API here: identity comes from the JWT, and no endpoint accepts a caller-
supplied user id, so user A structurally cannot read or mutate user B's data.
"""
