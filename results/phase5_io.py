"""Shared writer for the Day-7 combined robustness CSV.

The three Day-7 tracks (ocr / active_learning / rag) produce different metrics, so
`results/phase5_robustness.csv` is kept in a tidy LONG format:

    track, variant, metric, value, note

`upsert_track(track, records)` replaces all rows for `track` and re-appends, so a
track script is safe to re-run without duplicating rows.
"""
from __future__ import annotations

import csv
import os

RESULTS = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(RESULTS, "phase5_robustness.csv")
FIELDS = ["track", "variant", "metric", "value", "note"]


def upsert_track(track: str, records: list[dict]) -> str:
    """records: list of {variant, metric, value, note?}. Replaces this track's rows."""
    existing = []
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            existing = [r for r in csv.DictReader(f) if r.get("track") != track]
    for r in records:
        r.setdefault("note", "")
        r["track"] = track
    rows = existing + [{k: r.get(k, "") for k in FIELDS} for r in records]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return CSV_PATH
