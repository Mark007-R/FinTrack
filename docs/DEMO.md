# FinTrack — 60-Second Demo

**Run it live (no server, no DB, no API keys):**

```bash
python demo.py
```

It calls the `src/` champions directly and walks the four headline upgrades in ~60s.

## Talk-track (what to say per section)

**0:00 — the hook.** "FinTrack shipped a 'smart bill scanner' that was actually a
one-line regex, and a multi-tenancy bug where every user could see and delete
everyone else's transactions. I turned it into a measured, multi-tenant ML
service. Let me run it."

**0:10 — Receipt extraction.** "The old regex takes the *first* decimal on the
page — here that's the **subtotal 40.00**, not the 42.40 total. The champion
anchors on the TOTAL keyword, handles the GST two-column case, and pulls the
merchant. On 100 SROIE receipts that's amount accuracy **0.15 → 0.83**, merchant
**0 → 0.77** — at **$0**."

**0:25 — Expense categorizer.** "There was *no* categorizer before. This is
TF-IDF + LinearSVC, **macro-F1 0.658 → 0.975**, zero inference cost. Note
'uber eats' → **dining** not transport — a Day-6 disambiguation fix."

**0:40 — Anomaly detection.** "The genuine insight: a global amount threshold
flags your **recurring rent** as fraud and *misses* a real $600 dining charge
that's small in dollars but 24× the dining norm. Context-relative features fix
that — anomaly AP **0.40 → 0.98**."

**0:50 — Multi-tenancy.** "Every read/write is scoped to the authenticated user;
user A physically cannot address user B's rows. That's the security bug closed,
with a regression test."

**0:58 — the honest close.** "And I measured where I *lose*: on novel merchants
the cheap model drops to 0.24 while an LLM holds 1.00 — so I ship the cheap model
as default and route low-confidence rows to an LLM fallback. 66 tests, all green."

## If asked to prove the tests

```bash
pytest tests/ -q          # 66 passed
```

## If asked to see the live API

```bash
uvicorn api:app --port 8000     # then open http://localhost:8000/docs
```
