# AI-Personal-Finance-Manager

**🔗 Live demo:** https://iambatman07-ai-personal-finance-manager.hf.space

A multi-tenant personal-finance service: receipt extraction, a trained expense categorizer, anomaly and recurring-charge detection, cash-flow forecasting, and per-user risk-profiled investment recommendations — behind a JWT-scoped FastAPI service, with a Flask UI.

> Data discipline: every experiment uses **public** (SROIE receipts) or **synthetic** transaction data. No real personal financial data is used anywhere.

---

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate      # (Windows) or source .venv/bin/activate
pip install -r requirements-api.txt                 # API/service stack
python -m src.categorization.train                  # build models/expense_classifier.joblib

uvicorn api:app --port 8000                          # ML API  → http://localhost:8000/docs
# or the full stack:
docker compose up                                    # FastAPI + Redis
```

Research stack (the experiment scripts under `results/`):

```bash
pip install -r requirements-experiments.txt
python results/run_day2_extraction.py               # extraction bake-off
python results/run_day3_categorize.py               # categorizer bake-off
python results/run_day4_anomaly_forecast.py         # anomaly + forecast
```

---

## Tests

```bash
pytest tests/ -q
```

Covers extraction, categorization, anomaly detection, forecasting, authentication, per-user tenant isolation, and the HTTP API.

---

## License

MIT — see [LICENSE](LICENSE).
