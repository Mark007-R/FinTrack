# FinTrack ML API — Day-9 production image.
# Builds a slim, non-root container for the async FastAPI service (api.py).
# The Flask UI (app.py) is a separate concern and is NOT built here.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps kept minimal; pdfplumber/scipy wheels are manylinux so no build tools needed.
COPY requirements-api.txt ./
RUN pip install --upgrade pip && pip install -r requirements-api.txt

# App code + trained model artifact (the classifier joblib is baked in at build).
COPY src/ ./src/
COPY api.py ./
COPY models/ ./models/

# Non-root runtime.
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Container healthcheck hits the service's own /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
