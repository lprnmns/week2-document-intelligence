FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY projects/document_intelligence_service/pyproject.toml ./
COPY projects/document_intelligence_service/app ./app
COPY projects/document_intelligence_service/eval ./eval
COPY data/evaluations ./data/evaluations
COPY data/evaluations/atlas_orion_demo ./data/evaluations/atlas_orion_demo

# The host is CPU-only. Installing torch from the CPU wheel index prevents
# pip from pulling the much larger CUDA runtime into the API image.
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      "torch==2.13.0+cpu" \
    && pip install --no-cache-dir .

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=15s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/health/live', timeout=2)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
