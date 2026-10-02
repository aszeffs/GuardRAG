FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FASTEMBED_CACHE_PATH=/models
WORKDIR /app

COPY pyproject.toml ./
COPY guardrag ./guardrag
RUN pip install .

RUN useradd --create-home --uid 10001 app && mkdir -p /models /app/data && chown app /models /app/data
USER app

EXPOSE 8000
CMD ["uvicorn", "guardrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
