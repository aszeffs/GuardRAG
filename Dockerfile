# Pinned by digest, so a moved tag cannot change what gets built. Dependabot
# bumps the tag and digest together.
FROM python:3.12-slim@sha256:dddfd7e07f9d15aeeca61529320492139d21cac7f0070c00609243e51e4e0016 AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FASTEMBED_CACHE_PATH=/models
WORKDIR /app

# The base image lags Debian's security archive by days to weeks. Upgrading here
# takes fixes Debian has already published; Trivy then scans exactly the result.
RUN apt-get update \
    && apt-get upgrade --yes --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY guardrag ./guardrag
# pip is needed only to install; removing it takes its CVEs out of the image.
RUN pip install . && pip uninstall --yes pip

RUN useradd --create-home --uid 10001 app && mkdir -p /models /app/data && chown app /models /app/data
USER app

EXPOSE 8000
CMD ["uvicorn", "guardrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
