FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_MODE=space \
    HOME=/home/user

# System libraries needed by LightGBM/scikit-learn (libgomp1) and geospatial stacks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libexpat1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user

# Install uv for fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:0.9.6 /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml ./
RUN uv pip install --system --no-cache ".[gee,ai]"

COPY app ./app
COPY core ./core
COPY ai ./ai
COPY models ./models
COPY plugins ./plugins
COPY README.md ./

RUN mkdir -p /app/data/stacks /app/data/outputs && chown -R user:user /app

USER user

EXPOSE 8501

CMD ["streamlit", "run", "app/main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
