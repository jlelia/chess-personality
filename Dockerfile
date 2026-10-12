FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install python dependencies using frozen lockfile
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application code and models
COPY app/ ./app/
COPY model.py preprocess_games.py demo.py test.pgn ./

# Download release model checkpoints (13MB)
RUN mkdir -p checkpoints_v2 && \
    curl -fL -o checkpoints_v2/supcon_best.pt https://github.com/jlelia/chess-personality/releases/download/v1.0.0/supcon_best.pt && \
    curl -fL -o checkpoints_v2/gm_centroids.pt https://github.com/jlelia/chess-personality/releases/download/v1.0.0/gm_centroids.pt

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 7860
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
