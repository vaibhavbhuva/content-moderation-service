FROM python:3.12-slim

# Install minimal dependencies
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy project files
COPY requirements.txt .
COPY pyproject.toml* uv.lock* ./

# Install dependencies using uv
RUN uv sync --frozen --no-cache

# Copy source code
COPY src ./src
COPY scripts ./scripts

# Pre-download Hugging Face models (stored in default cache)
RUN uv run python scripts/download_models.py

EXPOSE 8000

# Health check
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1

# Run application   
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]