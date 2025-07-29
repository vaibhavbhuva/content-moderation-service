FROM python:3.12-slim

# Install minimal dependencies
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=8000

WORKDIR /app

# Copy project files
COPY requirements.txt .
COPY pyproject.toml* uv.lock* ./

# Install dependencies using uv
RUN uv sync --frozen --no-cache

# Copy source code
COPY src ./src

EXPOSE 8000

# Health check
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1

# Run application
# CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port $PORT"]
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
