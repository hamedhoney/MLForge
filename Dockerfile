FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

WORKDIR /app

# Python runtime settings
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV UV_LINK_MODE=copy
ENV PATH="/app/.venv/bin:$PATH"

# Install project dependencies
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

RUN uv sync --locked --no-dev --no-editable

# Model artifacts are downloaded at runtime,
# not copied from the Git repository.
RUN mkdir -p /app/artifacts/tuning

EXPOSE 7860

CMD ["uvicorn", "mlforge.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
