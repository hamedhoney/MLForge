FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

RUN uv sync --locked --no-dev --no-editable

COPY artifacts/tuning/best_model.joblib \
     /app/artifacts/tuning/best_model.joblib

COPY artifacts/tuning/metadata.json \
     /app/artifacts/tuning/metadata.json

ENV PATH="/app/.venv/bin:$PATH"
ENV MLFORGE_MODEL_DIR="/app/artifacts/tuning"
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["uvicorn", "mlforge.api.main:app", \
     "--host", "0.0.0.0", "--port", "7860"]
