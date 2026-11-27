# MLForge historical telemetry integration

Copy the files in this ZIP into the repository root, preserving paths.

## Add the endpoint to `src/mlforge/api/main.py`

Add the following code below the existing `/monitoring/summary` endpoint. It reuses the existing token validation pattern and returns aggregated data only. For simplicity, this endpoint repeats the authentication check; you can refactor both endpoints into a shared dependency later.

```python
@app.get("/monitoring/history")
def monitoring_history(
    request: Request,
    x_monitoring_token: str | None = Header(default=None),
    days: int = 30,
    model_version: str | None = None,
):
    expected = os.getenv("MLFORGE_MONITORING_TOKEN")
    if not expected:
        raise HTTPException(status_code=503, detail="Monitoring access not configured")
    if not x_monitoring_token or not secrets.compare_digest(
        x_monitoring_token, expected
    ):
        raise HTTPException(status_code=403, detail="Forbidden")
    if not 1 <= days <= 365:
        raise HTTPException(status_code=422, detail="days must be between 1 and 365")
    if model_version is not None and (not model_version or len(model_version) > 64):
        raise HTTPException(status_code=422, detail="Invalid model version")
    store = getattr(request.app.state, "telemetry_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Persistent telemetry unavailable")
    try:
        return store.history(days=days, model_version=model_version)
    except Exception:
        logger.exception("Unable to query historical telemetry")
        raise HTTPException(
            status_code=503, detail="Historical telemetry unavailable"
        ) from None
```

The `history()` implementation groups UTC daily observations in Python for portability between SQLite and PostgreSQL. This is suitable for a low-traffic demo, but for high volume move aggregation into SQL and add preaggregated rollups. No schema migration is needed: it reads the existing `inference_events` table.

## CI

Add to the existing API test step in `.github/workflows/api.yml`:

```yaml
uv run --locked pytest tests/test_history_storage.py -q
uv run --locked pytest tests/test_history_client.py -q
```

The root-level `api_client.py` is imported by Streamlit. Run its tests with `PYTHONPATH=.` or configure pytest's pythonpath to include the repository root. For example, in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
pythonpath = [".", "src"]
```

Do not overwrite an existing pytest configuration block; merge these values.

## Verify

After CI and Render deployment, open the authenticated `/monitoring/history` endpoint in Swagger and check that `points` contains daily aggregates. The Streamlit dashboard then shows the volume, latency, and prediction-rate charts. This is *prediction telemetry*, not ground-truth performance or feature drift.
