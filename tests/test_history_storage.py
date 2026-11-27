import pytest

from mlforge.monitoring.storage import TelemetryStore


def test_history_persists_and_aggregates(tmp_path):
    url = f"sqlite:///{tmp_path / 'events.db'}"
    store = TelemetryStore(url)
    store.initialize()
    for version, latency, probability, prediction in [
        ("v0.3.1", 10, 0.2, 0),
        ("v0.3.1", 30, 0.8, 1),
        ("v0.3.0", 20, 0.4, 1),
    ]:
        store.record(
            model_version=version,
            latency_ms=latency,
            probability=probability,
            prediction=prediction,
        )
    store.close()
    reopened = TelemetryStore(url)
    try:
        result = reopened.history(days=7, model_version="v0.3.1")
        assert len(result["points"]) == 1
        point = result["points"][0]
        assert point["predictions"] == 2
        assert point["mean_latency_ms"] == pytest.approx(20)
        assert point["p95_latency_ms"] == pytest.approx(29)
        assert point["predicted_churn_rate"] == pytest.approx(0.5)
        assert result["model_versions"] == ["v0.3.1"]
    finally:
        reopened.close()


def test_history_empty_and_invalid(tmp_path):
    store = TelemetryStore(f"sqlite:///{tmp_path / 'empty.db'}")
    store.initialize()
    try:
        assert store.history()["points"] == []
        with pytest.raises(ValueError):
            store.history(days=0)
    finally:
        store.close()
