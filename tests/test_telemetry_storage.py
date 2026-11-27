import pytest

from mlforge.monitoring.storage import TelemetryStore


@pytest.fixture
def store(tmp_path):
    database = tmp_path / "telemetry.db"
    instance = TelemetryStore(f"sqlite:///{database}")
    instance.initialize()
    yield instance
    instance.close()


def test_persistence(store, tmp_path):
    store.record(
        model_version="v0.3.1",
        latency_ms=15.0,
        probability=0.8,
        prediction=1,
    )

    another = TelemetryStore(f"sqlite:///{tmp_path / 'telemetry.db'}")
    try:
        result = another.summary()
        assert result["total_predictions"] == 1
        assert result["mean_probability"] == pytest.approx(0.8)
    finally:
        another.close()


def test_empty_database(store):
    result = store.summary()
    assert result["total_predictions"] == 0
    assert result["mean_latency_ms"] is None


def test_invalid_probability(store):
    with pytest.raises(ValueError):
        store.record(
            model_version="v0.3.1",
            latency_ms=10,
            probability=1.5,
            prediction=1,
        )


def test_invalid_latency(store):
    with pytest.raises(ValueError):
        store.record(
            model_version="v0.3.1",
            latency_ms=-1,
            probability=0.5,
            prediction=0,
        )


def test_retention(store):
    store.record(
        model_version="v0.3.1",
        latency_ms=10,
        probability=0.5,
        prediction=1,
    )

    assert store.delete_older_than(days=90) == 0
    assert store.summary()["total_predictions"] == 1


def test_expired_records_are_deleted(store):
    from datetime import UTC, datetime, timedelta

    from sqlalchemy import insert

    from mlforge.monitoring.storage import InferenceRecord

    old_timestamp = datetime.now(UTC) - timedelta(days=120)

    with store.engine.begin() as connection:
        connection.execute(
            insert(InferenceRecord).values(
                timestamp=old_timestamp,
                model_version="v0.3.1",
                latency_ms=12.0,
                probability=0.4,
                prediction=0,
            )
        )

    store.record(
        model_version="v0.3.1",
        latency_ms=15.0,
        probability=0.8,
        prediction=1,
    )

    deleted = store.delete_older_than(days=90)

    assert deleted == 1
    assert store.summary(days=30)["total_predictions"] == 1
