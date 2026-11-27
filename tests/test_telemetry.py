import pytest

from mlforge.monitoring.telemetry import InferenceTelemetry


def test_empty_summary():
    collector = InferenceTelemetry()

    result = collector.summary()

    assert result["total_predictions"] == 0
    assert result["retained_events"] == 0
    assert result["mean_latency_ms"] is None


def test_record_prediction():
    collector = InferenceTelemetry()

    collector.record(
        model_version="v0.3.0",
        latency_ms=12.5,
        probability=0.8,
        prediction=1,
    )

    result = collector.summary()

    assert result["total_predictions"] == 1
    assert result["mean_latency_ms"] == pytest.approx(12.5)
    assert result["predicted_churn_rate"] == 1.0
    assert result["model_versions"] == ["v0.3.0"]


def test_bounded_storage():
    collector = InferenceTelemetry(max_events=2)

    for probability in (0.1, 0.5, 0.9):
        collector.record(
            model_version="v0.3.0",
            latency_ms=10,
            probability=probability,
            prediction=int(probability >= 0.5),
        )

    result = collector.summary()

    assert result["total_predictions"] == 3
    assert result["retained_events"] == 2


def test_invalid_probability():
    collector = InferenceTelemetry()

    with pytest.raises(ValueError):
        collector.record(
            model_version="v0.3.0",
            latency_ms=10,
            probability=1.5,
            prediction=1,
        )


def test_error_counter():
    collector = InferenceTelemetry()

    collector.record_error()
    collector.record_error()

    assert collector.summary()["error_count"] == 2


def test_invalid_capacity():
    with pytest.raises(ValueError):
        InferenceTelemetry(max_events=0)
