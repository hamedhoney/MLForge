"""Focused tests for persistent telemetry integration without production credentials."""

from types import SimpleNamespace
from unittest.mock import Mock

from mlforge.api.main import _predict_and_record
from mlforge.monitoring.telemetry import InferenceTelemetry


def test_inference_writes_one_event_per_prediction():
    results = [
        SimpleNamespace(churn_probability=0.2, prediction=0),
        SimpleNamespace(churn_probability=0.8, prediction=1),
    ]
    service = SimpleNamespace(
        model_version="v0.3.1", predict=Mock(return_value=results)
    )
    store = Mock()
    collector = InferenceTelemetry()
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                model_service=service, telemetry=collector, telemetry_store=store
            )
        )
    )
    assert _predict_and_record([object(), object()], request) == results
    assert store.record.call_count == 2
    assert collector.summary()["total_predictions"] == 2
    assert store.record.call_args_list[0].kwargs["model_version"] == "v0.3.1"


def test_database_failure_does_not_block_predictions():
    result = SimpleNamespace(churn_probability=0.7, prediction=1)
    service = SimpleNamespace(
        model_version="v0.3.1", predict=Mock(return_value=[result])
    )
    store = Mock()
    store.record.side_effect = RuntimeError("database unavailable")
    collector = InferenceTelemetry()
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                model_service=service, telemetry=collector, telemetry_store=store
            )
        )
    )
    assert _predict_and_record([object()], request) == [result]
    assert collector.summary()["total_predictions"] == 1
