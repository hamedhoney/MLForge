import io
import json
from urllib.error import HTTPError

import pytest

import api_client


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return io.BytesIO(json.dumps(self.data).encode())

    def __exit__(self, *args):
        return False


def test_predict_calls_render_api(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["payload"] = json.loads(request.data)
        return FakeResponse({"prediction": 1, "churn_probability": 0.6})

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)
    result, latency = api_client.predict_customer("https://example.com", {"tenure": 12})
    assert seen["url"] == "https://example.com/predict"
    assert seen["payload"] == {"tenure": 12}
    assert result["prediction"] == 1
    assert latency >= 0


def test_monitoring_header(monkeypatch):
    def fake_urlopen(request, timeout):
        assert request.get_header("X-monitoring-token") == "secret"
        assert request.full_url.endswith("/monitoring/summary?days=30")
        return FakeResponse({"source": "persistent_database", "total_predictions": 3})

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)
    assert (
        api_client.get_summary("https://example.com", "secret")["total_predictions"]
        == 3
    )


def test_http_error_does_not_expose_body(monkeypatch):
    def fake_urlopen(request, timeout):
        raise HTTPError(request.full_url, 503, "error", {}, None)

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)
    with pytest.raises(api_client.APIClientError, match="HTTP 503"):
        api_client.get_model_info("https://example.com")


def test_http_required():
    with pytest.raises(api_client.APIClientError, match="HTTPS"):
        api_client.get_model_info("http://example.com")
