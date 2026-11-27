import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "api_client.py"

spec = importlib.util.spec_from_file_location(
    "api_client",
    MODULE_PATH,
)

if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load API client: {MODULE_PATH}")

api_client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api_client)


def test_get_history(monkeypatch):
    """Verify historical monitoring request parameters."""

    def fake_request(base_url, path, *, token=None, **kwargs):
        assert base_url == "https://example.org"
        assert path == ("/monitoring/history?days=7&model_version=v0.3.1")
        assert token == "secret"

        return {
            "points": [],
            "window_days": 7,
        }

    monkeypatch.setattr(
        api_client,
        "_request",
        fake_request,
    )

    result = api_client.get_history(
        "https://example.org",
        "secret",
        days=7,
        model_version="v0.3.1",
    )

    assert result["points"] == []
    assert result["window_days"] == 7


def test_history_requires_token():
    """Reject historical monitoring requests without a token."""

    with pytest.raises(
        api_client.APIClientError,
        match="Monitoring token is required",
    ):
        api_client.get_history(
            "https://example.org",
            "",
        )


def test_history_rejects_invalid_days():
    """Reject unsupported observation windows."""

    with pytest.raises(ValueError):
        api_client.get_history(
            "https://example.org",
            "secret",
            days=0,
        )

    with pytest.raises(ValueError):
        api_client.get_history(
            "https://example.org",
            "secret",
            days=366,
        )


def test_history_rejects_invalid_response(monkeypatch):
    """Reject malformed historical telemetry responses."""

    def fake_request(*args, **kwargs):
        return {"invalid": True}

    monkeypatch.setattr(
        api_client,
        "_request",
        fake_request,
    )

    with pytest.raises(
        api_client.APIClientError,
        match="Invalid monitoring history response",
    ):
        api_client.get_history(
            "https://example.org",
            "secret",
        )
