import pytest

import api_client


def test_history_query(monkeypatch):
    captured = {}

    def fake_request(base_url, path, *, token=None, **kwargs):
        captured.update(base_url=base_url, path=path, token=token)
        return {"points": [], "window_days": 7}

    monkeypatch.setattr(api_client, "_request", fake_request)
    assert (
        api_client.get_history("https://example.org", "secret", 7, "v0.3.1")["points"]
        == []
    )
    assert "model_version=v0.3.1" in captured["path"]
    assert captured["token"] == "secret"


def test_history_requires_token():
    with pytest.raises(api_client.APIClientError):
        api_client.get_history("https://example.org", "")
