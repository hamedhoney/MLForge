import json
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class APIClientError(RuntimeError):
    """A safe-to-display API communication error."""


def _request(base_url: str, path: str, *, payload=None, token=None, timeout=90):
    if not base_url.startswith("https://"):
        raise APIClientError("API_URL must use HTTPS")
    headers = {"Accept": "application/json"}
    if token:
        headers["X-Monitoring-Token"] = token
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    request = Request(base_url.rstrip("/") + path, data=data, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        raise APIClientError(f"API returned HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        raise APIClientError("Could not communicate with inference API") from exc


def get_model_info(base_url: str) -> dict:
    result = _request(base_url, "/model/info")
    if not isinstance(result, dict):
        raise APIClientError("Invalid model information response")
    return result


def predict_customer(base_url: str, customer: dict) -> tuple[dict, float]:
    start = time.perf_counter()
    result = _request(base_url, "/predict", payload=customer)
    elapsed_ms = (time.perf_counter() - start) * 1000
    if not isinstance(result, dict):
        raise APIClientError("Invalid prediction response")
    return result, elapsed_ms


def get_summary(base_url: str, token: str, days: int = 30) -> dict:
    if not token:
        raise APIClientError("Monitoring token is required")
    if not 1 <= days <= 365:
        raise ValueError("days must be between 1 and 365")
    result = _request(
        base_url, "/monitoring/summary?" + urlencode({"days": days}), token=token
    )
    if not isinstance(result, dict):
        raise APIClientError("Invalid monitoring response")
    return result


def get_history(
    base_url: str, token: str, days: int = 30, model_version: str | None = None
) -> dict:
    """Retrieve aggregated historical monitoring points."""
    if not token:
        raise APIClientError("Monitoring token is required")
    if not 1 <= days <= 365:
        raise ValueError("days must be between 1 and 365")
    params = {"days": days}
    if model_version:
        params["model_version"] = model_version
    result = _request(base_url, "/monitoring/history?" + urlencode(params), token=token)
    if not isinstance(result, dict) or not isinstance(result.get("points"), list):
        raise APIClientError("Invalid monitoring history response")
    return result
