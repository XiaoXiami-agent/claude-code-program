import requests
from coze_chat_tool.errors import (
    CozeAPIError,
    CozeAuthError,
    CozePermissionError,
    CozeNotFoundError,
    CozeRateLimitError,
    CozeNetworkError,
)


def raise_for_code(http_status: int, coze_code: int, message: str) -> None:
    if http_status == 401:
        raise CozeAuthError(message, status_code=401, code=coze_code)
    if http_status == 403:
        raise CozePermissionError(message, status_code=403, code=coze_code)
    if http_status == 404:
        raise CozeNotFoundError(message, status_code=404, code=coze_code)
    if http_status == 429:
        raise CozeRateLimitError(message, status_code=429, code=coze_code)
    raise CozeAPIError(message, status_code=http_status, code=coze_code)


def request(session: requests.Session, method: str, url: str, timeout: float, **kwargs) -> dict:
    kwargs.setdefault("timeout", timeout)
    try:
        resp = session.request(method, url, **kwargs)
    except requests.RequestException as e:
        raise CozeNetworkError(str(e)) from e

    try:
        body = resp.json()
    except ValueError:
        raise CozeAPIError(
            f"Invalid JSON response: {resp.text[:200]}",
            status_code=resp.status_code,
        )

    coze_code = body.get("code", -1)
    if coze_code != 0:
        raise_for_code(resp.status_code, coze_code, body.get("msg", "Unknown error"))

    return body


def post_stream(session: requests.Session, url: str, payload: dict, timeout: float):
    try:
        resp = session.post(url, json=payload, stream=True, timeout=timeout)
    except requests.RequestException as e:
        raise CozeNetworkError(str(e)) from e

    if resp.status_code >= 400:
        try:
            body = resp.json()
        except Exception:
            body = {"code": -1, "msg": resp.text[:200]}
        raise_for_code(resp.status_code, body.get("code", -1), body.get("msg", "Unknown"))
    return resp
