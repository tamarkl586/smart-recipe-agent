"""HTTP settings shared by both Gemini clients so the two implementations are
measured under identical network conditions."""

from __future__ import annotations

import time
from typing import Any, Callable

import httpx
from google.genai import types

# The client retries 429/5xx and timeouts itself; the timeout turns a hung
# request into a retryable error instead of waiting for the server to drop it.
HTTP_OPTIONS = types.HttpOptions(
    timeout=120_000,  # milliseconds
    retry_options=types.HttpRetryOptions(
        attempts=6, initial_delay=5.0, max_delay=65.0, http_status_codes=[408, 429, 500, 502, 503, 504]),
)

# Not retried by the client: the server closing the connection mid-request.
_DISCONNECTS = (httpx.RemoteProtocolError, httpx.ReadError)


def retry_disconnects(call: Callable[..., Any], attempts: int = 3, delay: float = 10.0) -> Callable[..., Any]:
    """Wrap a single model call so a dropped connection is retried."""

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        for attempt in range(1, attempts + 1):
            try:
                return call(*args, **kwargs)
            except _DISCONNECTS as exc:
                if attempt == attempts:
                    raise
                print(f"[transport] {type(exc).__name__}: retrying in {delay:.0f}s "
                      f"(attempt {attempt + 1}/{attempts})", flush=True)
                time.sleep(delay)

    return wrapped
