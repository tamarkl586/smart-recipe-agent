"""Prompt caching: places cache breakpoints on stable prefixes (system prompt,
tools, older history) to reduce latency and cost on repeated requests.

Gemini has no per-block breakpoints. It offers:
  - implicit caching (automatic on repeated request prefixes), which this module
    supports by freezing the system instruction + tools for the whole session; and
  - explicit caching (a server-side CachedContent holding the stable prefix),
    used here when enabled and the prefix is large enough; otherwise it falls
    back to implicit caching.
"""

from __future__ import annotations

from typing import Any

from google.genai import types

_TAG = "[API/harness/caching.py]"

# Explicit caches below this size are rejected by the API.
MIN_EXPLICIT_CACHE_TOKENS = 1024


def _log(msg: str) -> None:
    print(f"{_TAG} {msg}", flush=True)


class PromptCache:
    def __init__(self, client: Any, model: str, system_instruction: str,
                 tools: list[types.Tool], *, explicit: bool = False, ttl: str = "600s") -> None:
        self.system_instruction = system_instruction
        self.tools = tools
        self.cache_name: str | None = None
        if explicit:
            self.cache_name = self._create_explicit(client, model, ttl)
        else:
            _log("implicit caching: stable system instruction + tools prefix frozen for session")

    def _create_explicit(self, client: Any, model: str, ttl: str) -> str | None:
        try:
            size = client.models.count_tokens(
                model=model, contents="x",
                config=types.CountTokensConfig(system_instruction=self.system_instruction, tools=self.tools),
            ).total_tokens
            if size < MIN_EXPLICIT_CACHE_TOKENS:
                _log(f"prefix is {size} tokens (< {MIN_EXPLICIT_CACHE_TOKENS}); using implicit caching")
                return None
            cache = client.caches.create(model=model, config=types.CreateCachedContentConfig(
                system_instruction=self.system_instruction, tools=self.tools, ttl=ttl))
            _log(f"explicit cache {cache.name} created for {size}-token prefix")
            return cache.name
        except Exception as exc:
            _log(f"explicit cache unavailable ({type(exc).__name__}: {exc}); using implicit caching")
            return None

    def prefix_config(self) -> dict[str, Any]:
        """Config fields for the stable prefix: either a cache reference or the
        identical system instruction + tools on every request."""
        if self.cache_name:
            return {"cached_content": self.cache_name}
        return {"system_instruction": self.system_instruction, "tools": self.tools}

    @staticmethod
    def log_hit(usage: Any) -> None:
        prompt = getattr(usage, "prompt_token_count", 0) or 0
        cached = getattr(usage, "cached_content_token_count", 0) or 0
        pct = 100 * cached / prompt if prompt else 0
        _log(f"cache hit: {cached}/{prompt} prompt tokens ({pct:.0f}%)")
