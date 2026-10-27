"""Token accounting: counts/estimates tokens for requests, tracks usage reported
by the API, and exposes budget checks used by compaction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

_TAG = "[API/harness/tokens.py]"


def _log(msg: str) -> None:
    print(f"{_TAG} {msg}", flush=True)


DEFAULT_MODEL = "gemini-3.5-flash"

# USD per million tokens: (uncached input, output incl. thinking, cache write, cache read).
# Google AI Studio free tier is not billed, so every rate is zero. Add a paid-tier
# entry here (and pass tier="paid") to estimate paid costs.
PRICING: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "free": {},
}


def _field(usage: Any, name: str) -> int:
    value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
    return int(value or 0)


def estimate_tokens(obj: Any) -> int:
    """Rough token estimate (~4 chars/token) for when no API usage is available."""
    text = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    return max(1, len(text) // 4)


@dataclass
class TokenTracker:
    model: str = DEFAULT_MODEL
    tier: str = "free"
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    requests: int = 0
    last_context_tokens: int = 0
    per_request: list[dict[str, Any]] = field(default_factory=list)

    def record(self, usage: Any, label: str = "request") -> dict[str, Any]:
        """Add one response's usage_metadata; returns that request's breakdown.

        Gemini's prompt_token_count includes cached tokens, so uncached input is
        prompt - cached. Implicit caching has no separate write charge (0).
        """
        prompt = _field(usage, "prompt_token_count") + _field(usage, "tool_use_prompt_token_count")
        cached = _field(usage, "cached_content_token_count")
        thoughts = _field(usage, "thoughts_token_count")
        entry = {
            "input": prompt - cached,
            "output": _field(usage, "candidates_token_count") + thoughts,
            "thinking": thoughts,
            "cache_read": cached,
            "cache_write": 0,
        }
        self.input_tokens += entry["input"]
        self.output_tokens += entry["output"]
        self.thinking_tokens += entry["thinking"]
        self.cache_read_tokens += entry["cache_read"]
        self.requests += 1
        self.last_context_tokens = prompt
        self.per_request.append({"label": label, **entry})
        _log(
            f"{label}: in={entry['input']} out={entry['output']} (thinking={thoughts}) "
            f"cache_read={cached} context={prompt}"
        )
        return entry

    def cost_usd(self) -> float:
        rates = PRICING.get(self.tier, {}).get(self.model)
        if not rates:
            return 0.0
        p_in, p_out, p_write, p_read = rates
        return (self.input_tokens * p_in + self.output_tokens * p_out
                + self.cache_write_tokens * p_write + self.cache_read_tokens * p_read) / 1_000_000

    def over_budget(self, threshold: int) -> bool:
        over = self.last_context_tokens > threshold
        if over:
            _log(f"context {self.last_context_tokens} exceeds threshold {threshold}")
        return over

    def summary(self) -> dict[str, Any]:
        cost = round(self.cost_usd(), 6)
        _log(f"totals: requests={self.requests} input={self.input_tokens} output={self.output_tokens} "
             f"cache_read={self.cache_read_tokens} cost=${cost} ({self.tier} tier)")
        return {
            "model": self.model,
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "estimated_cost_usd": cost,
        }
