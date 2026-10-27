"""Recipe agent built on the raw Gemini generate_content API, wired from the
hand-written harness pieces in API/harness/."""

from __future__ import annotations

import os
import time
from typing import Any

from google import genai

from shared.transport import HTTP_OPTIONS, retry_disconnects

from .harness import compaction, context, loop, tools
from .harness.caching import PromptCache
from .harness.history import History
from .harness.tokens import DEFAULT_MODEL, TokenTracker

DEFAULTS: dict[str, Any] = {
    "model": DEFAULT_MODEL,
    "max_tokens": 8192,
    "thinking_budget": None,  # None = model default (dynamic thinking)
    "max_tool_iterations": 8,
    "compaction_threshold_tokens": 20000,
    "explicit_cache": False,
}

def api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def make_client() -> genai.Client:
    key = api_key()
    if not key:
        raise PermissionError("GEMINI_API_KEY is not set (add it to .env)")
    return genai.Client(api_key=key, http_options=HTTP_OPTIONS)


class RecipeAgent:
    def __init__(self, client: Any = None, history: History | None = None, **options: Any) -> None:
        self.options = {**DEFAULTS, **{k: v for k, v in options.items() if v is not None}}
        self.client = client or make_client()
        self.history = history or History()
        self.tracker = TokenTracker(model=self.options["model"])
        self.cache = PromptCache(self.client, self.options["model"], context.SYSTEM_PROMPT,
                                 tools.get_tools(), explicit=self.options["explicit_cache"])

    def _send(self, history: History, *, allow_tools: bool = True) -> Any:
        config = context.build_config(self.cache, max_output_tokens=self.options["max_tokens"],
                                      thinking_budget=self.options["thinking_budget"],
                                      allow_tools=allow_tools)
        request = context.build_request(history.as_contents(), model=self.options["model"], config=config)
        return retry_disconnects(self.client.models.generate_content)(**request)

    def _summarize(self, request: History) -> str | None:
        response = self._send(request, allow_tools=False)
        if response.usage_metadata:
            self.tracker.record(response.usage_metadata, label="compaction summary")
        return loop.final_text(loop.response_parts(response)) or None

    def chat(self, user_text: str) -> dict[str, Any]:
        """Handle one user message end to end and return the answer with metadata."""
        start = time.perf_counter()
        summary = None
        if compaction.needs_compaction(self.history, self.tracker,
                                       self.options["compaction_threshold_tokens"]):
            summary = compaction.compact(self.history, self._summarize)
        self.history.append_user(context.build_user_turn(user_text, summary))
        result = loop.run_turn(self.history, self.tracker, self._send,
                               max_iterations=self.options["max_tool_iterations"])
        result["compacted"] = summary is not None
        result["latency_s"] = round(time.perf_counter() - start, 2)
        return result

    def run_session(self, name: str, turns: list[str]) -> dict[str, Any]:
        start = time.perf_counter()
        results = []
        for i, user_text in enumerate(turns, 1):
            print(f"\n=== [{name}] Turn {i}: {user_text}", flush=True)
            result = self.chat(user_text)
            print(f"=== [{name}] Assistant: {result['text']}\n", flush=True)
            results.append({"user": user_text, **result})
        summary = self.tracker.summary()
        return {
            "name": name,
            "turns": results,
            "usage": {
                "input_tokens": summary["input_tokens"],
                "output_tokens": summary["output_tokens"],
                "thinking_tokens": summary["thinking_tokens"],
                "cache_read_tokens": summary["cache_read_tokens"],
                "cache_write_tokens": summary["cache_write_tokens"],
                "model_calls": summary["requests"],
                "estimated_cost_usd": summary["estimated_cost_usd"],
                "usage_complete": True,
            },
            "per_request_usage": self.tracker.per_request,
            "compactions": self.history.compactions,
            "duration_s": round(time.perf_counter() - start, 2),
        }
