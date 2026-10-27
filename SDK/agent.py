"""Recipe agent built on the Google GenAI SDK's high-level constructs: a Chat
session (client.chats) with automatic function calling (AFC). The SDK owns the
tool loop, schema generation, dispatch and conversation history that the API
version implements by hand in API/harness/.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from shared import load_memory
from shared.transport import HTTP_OPTIONS, retry_disconnects

from .tools import TOOLS

DEFAULT_MODEL = "gemini-3.5-flash"

SYSTEM_PROMPT = """You are a friendly, concise cooking assistant.

Tools:
- search_recipes: look up recipes in the internal database before suggesting one.
- fetch_external_nutrition_info: get nutrition facts; never invent numbers.
- save_user_preference_or_recipe: when the user states a lasting preference, allergy, \
dislike, or asks you to remember/save something, store it.

Each user message may start with a <user_memory> block listing what you already know \
about the user. Respect it (e.g. never suggest dishes that conflict with their diet or \
allergies, and point out the conflict if the user asks for one).

Keep answers short: recipe name, key ingredients, brief steps, and nutrition when asked."""

def memory_block() -> str | None:
    """Compact summary of persisted user facts to prepend to a prompt."""
    mem = load_memory()
    facts = [f"{k}: {v}" for k, v in mem["preferences"].items()]
    facts += [f"note: {n['text']}" for n in mem["notes"][-5:]]
    if mem["saved_recipes"]:
        facts.append("saved recipes: " + ", ".join(r["title"] for r in mem["saved_recipes"][-5:]))
    return "<user_memory>\n" + "\n".join(f"- {f}" for f in facts) + "\n</user_memory>" if facts else None


class UsageMeter:
    """Measurement only (not agent logic): AFC runs several model calls inside one
    send_message() but the response carries only the last call's usage_metadata.
    This wraps the client's per-call method so every call's usage is counted.
    It also retries a dropped connection for that single call (same policy as
    the API harness), below the AFC loop so chat history is never duplicated."""

    def __init__(self, client: genai.Client) -> None:
        self.calls: list[types.GenerateContentResponseUsageMetadata] = []
        models = client.models
        original = getattr(models, "_generate_content_with_continuation", None)
        self.complete = original is not None
        if original is None:
            return

        call = retry_disconnects(original)

        def metered(*args: Any, **kwargs: Any) -> Any:
            response = call(*args, **kwargs)
            if response is not None and response.usage_metadata:
                self.calls.append(response.usage_metadata)
            return response

        models._generate_content_with_continuation = metered

    def totals(self) -> dict[str, int]:
        def s(name: str) -> int:
            return sum(getattr(u, name, 0) or 0 for u in self.calls)

        prompt = s("prompt_token_count") + s("tool_use_prompt_token_count")
        cached = s("cached_content_token_count")
        thoughts = s("thoughts_token_count")
        return {
            "input_tokens": prompt - cached,
            "output_tokens": s("candidates_token_count") + thoughts,
            "thinking_tokens": thoughts,
            "cache_read_tokens": cached,
            "cache_write_tokens": 0,
            "model_calls": len(self.calls),
        }


class RecipeAgent:
    def __init__(self, *, model: str | None = None, max_tool_calls: int | None = None,
                 thinking_budget: int | None = None, resume_history: str | Path | None = None,
                 client: genai.Client | None = None) -> None:
        self.model = model or DEFAULT_MODEL
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if client is None and not key:
            raise PermissionError("GEMINI_API_KEY is not set (add it to .env)")
        self.client = client or genai.Client(api_key=key, http_options=HTTP_OPTIONS)
        self.meter = UsageMeter(self.client)
        self.compactions = 0
        self.config = types.GenerateContentConfig(
            # replaces: context.py (frozen system prompt)
            system_instruction=SYSTEM_PROMPT,
            # replaces: tools.py (declarations generated from Python callables)
            tools=TOOLS,
            # replaces: loop.py (AFC runs call -> execute -> respond until a text answer)
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=max_tool_calls or 8),
            thinking_config=(types.ThinkingConfig(thinking_budget=thinking_budget)
                             if thinking_budget is not None else None),
        )
        history = None
        if resume_history and Path(resume_history).exists():
            # replaces: history.py (load a saved conversation to resume it)
            history = [types.Content.model_validate(c)
                       for c in json.loads(Path(resume_history).read_text(encoding="utf-8"))]
        # replaces: history.py (the Chat object keeps the ordered transcript)
        # replaces: caching.py (no explicit cache; Gemini's implicit caching applies to the repeated prefix)
        # replaces: compaction.py (none in the SDK: the chat grows unbounded within the 1M-token window)
        self.chat_session = self.client.chats.create(model=self.model, config=self.config, history=history)

    def save_history(self, path: str | Path) -> None:
        # replaces: history.py (persist the transcript for later resumption)
        history = [c.model_dump(mode="json", exclude_none=True) for c in self.chat_session.get_history()]
        Path(path).write_text(json.dumps(history, ensure_ascii=False, indent=1), encoding="utf-8")

    def chat(self, user_text: str) -> dict[str, Any]:
        start = time.perf_counter()
        calls_before = len(self.meter.calls)
        # replaces: memory.py (inject persisted user facts into the turn)
        mem = memory_block()
        prompt = f"{mem}\n\n{user_text}" if mem else user_text

        # replaces: loop.py (one call drives the whole function-calling round trip)
        history_before = len(self.chat_session.get_history())
        response = self.chat_session.send_message(prompt)

        # The chat records every AFC round (model function_call turn, then the SDK's
        # function_response turn) in its history, in order.
        tool_calls: list[dict[str, Any]] = []
        for content in self.chat_session.get_history()[history_before:]:
            parts = content.parts or []
            responses = [p.function_response for p in parts if p.function_response]
            for fc in (p.function_call for p in parts if p.function_call):
                tool_calls.append({"name": fc.name, "input": dict(fc.args or {}), "is_error": False})
            if responses:
                for call, fr in zip(tool_calls[-len(responses):], responses):
                    call["is_error"] = "error" in (fr.response or {})
        for call in tool_calls:
            print(f"[SDK/agent.py] tool call {call['name']} {call['input']}", flush=True)

        candidate = response.candidates[0] if response.candidates else None
        reason = getattr(candidate.finish_reason, "value", None) if candidate else "BLOCKED"
        return {
            "text": (response.text or "").strip(),
            "stop_reason": "end_turn" if reason == "STOP" else str(reason).lower(),
            "model_calls": len(self.meter.calls) - calls_before,
            "tool_calls": tool_calls,
            "compacted": False,
            "latency_s": round(time.perf_counter() - start, 2),
        }

    def run_session(self, name: str, turns: list[str]) -> dict[str, Any]:
        start = time.perf_counter()
        results = []
        for i, user_text in enumerate(turns, 1):
            print(f"\n=== [{name}] Turn {i}: {user_text}", flush=True)
            result = self.chat(user_text)
            print(f"=== [{name}] Assistant: {result['text']}\n", flush=True)
            results.append({"user": user_text, **result})
        return {
            "name": name,
            "turns": results,
            # replaces: tokens.py (usage_metadata per call, gathered by UsageMeter; free tier = $0)
            "usage": {**self.meter.totals(), "estimated_cost_usd": 0.0,
                      "usage_complete": self.meter.complete},
            "compactions": self.compactions,
            "duration_s": round(time.perf_counter() - start, 2),
        }
