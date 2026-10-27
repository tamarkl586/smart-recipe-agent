"""Context assembly: builds the request sent to the model each turn from the
system prompt, retrieved memory, tool definitions and conversation history.

The system instruction and tool list are frozen for the whole session. Per-turn
context (memory facts, a compaction summary) is added as text parts inside the
new user message, which then stays in history unchanged.
"""

from __future__ import annotations

from typing import Any

from google.genai import types

from . import memory
from .caching import PromptCache

_TAG = "[API/harness/context.py]"

SYSTEM_PROMPT = """You are a friendly, concise cooking assistant.

Tools:
- search_recipes: look up recipes in the internal database before suggesting one.
- fetch_external_nutrition_info: get nutrition facts; never invent numbers.
- save_user_preference_or_recipe: when the user states a lasting preference, allergy, \
dislike, or asks you to remember/save something, store it.

Each user message may start with a <user_memory> block listing what you already know \
about the user. Respect it (e.g. never suggest dishes that conflict with their diet or \
allergies, and point out the conflict if the user asks for one). If a \
<conversation_summary> block is present, it summarizes earlier turns.

Keep answers short: recipe name, key ingredients, brief steps, and nutrition when asked."""


def _log(msg: str) -> None:
    print(f"{_TAG} {msg}", flush=True)


def build_user_turn(user_text: str, summary: str | None = None) -> list[dict[str, Any]]:
    """Parts for a new user message: [summary] [memory] text."""
    parts: list[dict[str, Any]] = []
    if summary:
        parts.append({"text": f"<conversation_summary>\n{summary}\n</conversation_summary>"})
    mem = memory.format_memory_block(user_text)
    if mem:
        parts.append({"text": mem})
    parts.append({"text": user_text})
    _log(f"built user turn with {len(parts)} part(s) (summary={'yes' if summary else 'no'}, "
         f"memory={'yes' if mem else 'no'})")
    return parts


def build_config(cache: PromptCache, *, max_output_tokens: int, thinking_budget: int | None,
                 allow_tools: bool = True) -> types.GenerateContentConfig:
    config: dict[str, Any] = {
        **cache.prefix_config(),
        "max_output_tokens": max_output_tokens,
        # The harness runs the tool loop itself (loop.py), never the SDK.
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        "tool_config": types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
            mode="AUTO" if allow_tools else "NONE")),
    }
    if thinking_budget is not None:
        config["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    return types.GenerateContentConfig(**config)


def build_request(contents: list[types.Content], *, model: str,
                  config: types.GenerateContentConfig) -> dict[str, Any]:
    """Assemble kwargs for client.models.generate_content."""
    _log(f"request ready: model={model} messages={len(contents)} "
         f"tools_mode={config.tool_config.function_calling_config.mode.value} "
         f"cached_content={'yes' if config.cached_content else 'no'}")
    return {"model": model, "contents": contents, "config": config}
