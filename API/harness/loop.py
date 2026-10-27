"""Agent loop: sends the conversation to the model, executes requested tool calls,
appends the results, and repeats until the model produces a final answer or a
turn limit is reached.
"""

from __future__ import annotations

from typing import Any, Callable

from . import tools
from .caching import PromptCache
from .history import History
from .tokens import TokenTracker

_TAG = "[API/harness/loop.py]"

_REFUSALS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "IMAGE_SAFETY"}


def _log(msg: str) -> None:
    print(f"{_TAG} {msg}", flush=True)


def response_parts(response: Any) -> list[dict[str, Any]]:
    """The model turn's parts as JSON-mode dicts, unchanged (keeps thought signatures)."""
    if not response.candidates or not response.candidates[0].content:
        return []
    content = response.candidates[0].content.model_dump(mode="json", exclude_none=True)
    return content.get("parts", [])


def stop_reason(response: Any, parts: list[dict[str, Any]]) -> str:
    """Normalize Gemini's finish_reason (which is STOP even for function calls)."""
    if any(p.get("function_call") for p in parts):
        return "tool_use"
    if not response.candidates:
        feedback = getattr(response, "prompt_feedback", None)
        return "refusal" if feedback and feedback.block_reason else "empty"
    reason = response.candidates[0].finish_reason
    name = getattr(reason, "value", str(reason or "STOP"))
    if name == "STOP":
        return "end_turn"
    if name == "MAX_TOKENS":
        return "max_tokens"
    return "refusal" if name in _REFUSALS else name.lower()


def final_text(parts: list[dict[str, Any]]) -> str:
    return "\n".join(p["text"] for p in parts if p.get("text") and not p.get("thought")).strip()


def run_turn(
    history: History,
    tracker: TokenTracker,
    send: Callable[[History], Any],
    *,
    max_iterations: int = 8,
) -> dict[str, Any]:
    """Run model requests while the model keeps requesting function calls
    (stop_reason == "tool_use"). Returns the final text, stop reason and the
    tool calls made this turn."""
    tool_log: list[dict[str, Any]] = []
    reason = None
    parts: list[dict[str, Any]] = []

    for iteration in range(1, max_iterations + 1):
        _log(f"Executing turn iteration {iteration}/{max_iterations}")
        response = send(history)
        if response.usage_metadata:
            tracker.record(response.usage_metadata, label=f"iteration {iteration}")
            PromptCache.log_hit(response.usage_metadata)
        parts = response_parts(response)
        reason = stop_reason(response, parts)
        if parts:
            history.append_model(parts)
        _log(f"stop_reason={reason}")

        if reason == "tool_use":
            calls = tools.parse_tool_calls(parts)
            results, errors = tools.execute_tool_calls(calls)
            for call, is_error in zip(calls, errors):
                tool_log.append({"name": call.get("name"), "input": call.get("args") or {},
                                 "is_error": is_error})
            # All function responses for this model turn go back in a single user message.
            history.append_user(results)
            continue
        if reason == "refusal":
            _log("response blocked by safety filters")
        elif reason == "max_tokens":
            _log("hit max_output_tokens; returning partial answer")
        break
    else:
        _log("max iterations reached; stopping tool loop")
        reason = "max_iterations"

    _close_open_tool_calls(history)
    text = final_text(parts)
    if not text and reason in ("refusal", "empty"):
        text = "Sorry, I can't help with that request."
    return {"text": text, "stop_reason": reason, "tool_calls": tool_log}


def _close_open_tool_calls(history: History) -> None:
    """If the turn ended with unanswered function calls, answer them so the
    next request is valid (every function_call needs a function_response)."""
    if not history.messages or history.messages[-1]["role"] != "model":
        return
    pending = tools.parse_tool_calls(history.messages[-1]["parts"])
    if pending:
        _log(f"closing {len(pending)} unanswered function call(s)")
        history.append_user([
            {"function_response": {"name": c.get("name"), "response": {"error": "Not executed: turn ended."},
                                   **({"id": c["id"]} if c.get("id") else {})}}
            for c in pending
        ])
