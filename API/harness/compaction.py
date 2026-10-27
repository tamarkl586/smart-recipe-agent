"""History compaction: when the conversation approaches the context budget,
summarizes or trims older messages so the request stays within limits.

Uses "simple compaction": the whole history is summarized into one block and
nothing else is replayed. Compaction only runs between user turns, never in
the middle of a function-call round.
"""

from __future__ import annotations

from typing import Any, Callable

from .history import History
from .tokens import TokenTracker, estimate_tokens

_TAG = "[API/harness/compaction.py]"

SUMMARY_PROMPT = (
    "Summarize our conversation so far so it can replace the full history. Keep: the "
    "user's stated preferences, allergies and goals; recipes discussed or saved (names and "
    "ids); nutrition figures already fetched; open questions or next steps. Be concise, "
    "use bullet points. Do not call any tools; reply with the summary text only."
)
MAX_FALLBACK_CHARS = 1500


def _log(msg: str) -> None:
    print(f"{_TAG} {msg}", flush=True)


def needs_compaction(history: History, tracker: TokenTracker, threshold: int) -> bool:
    if len(history) < 2:
        return False
    if tracker.requests:
        return tracker.over_budget(threshold)
    est = estimate_tokens(history.messages)
    _log(f"estimated history size {est} tokens (threshold {threshold})")
    return est > threshold


def _fallback_summary(history: History) -> str:
    """Offline trim: keep the latest text from each side, newest last."""
    lines: list[str] = []
    for msg in history.messages:
        for part in msg.get("parts", []):
            text = (part.get("text") or "").strip()
            if not text or part.get("thought"):
                continue
            if text.startswith("<user_memory>") or text.startswith("<conversation_summary>"):
                continue
            lines.append(f"{msg['role']}: {text}")
    return "\n".join(lines)[-MAX_FALLBACK_CHARS:]


def compact(history: History, summarize: Callable[[History], str | None]) -> str:
    """Summarize the history, clear it, and return the summary for the next user turn.

    `summarize` receives a copy of the history with the summary request appended.
    """
    _log(f"compacting {len(history)} message(s)")
    request = History(history.snapshot())
    request.append_user([{"text": SUMMARY_PROMPT}])
    try:
        summary = summarize(request)
    except Exception as exc:
        _log(f"summary request failed ({type(exc).__name__}: {exc}); using trimmed transcript")
        summary = None
    if not summary:
        summary = _fallback_summary(history)
    history.replace_all([])
    _log(f"compaction done; summary is {len(summary)} chars")
    return summary
