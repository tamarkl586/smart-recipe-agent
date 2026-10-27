"""Long-term memory: persists facts across sessions (e.g. user preferences,
dietary restrictions) and retrieves the relevant ones for the current turn.
"""

from __future__ import annotations

import re
from typing import Any

from shared import load_memory

_TAG = "[API/harness/memory.py]"

MAX_NOTES = 5
MAX_SAVED_TITLES = 5


def _log(msg: str) -> None:
    print(f"{_TAG} {msg}", flush=True)


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2}


def relevant_facts(user_text: str, memory: dict[str, Any] | None = None) -> list[str]:
    """Pick the memory facts worth showing for this turn.

    Preferences are always included (few, high value). Notes and saved recipes
    are included when they share words with the user's message, falling back
    to the most recent ones.
    """
    memory = memory if memory is not None else load_memory()
    facts = [f"{k}: {v}" for k, v in memory.get("preferences", {}).items()]

    query = _words(user_text)
    notes = [n.get("text", "") for n in memory.get("notes", [])]
    matched = [n for n in notes if _words(n) & query]
    facts += [f"note: {n}" for n in (matched or notes[-MAX_NOTES:])[:MAX_NOTES]]

    saved = memory.get("saved_recipes", [])
    titles = [f"{r.get('title')} (id {r.get('id')})" for r in saved
              if _words(r.get("title", "")) & query] or \
             [f"{r.get('title')} (id {r.get('id')})" for r in saved[-MAX_SAVED_TITLES:]]
    if titles:
        facts.append("saved recipes: " + ", ".join(titles[:MAX_SAVED_TITLES]))

    _log(f"selected {len(facts)} fact(s) from memory")
    return facts


def format_memory_block(user_text: str) -> str | None:
    """Render relevant facts as a compact block for the user turn, or None if empty."""
    facts = relevant_facts(user_text)
    if not facts:
        _log("no stored facts for this user")
        return None
    return "<user_memory>\n" + "\n".join(f"- {f}" for f in facts) + "\n</user_memory>"
