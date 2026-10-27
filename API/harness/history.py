"""Conversation history: stores the ordered list of user/model/function-response
messages for the current session and provides helpers to append and read them.

History is append-only and stored as JSON-mode Gemini Content dicts. Model turns
are kept exactly as returned (including thought signatures, which Gemini needs
back on later requests). The only rewrite is compaction (see compaction.py).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from google.genai import types

_TAG = "[API/harness/history.py]"


def _log(msg: str) -> None:
    print(f"{_TAG} {msg}", flush=True)


def _as_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]
    return list(content)


class History:
    def __init__(self, messages: list[dict[str, Any]] | None = None) -> None:
        self.messages: list[dict[str, Any]] = messages or []
        self.compactions = 0

    def append_user(self, content: Any) -> None:
        """Append user parts; merges into a trailing user message so roles alternate."""
        parts = _as_parts(content)
        if self.messages and self.messages[-1]["role"] == "user":
            self.messages[-1]["parts"] = self.messages[-1]["parts"] + parts
            _log(f"merged {len(parts)} part(s) into trailing user message")
        else:
            self.messages.append({"role": "user", "parts": parts})
            _log(f"appended user message ({len(parts)} part(s)); total={len(self.messages)}")

    def append_model(self, parts: list[dict[str, Any]]) -> None:
        self.messages.append({"role": "model", "parts": parts})
        _log(f"appended model message ({len(parts)} part(s)); total={len(self.messages)}")

    def replace_all(self, messages: list[dict[str, Any]]) -> None:
        _log(f"replacing {len(self.messages)} message(s) with {len(messages)}")
        self.messages = messages
        self.compactions += 1

    def snapshot(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self.messages)

    def as_contents(self) -> list[types.Content]:
        """Typed contents for a request (decodes base64 thought signatures)."""
        return [types.Content.model_validate(m) for m in self.snapshot()]

    def __len__(self) -> int:
        return len(self.messages)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"messages": self.messages, "compactions": self.compactions},
                                ensure_ascii=False, indent=1), encoding="utf-8")
        _log(f"saved {len(self.messages)} message(s) to {p}")

    @classmethod
    def load(cls, path: str | Path) -> "History":
        p = Path(path)
        if not p.exists():
            _log(f"no history at {p}; starting fresh")
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        h = cls(data.get("messages", []))
        h.compactions = data.get("compactions", 0)
        _log(f"loaded {len(h.messages)} message(s) from {p}")
        return h
