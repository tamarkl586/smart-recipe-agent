"""Scenario helpers shared by the API and SDK runners: loading a multi-session
scenario, isolating persisted state per implementation, evaluating pass/fail
assertions against saved state, and collecting result metrics.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from .recipes_db import load_memory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = PROJECT_ROOT / "results" / "state"


def load_dotenv(path: str | Path = PROJECT_ROOT / ".env") -> None:
    """Load KEY=VALUE lines from the project .env without overriding the environment."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load_scenario(path: str | Path) -> dict[str, Any]:
    scenario = json.loads(Path(path).read_text(encoding="utf-8"))
    if "sessions" not in scenario and scenario.get("turns"):
        scenario["sessions"] = [{"name": "session_1", "turns": scenario["turns"]}]
    if not any(s.get("turns") for s in scenario.get("sessions", [])):
        raise ValueError(f"Scenario {path} has no sessions with turns")
    return scenario


def prepare_state(impl: str, scenario: dict[str, Any], state_dir: str | Path | None = None) -> dict[str, Any]:
    """Point shared memory at a per-implementation file and reset it if asked.

    Returns the paths used: the memory file and one state file per session.
    """
    state_dir = Path(state_dir or STATE_DIR)
    state_dir.mkdir(parents=True, exist_ok=True)
    memory_path = state_dir / f"{impl}_memory.json"
    os.environ["RECIPE_MEMORY_PATH"] = str(memory_path)
    if scenario.get("fresh_memory", True):
        memory_path.unlink(missing_ok=True)
    suffix = "history.json" if impl == "api" else "session_id.txt"
    session_files = {s["name"]: state_dir / f"{impl}_{s['name']}_{suffix}" for s in scenario["sessions"]}
    if scenario.get("fresh_memory", True):
        for p in session_files.values():
            p.unlink(missing_ok=True)
    return {"memory": memory_path, "sessions": session_files}


# --------------------------------------------------------------------------- #
# Assertions
# --------------------------------------------------------------------------- #

def _memory_strings(memory: dict[str, Any], section: str) -> list[str]:
    out: list[str] = []
    if section in ("preferences", "any"):
        out += [f"{k} {json.dumps(v)}" for k, v in memory.get("preferences", {}).items()]
    if section in ("notes", "any"):
        out += [n.get("text", "") for n in memory.get("notes", [])]
    if section in ("saved_recipes", "any"):
        out += [json.dumps(r) for r in memory.get("saved_recipes", [])]
    return [s.lower() for s in out]


def _session_turns(results: dict[str, Any], name: str | None) -> list[dict[str, Any]]:
    sessions = results["sessions"] if name is None else [s for s in results["sessions"] if s["name"] == name]
    return [t for s in sessions for t in s["turns"]]


def evaluate_assertion(a: dict[str, Any], results: dict[str, Any], paths: dict[str, Any]) -> dict[str, Any]:
    memory = load_memory(paths["memory"])
    kind = a["type"]
    detail = ""
    if kind == "memory_contains":
        found = [s for s in _memory_strings(memory, a.get("section", "any")) if a["value"].lower() in s]
        passed, detail = bool(found), (found[0] if found else "not found")
    elif kind == "memory_not_contains":
        found = [s for s in _memory_strings(memory, a.get("section", "any")) if a["value"].lower() in s]
        passed, detail = not found, (found[0] if found else "absent")
    elif kind == "saved_recipe_exists":
        titles = [r.get("title", "") for r in memory.get("saved_recipes", [])]
        match = [t for t in titles if a["title_contains"].lower() in t.lower()]
        passed, detail = bool(match), f"saved: {titles}"
    elif kind == "state_file_exists":
        p = Path(paths["sessions"][a["file"]])
        passed = p.exists() and p.stat().st_size > 0
        detail = str(p.relative_to(PROJECT_ROOT) if p.is_relative_to(PROJECT_ROOT) else p)
    elif kind == "memory_file_exists":
        p = Path(paths["memory"])
        passed, detail = p.exists() and p.stat().st_size > 0, p.name
    elif kind in ("tool_called", "tool_not_called"):
        names = [c["name"] for t in _session_turns(results, a.get("session")) for c in t["tool_calls"]
                 if not c.get("is_error")]
        called = a["tool"] in names
        passed = called if kind == "tool_called" else not called
        detail = f"tools: {names}"
    elif kind == "response_contains_any":
        turns = _session_turns(results, a.get("session"))
        idx = a.get("turn", len(turns)) - 1
        text = turns[idx]["text"].lower() if 0 <= idx < len(turns) else ""
        hits = [v for v in a["values"] if v.lower() in text]
        passed, detail = bool(hits), (f"matched {hits}" if hits else f"none of {a['values']}")
    else:
        passed, detail = False, f"unknown assertion type '{kind}'"
    return {"id": a.get("id"), "type": kind, "description": a.get("description", ""),
            "passed": passed, "detail": detail}


def evaluate_assertions(scenario: dict[str, Any], results: dict[str, Any], paths: dict[str, Any]) -> dict[str, Any]:
    checks = [evaluate_assertion(a, results, paths) for a in scenario.get("assertions", [])]
    for c in checks:
        print(f"[assert] {c['id']} {'PASS' if c['passed'] else 'FAIL'} - {c['description']} ({c['detail']})",
              flush=True)
    passed = sum(c["passed"] for c in checks)
    return {"passed": passed, "failed": len(checks) - passed, "total": len(checks), "checks": checks}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def line_counts(paths: list[str | Path]) -> dict[str, Any]:
    """Count total and non-blank, non-comment lines per file."""
    files = {}
    for p in sorted(Path(x) for x in paths):
        lines = p.read_text(encoding="utf-8").splitlines()
        code = [ln for ln in lines if ln.strip() and not ln.strip().startswith("#")]
        files[str(p.relative_to(PROJECT_ROOT))] = {"lines": len(lines), "code_lines": len(code)}
    return {
        "files": files,
        "total_lines": sum(f["lines"] for f in files.values()),
        "total_code_lines": sum(f["code_lines"] for f in files.values()),
    }


def tool_summary(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    calls = [c for s in sessions for t in s["turns"] for c in t["tool_calls"]]
    return {
        "total_calls": len(calls),
        "errors": sum(bool(c.get("is_error")) for c in calls),
        "by_tool": dict(Counter(c["name"] for c in calls)),
    }


TOKEN_KEYS = ("input_tokens", "output_tokens", "thinking_tokens", "cache_read_tokens",
              "cache_write_tokens", "model_calls", "estimated_cost_usd")


def sum_usage(usages: list[dict[str, Any]]) -> dict[str, Any]:
    total: dict[str, Any] = {k: 0 for k in TOKEN_KEYS}
    for u in usages:
        for k in TOKEN_KEYS:
            total[k] += u.get(k, 0) or 0
    total["estimated_cost_usd"] = round(total["estimated_cost_usd"], 6)
    total["usage_complete"] = all(u.get("usage_complete", True) for u in usages)
    prompt = total["input_tokens"] + total["cache_read_tokens"]
    total["cache_hit_pct"] = round(100 * total["cache_read_tokens"] / prompt, 1) if prompt else 0.0
    return total


def write_results(out: str | Path, results: dict[str, Any]) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return out
