"""Tool definitions: the JSON schemas advertised to the model and the Python
functions that execute each tool (e.g. searching the recipes database), plus
dispatch from a tool-use request to its handler.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from google.genai import types

from shared import (
    fetch_external_nutrition_info,
    save_user_preference_or_recipe,
    search_recipes,
)

_TAG = "[API/harness/tools.py]"


def _log(msg: str) -> None:
    print(f"{_TAG} {msg}", flush=True)


# Order and content are fixed so the request prefix stays identical across calls
# (lets Gemini's implicit prompt cache hit).
TOOL_DECLARATIONS: list[dict[str, Any]] = [
    {
        "name": "search_recipes",
        "description": (
            "Search the internal recipe database by keywords (dish name, ingredients) and an "
            "optional tag such as vegan, vegetarian, gluten-free, breakfast, dinner, quick, "
            "high-protein. Returns up to 3 full recipes, best match first."
        ),
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keywords; may be empty when filtering by tag."},
                "tag": {"type": "string", "description": "Optional tag filter."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "save_user_preference_or_recipe",
        "description": (
            "Persist something about the user to long-term memory. Use type 'preference' with "
            "key/value for stable facts (diet, allergies, dislikes, skill level), 'recipe' with "
            "a recipe object to save a recipe the user wants to keep, or 'note' with text for "
            "other free-form facts."
        ),
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["preference", "recipe", "note"]},
                "key": {"type": "string", "description": "Preference name, e.g. 'diet'."},
                "value": {"type": "string", "description": "Preference value, e.g. 'vegetarian'."},
                "text": {"type": "string", "description": "Free-form note text."},
                "recipe": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "title": {"type": "string"},
                        "ingredients": {"type": "array", "items": {"type": "string"}},
                        "instructions": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "prep_time": {"type": "integer"},
                    },
                    "required": ["title"],
                },
            },
            "required": ["type"],
        },
    },
    {
        "name": "fetch_external_nutrition_info",
        "description": (
            "Look up nutrition facts (calories, protein, carbs, fat, fiber; per 100g unless "
            "noted) for an ingredient or dish from the external nutrition service."
        ),
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "ingredient_or_dish": {"type": "string", "description": "Ingredient or dish name."}
            },
            "required": ["ingredient_or_dish"],
        },
    },
]

HANDLERS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "search_recipes": lambda args: search_recipes(args.get("query", ""), args.get("tag") or None),
    "save_user_preference_or_recipe": lambda args: save_user_preference_or_recipe(dict(args)),
    "fetch_external_nutrition_info": lambda args: fetch_external_nutrition_info(args.get("ingredient_or_dish", "")),
}


def get_tools() -> list[types.Tool]:
    _log(f"advertising {len(TOOL_DECLARATIONS)} function declarations")
    return [types.Tool(function_declarations=[types.FunctionDeclaration(**d) for d in TOOL_DECLARATIONS])]


def parse_tool_calls(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract function_call payloads from a model message's parts."""
    calls = [p["function_call"] for p in parts if p.get("function_call")]
    _log(f"parsed {len(calls)} function call(s): {[c.get('name') for c in calls]}")
    return calls


def _response_part(call: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    fr: dict[str, Any] = {"name": call.get("name"), "response": response}
    if call.get("id"):
        fr["id"] = call["id"]
    return {"function_response": fr}


def _error(call: dict[str, Any], message: str) -> tuple[dict[str, Any], bool]:
    # Gemini has no is_error flag; the convention is an {"error": ...} response payload.
    _log(f"tool error: {message}")
    return _response_part(call, {"error": message}), True


def execute_tool_call(call: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Run one function call; returns (function_response part, is_error)."""
    name, args = call.get("name"), call.get("args") or {}
    _log(f"dispatching {name} args={json.dumps(args, ensure_ascii=False)}")
    handler = HANDLERS.get(name)
    if handler is None:
        return _error(call, f"Unknown tool '{name}'. Available: {', '.join(HANDLERS)}")
    if not isinstance(args, dict):
        return _error(call, "Tool arguments must be an object.")
    try:
        result = handler(args)
    except Exception as exc:  # report to the model rather than crash the loop
        return _error(call, f"{type(exc).__name__}: {exc}")
    if isinstance(result, dict) and result.get("ok") is False:
        return _error(call, result.get("error", "failed"))
    _log(f"{name} ok ({len(json.dumps(result))} chars)")
    return _response_part(call, {"result": result}), False


def execute_tool_calls(calls: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[bool]]:
    """Run every call from one model turn; all responses go back in one user message."""
    pairs = [execute_tool_call(c) for c in calls]
    return [p for p, _ in pairs], [e for _, e in pairs]
