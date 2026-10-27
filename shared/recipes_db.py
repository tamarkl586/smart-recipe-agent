"""Domain logic shared by the API and SDK agents.

Provides a small in-memory recipe database, the three tool functions the agent
exposes, and helpers for persisting user memory/preferences to a JSON file.
All tool functions return compact, JSON-serializable dicts/lists to keep the
token cost of tool results low.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
NUTRITION_PATH = _HERE / "nutrition_data.json"
DEFAULT_MEMORY_PATH = _HERE.parent / "memory.json"

MAX_SEARCH_RESULTS = 3
MAX_MEMORY_ITEMS = 50

RECIPES: list[dict[str, Any]] = [
    {
        "id": "r1",
        "title": "Shakshuka",
        "ingredients": ["egg", "tomato", "onion", "bell pepper", "garlic", "cumin", "olive oil"],
        "instructions": "Saute onion, pepper and garlic in oil. Add tomatoes and cumin, simmer 10 min. "
                        "Make wells, crack in eggs, cover and cook until set.",
        "tags": ["vegetarian", "breakfast", "gluten-free"],
        "prep_time": 25,
    },
    {
        "id": "r2",
        "title": "Classic Hummus",
        "ingredients": ["chickpeas", "tahini", "lemon", "garlic", "olive oil", "salt"],
        "instructions": "Blend chickpeas, tahini, lemon juice, garlic and salt until smooth, adding "
                        "cold water as needed. Drizzle with olive oil.",
        "tags": ["vegan", "snack", "gluten-free"],
        "prep_time": 10,
    },
    {
        "id": "r3",
        "title": "Chicken Stir Fry",
        "ingredients": ["chicken breast", "broccoli", "bell pepper", "soy sauce", "ginger", "garlic", "rice"],
        "instructions": "Slice chicken and stir-fry until browned. Add vegetables, ginger and garlic, "
                        "cook 4 min. Toss with soy sauce and serve over rice.",
        "tags": ["dinner", "high-protein", "quick"],
        "prep_time": 20,
    },
    {
        "id": "r4",
        "title": "Red Lentil Soup",
        "ingredients": ["lentils", "carrot", "onion", "cumin", "vegetable stock", "lemon"],
        "instructions": "Saute onion and carrot, add lentils, cumin and stock. Simmer 20 min, blend, "
                        "finish with lemon juice.",
        "tags": ["vegan", "dinner", "gluten-free"],
        "prep_time": 30,
    },
    {
        "id": "r5",
        "title": "Tomato Basil Pasta",
        "ingredients": ["pasta", "tomato", "basil", "garlic", "olive oil", "parmesan"],
        "instructions": "Cook pasta. Meanwhile saute garlic in oil, add chopped tomatoes and cook 5 min. "
                        "Toss with pasta, basil and parmesan.",
        "tags": ["vegetarian", "dinner", "quick"],
        "prep_time": 20,
    },
    {
        "id": "r6",
        "title": "Baked Salmon with Spinach",
        "ingredients": ["salmon", "spinach", "lemon", "garlic", "olive oil"],
        "instructions": "Bake seasoned salmon at 200C for 12 min. Wilt spinach with garlic in oil. "
                        "Serve with lemon.",
        "tags": ["dinner", "high-protein", "gluten-free", "pescatarian"],
        "prep_time": 20,
    },
    {
        "id": "r7",
        "title": "Overnight Oats",
        "ingredients": ["oats", "milk", "banana", "honey", "cinnamon"],
        "instructions": "Mix oats, milk, honey and cinnamon in a jar. Refrigerate overnight. "
                        "Top with sliced banana.",
        "tags": ["vegetarian", "breakfast", "quick"],
        "prep_time": 5,
    },
    {
        "id": "r8",
        "title": "Crispy Tofu Bowl",
        "ingredients": ["tofu", "rice", "avocado", "soy sauce", "sesame", "cucumber"],
        "instructions": "Press and cube tofu, pan-fry until crisp, glaze with soy sauce. Serve over rice "
                        "with avocado, cucumber and sesame.",
        "tags": ["vegan", "dinner", "high-protein"],
        "prep_time": 25,
    },
]


# --------------------------------------------------------------------------- #
# Memory persistence
# --------------------------------------------------------------------------- #

def _memory_path(path: str | os.PathLike | None = None) -> Path:
    return Path(path or os.environ.get("RECIPE_MEMORY_PATH") or DEFAULT_MEMORY_PATH)


def _empty_memory() -> dict[str, Any]:
    return {"preferences": {}, "saved_recipes": [], "notes": []}


def load_memory(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load user memory from disk; returns an empty structure if missing or corrupt."""
    p = _memory_path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return _empty_memory()
    memory = _empty_memory()
    if isinstance(data, dict):
        memory.update({k: data[k] for k in memory if k in data})
    return memory


def save_memory(memory: dict[str, Any], path: str | os.PathLike | None = None) -> None:
    """Atomically write user memory to disk."""
    p = _memory_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".memory-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #

def _all_recipes(path: str | os.PathLike | None = None) -> list[dict[str, Any]]:
    """Built-in recipes plus any the user has saved to memory."""
    return RECIPES + load_memory(path)["saved_recipes"]


def get_recipe(recipe_id: str) -> dict[str, Any] | None:
    """Return the full recipe for an id, or None."""
    return next((r for r in _all_recipes() if r.get("id") == recipe_id), None)


def search_recipes(query: str, tag: str | None = None) -> list[dict[str, Any]]:
    """Internal tool: keyword search over titles, ingredients and tags.

    Returns at most MAX_SEARCH_RESULTS full recipes, best match first. An empty
    query with a tag returns recipes with that tag.
    """
    terms = [t for t in (query or "").lower().split() if t]
    tag_l = tag.lower().strip() if tag else None
    scored = []
    for r in _all_recipes():
        tags = [t.lower() for t in r.get("tags", [])]
        if tag_l and tag_l not in tags:
            continue
        title = r.get("title", "").lower()
        ingredients = " ".join(r.get("ingredients", [])).lower()
        score = sum(3 * (t in title) + 2 * (t in ingredients) + (t in tags) for t in terms)
        if terms and score == 0:
            continue
        scored.append((score, r.get("prep_time", 0), r))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [
        {k: r.get(k) for k in ("id", "title", "ingredients", "instructions", "tags", "prep_time")}
        for _, _, r in scored[:MAX_SEARCH_RESULTS]
    ]


def save_user_preference_or_recipe(data: dict[str, Any]) -> dict[str, Any]:
    """Internal memory tool: persist a preference, a recipe, or a free-form note.

    Accepted shapes:
      {"type": "preference", "key": "diet", "value": "vegetarian"}
      {"type": "recipe", "recipe": {"title": ..., "ingredients": [...], ...}}
      {"type": "note", "text": "dislikes cilantro"}
    If "type" is omitted it is inferred from the keys present.
    """
    if not isinstance(data, dict):
        return {"ok": False, "error": "data must be an object"}
    kind = data.get("type") or (
        "recipe" if "recipe" in data else "preference" if "key" in data else "note" if "text" in data else None
    )
    memory = load_memory()

    if kind == "preference":
        key, value = data.get("key"), data.get("value")
        if not key or value is None:
            return {"ok": False, "error": "preference needs 'key' and 'value'"}
        memory["preferences"][str(key)] = value
        result = {"ok": True, "saved": "preference", "key": key, "value": value}

    elif kind == "recipe":
        recipe = data.get("recipe")
        if not isinstance(recipe, dict) or not recipe.get("title"):
            return {"ok": False, "error": "recipe needs a 'title'"}
        recipe = {
            "id": recipe.get("id") or f"u{len(memory['saved_recipes']) + 1}",
            "title": recipe["title"],
            "ingredients": list(recipe.get("ingredients", [])),
            "instructions": recipe.get("instructions", ""),
            "tags": list(recipe.get("tags", [])),
            "prep_time": recipe.get("prep_time"),
        }
        memory["saved_recipes"] = [r for r in memory["saved_recipes"] if r.get("id") != recipe["id"]]
        memory["saved_recipes"].append(recipe)
        memory["saved_recipes"] = memory["saved_recipes"][-MAX_MEMORY_ITEMS:]
        result = {"ok": True, "saved": "recipe", "id": recipe["id"], "title": recipe["title"]}

    elif kind == "note":
        text = str(data.get("text", "")).strip()
        if not text:
            return {"ok": False, "error": "note needs 'text'"}
        memory["notes"].append({"text": text, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        memory["notes"] = memory["notes"][-MAX_MEMORY_ITEMS:]
        result = {"ok": True, "saved": "note", "text": text}

    else:
        return {"ok": False, "error": "type must be 'preference', 'recipe' or 'note'"}

    save_memory(memory)
    return result


def _load_nutrition_source() -> dict[str, Any]:
    try:
        return json.loads(NUTRITION_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def fetch_external_nutrition_info(ingredient_or_dish: str) -> dict[str, Any]:
    """External tool (simulated): look up nutrition facts from a mock API source.

    Tries an exact match, then (for a known recipe title) the per-ingredient
    facts that are available, then a partial name match.
    """
    name = (ingredient_or_dish or "").strip().lower()
    if not name:
        return {"found": False, "error": "empty query"}
    source = _load_nutrition_source()
    src = source.get("_source", "mock")
    items = {k: v for k, v in source.items() if not k.startswith("_")}

    if name in items:
        return {"found": True, "name": name, "source": src, **items[name]}

    recipe = next((r for r in _all_recipes() if r.get("title", "").lower() == name), None)
    if recipe:
        per_ingredient = {i: items[i] for i in recipe.get("ingredients", []) if i in items}
        if per_ingredient:
            return {"found": True, "name": recipe["title"], "source": src,
                    "note": "per-ingredient values, per 100g", "ingredients": per_ingredient}

    match = next((k for k in items if k in name or name in k), None)
    if match:
        return {"found": True, "name": match, "source": src, **items[match]}

    return {"found": False, "name": name, "error": "no nutrition data"}
