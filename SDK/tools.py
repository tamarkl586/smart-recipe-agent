"""Tool definitions for the Google GenAI SDK: plain typed Python functions
wrapping the shared recipe functions. The SDK builds the function declarations
from the signatures/docstrings and, with automatic function calling, executes
them itself.
"""

from typing import Any, Optional

from pydantic import BaseModel

from shared import (
    fetch_external_nutrition_info as _fetch_nutrition,
    save_user_preference_or_recipe as _save,
    search_recipes as _search,
)


class Recipe(BaseModel):
    title: str
    id: Optional[str] = None
    ingredients: list[str] = []
    instructions: str = ""
    tags: list[str] = []
    prep_time: Optional[int] = None


# replaces: tools.py (schemas come from these signatures + docstrings; AFC dispatches to them)
def search_recipes(query: str, tag: Optional[str] = None) -> dict[str, Any]:
    """Search the internal recipe database by keywords (dish name, ingredients) and an
    optional tag such as vegan, vegetarian, gluten-free, breakfast, dinner, quick,
    high-protein. Returns up to 3 full recipes, best match first.

    Args:
        query: Keywords; may be empty when filtering by tag.
        tag: Optional tag filter.
    """
    return {"result": _search(query, tag or None)}


def save_user_preference_or_recipe(
    type: str,
    key: Optional[str] = None,
    value: Optional[str] = None,
    text: Optional[str] = None,
    recipe: Optional[Recipe] = None,
) -> dict[str, Any]:
    """Persist something about the user to long-term memory. Use type 'preference' with
    key/value for stable facts (diet, allergies, dislikes, skill level), 'recipe' with a
    recipe object to save a recipe the user wants to keep, or 'note' with text for other
    free-form facts.

    Args:
        type: One of 'preference', 'recipe', 'note'.
        key: Preference name, e.g. 'diet'.
        value: Preference value, e.g. 'vegetarian'.
        text: Free-form note text.
        recipe: The recipe to save (title required).
    """
    data: dict[str, Any] = {"type": type, "key": key, "value": value, "text": text}
    if recipe is not None:
        data["recipe"] = recipe.model_dump(exclude_none=True)
    result = _save({k: v for k, v in data.items() if v is not None})
    # replaces: tools.py (error formatting: AFC sends {"error": ...} back to the model)
    return {"error": result["error"]} if result.get("ok") is False else {"result": result}


def fetch_external_nutrition_info(ingredient_or_dish: str) -> dict[str, Any]:
    """Look up nutrition facts (calories, protein, carbs, fat, fiber; per 100g unless
    noted) for an ingredient or dish from the external nutrition service.

    Args:
        ingredient_or_dish: Ingredient or dish name.
    """
    return {"result": _fetch_nutrition(ingredient_or_dish)}


TOOLS = [search_recipes, save_user_preference_or_recipe, fetch_external_nutrition_info]
TOOL_NAMES = [f.__name__ for f in TOOLS]
