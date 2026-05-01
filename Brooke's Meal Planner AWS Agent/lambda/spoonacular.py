"""Spoonacular API client.

Three endpoints cover the use case:
  - findByIngredients: pantry-first search ("what can I make with what I have?")
  - complexSearch: open-ended discovery with filters (cuisine, diet, calories)
  - recipeInformation: full recipe detail once the user picks one

Docs: https://spoonacular.com/food-api/docs
Free tier: ~150 requests/day.
"""
import os
from typing import Optional

import requests

API_KEY = os.environ.get("SPOONACULAR_API_KEY", "")
BASE = "https://api.spoonacular.com"
TIMEOUT = 10


class SpoonacularError(Exception):
    pass


def _get(path: str, params: dict) -> dict:
    if not API_KEY:
        raise SpoonacularError("SPOONACULAR_API_KEY not configured")
    params = {**params, "apiKey": API_KEY}
    r = requests.get(f"{BASE}{path}", params=params, timeout=TIMEOUT)
    if r.status_code == 402:
        raise SpoonacularError("Spoonacular daily quota exceeded")
    if not r.ok:
        raise SpoonacularError(f"Spoonacular {r.status_code}: {r.text[:200]}")
    return r.json()


def find_by_ingredients(
    ingredients: list[str],
    number: int = 5,
    ranking: int = 2,
) -> list[dict]:
    """Find recipes that use the given ingredients.

    ranking=1 maximizes used ingredients; ranking=2 minimizes missing ones.
    For waste-minimizing planning we default to ranking=2.
    """
    data = _get(
        "/recipes/findByIngredients",
        {
            "ingredients": ",".join(ingredients),
            "number": number,
            "ranking": ranking,
            "ignorePantry": "true",
        },
    )
    return data if isinstance(data, list) else []


def complex_search(
    query: Optional[str] = None,
    cuisine: Optional[str] = None,
    diet: Optional[str] = None,
    exclude_ingredients: Optional[list[str]] = None,
    max_ready_time: Optional[int] = None,
    min_calories: Optional[int] = None,
    max_calories: Optional[int] = None,
    number: int = 5,
) -> dict:
    """Open-ended recipe search with filters."""
    params: dict = {"number": number, "addRecipeNutrition": "true"}
    if query:
        params["query"] = query
    if cuisine:
        params["cuisine"] = cuisine
    if diet:
        params["diet"] = diet
    if exclude_ingredients:
        params["excludeIngredients"] = ",".join(exclude_ingredients)
    if max_ready_time:
        params["maxReadyTime"] = max_ready_time
    if min_calories:
        params["minCalories"] = min_calories
    if max_calories:
        params["maxCalories"] = max_calories
    return _get("/recipes/complexSearch", params)


def get_recipe(recipe_id: str) -> dict:
    """Full recipe information for a specific recipe."""
    return _get(
        f"/recipes/{recipe_id}/information",
        {"includeNutrition": "true"},
    )
