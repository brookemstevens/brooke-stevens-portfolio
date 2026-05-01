"""Tool definitions and dispatch.

Tools are declared in the Bedrock Converse `toolSpec` format. Each tool is
backed by a plain Python function; `dispatch_tool` routes a tool-use block
from the model to the right implementation.

Design notes:
  - We keep the recipe objects slim before handing them back to the model.
    Raw Spoonacular responses include image URLs, thumbnails, and metadata
    that burn tokens without helping the planner.
  - All tools take the user_id implicitly through dispatch, so the model
    can't accidentally (or maliciously) act on another user's data.
"""
from typing import Any
import base64
import json

from db import (
    add_disliked_recipe,
    add_pantry_item,
    clear_preferences,
    get_cached_recipe,
    get_disliked_recipe_ids,
    get_disliked_recipes,
    get_meal_plan,
    get_pantry,
    get_preferences,
    remove_disliked_recipe,
    remove_pantry_item,
    replace_preferences,
    save_cached_recipe,
    save_meal_plan,
    update_preferences,
)
from spoonacular import (
    SpoonacularError,
    complex_search,
    find_by_ingredients,
    get_recipe,
)
import packages

# ---------------------------------------------------------------------------
# Tool schemas (Bedrock Converse format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "toolSpec": {
            "name": "get_pantry",
            "description": (
                "Return the list of ingredients the user currently has in their "
                "kitchen, with quantities and expiration dates if known. Call "
                "this before suggesting recipes."
            ),
            "inputSchema": {"json": {"type": "object", "properties": {}}},
        }
    },
    {
        "toolSpec": {
            "name": "update_pantry",
            "description": (
                "Add, update, or remove items in the user's pantry. Use 'add' "
                "or 'set' when the user says they bought or have something. "
                "Use 'remove' when they finish or discard an item."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["add", "remove", "set"],
                        },
                        "item": {
                            "type": "string",
                            "description": "Ingredient name, e.g. 'chicken breast'",
                        },
                        "quantity": {
                            "type": "string",
                            "description": "Optional, e.g. '1 lb', '2 cups', '3'",
                        },
                        "expires": {
                            "type": "string",
                            "description": "Optional ISO date like '2026-05-01'",
                        },
                    },
                    "required": ["action", "item"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "get_preferences",
            "description": (
                "Return the user's stored preferences: liked/disliked foods, "
                "dietary restrictions, calorie target for dinner, cooking days, "
                "and any other persistent notes."
            ),
            "inputSchema": {"json": {"type": "object", "properties": {}}},
        }
    },
    {
        "toolSpec": {
            "name": "update_preferences",
            "description": (
                "Modify the user's stored preferences. Three modes:\n"
                "  - mode='merge' (default): shallow-merges the `updates` dict into "
                "existing preferences. Any top-level key you include replaces its "
                "whole value (e.g. passing `dislikes` replaces the full list).\n"
                "  - mode='replace': wipes all existing preferences and stores "
                "`updates` as the complete new preferences dict.\n"
                "  - mode='clear': deletes all preferences entirely.\n"
                "\n"
                "SEMANTICS — read carefully:\n"
                "  - `dislikes` is the HARD constraint: any ingredient in this "
                "list must NEVER appear in a suggested recipe.\n"
                "  - `favorites` is a SOFT signal: ingredients the user "
                "especially enjoys. Recipes featuring favorites should be "
                "preferred but NOT required. Any ingredient that is not in "
                "`dislikes` is acceptable by default — the absence of an "
                "ingredient from `favorites` does NOT mean the user dislikes "
                "it. Do not treat `favorites` as a whitelist.\n"
                "\n"
                "STRUCTURE RULES — follow these exactly:\n"
                "  1. Use ONLY these top-level keys: favorites, dislikes, diet, "
                "calorie_target_min, calorie_target_max, cooking_days, appliances, "
                "household_size, spice_threshold, notes.\n"
                "  2. `favorites` and `dislikes` MUST be FLAT lists of ingredient "
                "strings. Do NOT nest by category. Correct: "
                "`dislikes: ['bell peppers', 'mushrooms', 'blue cheese']`. "
                "WRONG: `dislikes: {vegetables: [...], dairy: [...]}` — never "
                "do this, it causes partial-update bugs.\n"
                "  3. `appliances` is a flat list of strings.\n"
                "  4. Anything that doesn't fit a defined key (texture rules, "
                "household notes, spice context, cooking style prose) goes in "
                "`notes` as free text — do NOT create new top-level keys for it.\n"
                "  5. To change one ingredient in `dislikes` or `favorites`, "
                "pass the full updated list. The merge replaces the whole list, "
                "so partial lists would drop items.\n"
                "\n"
                "If an update call fails, surface the actual error to the user. "
                "Do not invent alternate keys like 'dislikes_proteins' as a "
                "workaround — those create stale duplicate data."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["merge", "replace", "clear"],
                            "description": "Default 'merge' if omitted.",
                        },
                        "updates": {
                            "type": "object",
                            "description": (
                                "Required for merge/replace. For clear mode, "
                                "omit or pass empty object."
                            ),
                        },
                    },
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "search_recipes_by_ingredients",
            "description": (
                "Find recipes that use a given list of ingredients, ranked to "
                "minimize additional ingredients needed. Use this FIRST when "
                "planning, passing in what's already in the pantry."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "ingredients": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Ingredients available to use.",
                        },
                        "number": {
                            "type": "integer",
                            "description": "How many recipes to return (default 5).",
                        },
                    },
                    "required": ["ingredients"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "search_recipes",
            "description": (
                "Open-ended recipe search with filters. Use this when the user "
                "asks for something specific ('quick weeknight pasta', 'high "
                "protein vegetarian dinner under 600 cal') that's not driven "
                "by what's already in the pantry."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "cuisine": {"type": "string"},
                        "diet": {
                            "type": "string",
                            "description": (
                                "One of: gluten free, ketogenic, vegetarian, "
                                "vegan, pescetarian, paleo, primal, whole30"
                            ),
                        },
                        "exclude_ingredients": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "max_ready_time": {"type": "integer"},
                        "min_calories": {"type": "integer"},
                        "max_calories": {"type": "integer"},
                        "number": {"type": "integer"},
                    },
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "get_recipe_details",
            "description": (
                "Full details for one recipe: ingredient list with amounts, "
                "instructions, nutrition. Call this after the user picks a "
                "recipe from a search result."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "recipe_id": {
                            "type": "string",
                            "description": "Recipe ID. Spoonacular IDs come back as integers in search results — pass them as strings here (e.g. '649030'). Custom-cookbook recipes use prefixed IDs like 'cookbook_carbonara'.",
                        }
                    },
                    "required": ["recipe_id"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "save_meal_plan",
            "description": (
                "Persist a confirmed meal plan for a given week so it can be "
                "recalled later. Call this after the user confirms the plan."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "week_start": {
                            "type": "string",
                            "description": "ISO date for the Monday of the week, e.g. '2026-04-27'",
                        },
                        "plan": {
                            "type": "object",
                            "description": (
                                "Structure it however is most useful, but a "
                                "reasonable shape is "
                                "{day: {recipe_id, title, notes}} plus a "
                                "'shopping_list' key grouped by category."
                            ),
                        },
                    },
                    "required": ["week_start", "plan"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "get_meal_plan",
            "description": "Retrieve a previously saved meal plan for a given week.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "week_start": {"type": "string"}
                    },
                    "required": ["week_start"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "mark_recipe_disliked",
            "description": (
                "Record that the user disliked a recipe so it won't be "
                "suggested again. Call this when the user reacts negatively "
                "to a recipe you suggested, or when they mention having tried "
                "something they didn't enjoy. The recipe_id must come from a "
                "previous search result or recipe details call (Spoonacular "
                "integer IDs passed as strings, or 'cookbook_*' IDs for "
                "user-added recipes). Optional reason is free text (e.g. "
                "'too bland', 'too much work', 'didn't like the texture') — "
                "include it when the user gives a reason so the agent can "
                "learn patterns."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "recipe_id": {"type": "string"},
                        "title": {
                            "type": "string",
                            "description": "Recipe title, for easy reference later.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Optional short explanation from the user.",
                        },
                    },
                    "required": ["recipe_id"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "unmark_recipe_disliked",
            "description": (
                "Remove a recipe from the disliked list. Use if the user "
                "changes their mind or it was marked in error."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {"recipe_id": {"type": "string"}},
                    "required": ["recipe_id"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "get_disliked_recipes",
            "description": (
                "Return the full list of recipes the user has marked as "
                "disliked, with titles and reasons if stored. Search results "
                "already filter these out automatically, so call this only "
                "when the user wants to review the list or when you need to "
                "reference a reason in your response."
            ),
            "inputSchema": {"json": {"type": "object", "properties": {}}},
        }
    },
    {
        "toolSpec": {
            "name": "get_realistic_package_sizes",
            "description": (
                "Given a list of ingredients, return realistic grocery-store "
                "package sizes and prices. Data sources: (1) Kroger Products "
                "API live prices for the user's local store — accurate and "
                "current — or (2) a curated fallback catalog of approximate "
                "pricing when Kroger returns no match. Each returned item "
                "includes a `source` field: 'kroger' means real live price; "
                "'curated' means estimate. For 'unknown' items, neither "
                "source had data; estimate those yourself and note the "
                "uncertainty.\n"
                "\n"
                "Use this BEFORE presenting a shopping list so you can "
                "(1) show quantities the way the user will actually buy them "
                "('1 lb chicken breast' not '6 oz'), and (2) produce a real "
                "cost estimate. When Kroger provides a specific product, "
                "you may name it (e.g. 'Simple Truth Organic Chicken Breast "
                "1.5 lb, $9.89') so the user can find it in-store."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "ingredients": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Ingredient names, simple form (e.g. 'chicken breast', 'lemon', 'olive oil').",
                        }
                    },
                    "required": ["ingredients"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "format_shopping_list",
            "description": (
                "Render a shopping list as a structured component in the UI. "
                "Use this INSTEAD of writing a markdown table whenever you "
                "have a shopping list to present — the frontend renders this "
                "as a proper table. Never author markdown tables for shopping "
                "lists; they break visually.\n"
                "\n"
                "The tool returns a token string like "
                "'[[SHOPPING_LIST:abc123]]'. Include this token verbatim in "
                "your reply, on its own line, where you want the table to "
                "appear. Do NOT decode it, modify it, or describe it as a "
                "token — the user will see a rendered table in its place. "
                "You can write natural-language prose before and after the "
                "token (e.g. an intro sentence, shopping notes, leftover "
                "tips) but never a markdown table for the same data."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "sections": {
                            "type": "array",
                            "description": "Grouped rows (Protein, Produce, Pantry, etc.).",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {
                                        "type": "string",
                                        "description": "Section name e.g. 'Protein', 'Produce', 'Pantry'",
                                    },
                                    "emoji": {
                                        "type": "string",
                                        "description": "Optional single emoji for the section header.",
                                    },
                                    "items": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "item": {"type": "string", "description": "Generic name (e.g. 'chicken breast')."},
                                                "product": {"type": "string", "description": "Specific product/brand if known (e.g. 'Simple Truth Natural Boneless Skinless Chicken Breast'). Omit if unknown."},
                                                "qty": {"type": "string", "description": "Purchase quantity (e.g. '1 lb', '1 head', '1 each')."},
                                                "price": {"type": "string", "description": "Price string (e.g. '$6.99', '~$8-12 est.')."},
                                                "note": {"type": "string", "description": "Optional short caveat (e.g. 'grab from seafood counter; no packaged match')."},
                                            },
                                            "required": ["item"],
                                        },
                                    },
                                },
                                "required": ["title", "items"],
                            },
                        },
                        "totals": {
                            "type": "array",
                            "description": "Summary rows below the sections. Each is a label and a value (e.g. 'Confirmed QFC prices' → '$26.27', 'Grand total estimate' → '~$40-41').",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string"},
                                    "value": {"type": "string"},
                                    "emphasis": {
                                        "type": "boolean",
                                        "description": "Set true for the grand-total row so it's visually emphasized.",
                                    },
                                },
                                "required": ["label", "value"],
                            },
                        },
                        "store": {
                            "type": "string",
                            "description": "Optional store-name caption at the top (e.g. 'QFC - Redmond, WA').",
                        },
                        "note": {
                            "type": "string",
                            "description": "Optional one-line note shown at the very top of the component (e.g. a leftovers tip).",
                        },
                    },
                    "required": ["sections"],
                }
            },
        }
    },
]


# ---------------------------------------------------------------------------
# Response slimming
# ---------------------------------------------------------------------------

def _stringify_id(rid) -> str | None:
    """Spoonacular returns IDs as integers; we standardize on strings everywhere
    so they can coexist with custom 'cookbook_*' IDs and so disliked-set
    membership checks aren't tripped by int/str mismatches."""
    if rid is None:
        return None
    return str(rid)


def _slim_ingredient_search_result(r: dict) -> dict:
    return {
        "id": _stringify_id(r.get("id")),
        "title": r.get("title"),
        "used_ingredients": [i.get("name") for i in r.get("usedIngredients", [])],
        "missed_ingredients": [i.get("name") for i in r.get("missedIngredients", [])],
        "missed_count": r.get("missedIngredientCount", 0),
        "likes": r.get("likes", 0),
    }


def _slim_complex_search_result(r: dict) -> dict:
    out = {
        "id": _stringify_id(r.get("id")),
        "title": r.get("title"),
        "ready_in_minutes": r.get("readyInMinutes"),
        "servings": r.get("servings"),
    }
    nutrition = r.get("nutrition") or {}
    for n in nutrition.get("nutrients", []):
        if n.get("name") == "Calories":
            out["calories"] = n.get("amount")
            break
    return out


def _slim_recipe_details(r: dict) -> dict:
    return {
        "id": _stringify_id(r.get("id")),
        "title": r.get("title"),
        "ready_in_minutes": r.get("readyInMinutes"),
        "servings": r.get("servings"),
        "ingredients": [
            {
                "name": ing.get("name"),
                "amount": ing.get("amount"),
                "unit": ing.get("unit"),
                "original": ing.get("original"),
            }
            for ing in r.get("extendedIngredients", [])
        ],
        "instructions": r.get("instructions"),
        "source_url": r.get("sourceUrl"),
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def dispatch_tool(name: str, inp: dict, user_id: str) -> dict[str, Any]:
    """Route a tool-use block to the right implementation. Raises on unknown tool."""
    if name == "get_pantry":
        return {"pantry": get_pantry(user_id)}

    if name == "update_pantry":
        action = inp["action"]
        item = inp["item"]
        if action in ("add", "set"):
            add_pantry_item(user_id, item, inp.get("quantity"), inp.get("expires"))
            return {"status": "ok", "action": action, "item": item}
        if action == "remove":
            remove_pantry_item(user_id, item)
            return {"status": "ok", "action": "remove", "item": item}
        raise ValueError(f"Unknown pantry action: {action}")

    if name == "get_preferences":
        return {"preferences": get_preferences(user_id)}

    if name == "update_preferences":
        mode = inp.get("mode", "merge")
        updates = inp.get("updates", {})
        if mode == "clear":
            clear_preferences(user_id)
            return {"status": "ok", "preferences": {}}
        if mode == "replace":
            prefs = replace_preferences(user_id, updates)
            return {"status": "ok", "preferences": prefs}
        # Default: merge
        updated = update_preferences(user_id, updates)
        return {"status": "ok", "preferences": updated}

    if name == "search_recipes_by_ingredients":
        try:
            results = find_by_ingredients(
                inp["ingredients"], number=inp.get("number", 5)
            )
        except SpoonacularError as e:
            return {"error": str(e)}
        slim = [_slim_ingredient_search_result(r) for r in results]
        disliked = get_disliked_recipe_ids(user_id)
        kept = [r for r in slim if r.get("id") not in disliked]
        return {
            "recipes": kept,
            "_filtered_count": len(slim) - len(kept),
        }

    if name == "search_recipes":
        try:
            results = complex_search(
                query=inp.get("query"),
                cuisine=inp.get("cuisine"),
                diet=inp.get("diet"),
                exclude_ingredients=inp.get("exclude_ingredients"),
                max_ready_time=inp.get("max_ready_time"),
                min_calories=inp.get("min_calories"),
                max_calories=inp.get("max_calories"),
                number=inp.get("number", 5),
            )
        except SpoonacularError as e:
            return {"error": str(e)}
        slim = [_slim_complex_search_result(r) for r in results.get("results", [])]
        disliked = get_disliked_recipe_ids(user_id)
        kept = [r for r in slim if r.get("id") not in disliked]
        return {
            "recipes": kept,
            "_filtered_count": len(slim) - len(kept),
        }

    if name == "get_recipe_details":
        recipe_id = str(inp["recipe_id"])

        # Cache check first. Recipe content never changes, so cached data
        # is always valid. Zero API points, ~10ms DynamoDB read.
        cached = get_cached_recipe(recipe_id)
        if cached is not None:
            cached["_cached"] = True  # lets observability distinguish hits
            return cached

        # Custom IDs (anything non-numeric, like 'cookbook_*') never come
        # from Spoonacular — there's no fallback to fetch them. If they
        # aren't in cache, they don't exist.
        if not recipe_id.isdigit():
            return {
                "error": (
                    f"Recipe '{recipe_id}' not found in cache. Custom "
                    "(non-Spoonacular) recipes must be seeded into the "
                    "cache directly; they cannot be fetched from any API."
                )
            }

        # Cache miss with a numeric (Spoonacular) ID — hit the API.
        try:
            r = get_recipe(recipe_id)
        except SpoonacularError as e:
            return {"error": str(e)}
        slim = _slim_recipe_details(r)
        try:
            save_cached_recipe(recipe_id, slim)
        except Exception:
            # Cache write failures should never block a request. The user
            # still gets their recipe; the next call just re-fetches.
            pass
        return slim

    if name == "save_meal_plan":
        save_meal_plan(user_id, inp["week_start"], inp["plan"])
        return {"status": "ok"}

    if name == "get_meal_plan":
        return {"plan": get_meal_plan(user_id, inp["week_start"])}

    if name == "mark_recipe_disliked":
        rid = str(inp["recipe_id"])
        add_disliked_recipe(
            user_id,
            rid,
            title=inp.get("title"),
            reason=inp.get("reason"),
        )
        return {"status": "ok", "recipe_id": rid}

    if name == "unmark_recipe_disliked":
        rid = str(inp["recipe_id"])
        remove_disliked_recipe(user_id, rid)
        return {"status": "ok", "recipe_id": rid}

    if name == "get_disliked_recipes":
        return {"disliked": get_disliked_recipes(user_id)}

    if name == "get_realistic_package_sizes":
        return packages.lookup_many(inp["ingredients"])

    if name == "format_shopping_list":
        # Encode the payload as base64 so the opaque token the model passes
        # back can't collide with anything in natural-language prose.
        payload = json.dumps(inp, separators=(",", ":"))
        encoded = base64.b64encode(payload.encode("utf-8")).decode("ascii")
        token = f"[[SHOPPING_LIST:{encoded}]]"
        return {
            "token": token,
            "instruction": (
                "Include the token string above verbatim in your reply on "
                "its own line where the table should appear. Do not author "
                "a markdown table — the frontend will render the token as "
                "a styled table."
            ),
        }

    raise ValueError(f"Unknown tool: {name}")
