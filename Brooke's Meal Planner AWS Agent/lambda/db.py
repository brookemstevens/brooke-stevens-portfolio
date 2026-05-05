"""DynamoDB data access.

Single-table design for a simple, low-cost setup:

    Table: grocery-agent
    PK (hash):  pk  = one of:
        "USER#<user_id>"     -> per-user data
        "CACHE"              -> shared cache rows (not user-specific)
    SK (range): sk  = one of:
        "PROFILE"              -> preferences (dict under `data`)
        "PANTRY#<item>"        -> pantry item (dict under `data`)
        "MEALPLAN#<date>"      -> weekly meal plan (dict under `data`)
        "DISLIKED#<recipe_id>" -> recipe the user rejected (dict under `data`)
        "CHAT"                 -> full chat history (JSON string under `messages_json`)
        "RECIPE#<id>"          -> cached Spoonacular recipe details (when pk=CACHE)

Chat history is JSON-serialized because Bedrock Converse messages contain
nested content blocks that are annoying to represent as native DynamoDB maps.
Simpler to store as a string.

Recipe cache is shared across users (pk="CACHE") because Spoonacular recipe
IDs are globally unique and recipe content never changes. Two users asking
for "recipe 12345" should hit the same cached row.
"""
import json
import os
import time
from typing import Any

import boto3
from boto3.dynamodb.conditions import Key

TABLE_NAME = os.environ.get("TABLE_NAME", "grocery-agent")
_dynamodb = boto3.resource("dynamodb")
_table = _dynamodb.Table(TABLE_NAME)


def _pk(user_id: str) -> str:
    return f"USER#{user_id}"


def _now() -> int:
    return int(time.time())


# --- Preferences ---------------------------------------------------------

def get_preferences(user_id: str) -> dict:
    resp = _table.get_item(Key={"pk": _pk(user_id), "sk": "PROFILE"})
    return resp.get("Item", {}).get("data", {})


def _deep_merge(dst: dict, src: dict) -> dict:
    """Recursively merge src into dst. Nested dicts are merged, not replaced.

    For lists/scalars, src wins (full replacement). This is the right behavior
    for preferences: if the user says "change nuts to [peanuts]", we want the
    whole nuts list replaced, not appended to.
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def update_preferences(user_id: str, updates: dict) -> dict:
    current = get_preferences(user_id)
    _deep_merge(current, updates)
    _table.put_item(
        Item={
            "pk": _pk(user_id),
            "sk": "PROFILE",
            "data": current,
            "updated_at": _now(),
        }
    )
    return current


def replace_preferences(user_id: str, new_prefs: dict) -> dict:
    """Full overwrite — blows away the existing profile and writes new_prefs.

    Use when the stored prefs have drifted or contain bad keys that a patch
    update can't reach.
    """
    _table.put_item(
        Item={
            "pk": _pk(user_id),
            "sk": "PROFILE",
            "data": new_prefs,
            "updated_at": _now(),
        }
    )
    return new_prefs


def clear_preferences(user_id: str) -> None:
    """Delete the profile row entirely. Next get_preferences returns {}."""
    _table.delete_item(Key={"pk": _pk(user_id), "sk": "PROFILE"})


# --- Pantry --------------------------------------------------------------

def get_pantry(user_id: str) -> list[dict]:
    resp = _table.query(
        KeyConditionExpression=Key("pk").eq(_pk(user_id)) & Key("sk").begins_with("PANTRY#")
    )
    items = []
    for row in resp.get("Items", []):
        item_name = row["sk"].replace("PANTRY#", "", 1)
        items.append({"item": item_name, **row.get("data", {})})
    return items


def add_pantry_item(
    user_id: str,
    item: str,
    quantity: str | None = None,
    expires: str | None = None,
) -> None:
    data: dict[str, Any] = {}
    if quantity:
        data["quantity"] = quantity
    if expires:
        data["expires"] = expires
    _table.put_item(
        Item={
            "pk": _pk(user_id),
            "sk": f"PANTRY#{item.strip().lower()}",
            "data": data,
            "updated_at": _now(),
        }
    )


def remove_pantry_item(user_id: str, item: str) -> None:
    _table.delete_item(
        Key={"pk": _pk(user_id), "sk": f"PANTRY#{item.strip().lower()}"}
    )


# --- Meal plans ----------------------------------------------------------

def save_meal_plan(user_id: str, week_start: str, plan: dict) -> None:
    _table.put_item(
        Item={
            "pk": _pk(user_id),
            "sk": f"MEALPLAN#{week_start}",
            "data": plan,
            "updated_at": _now(),
        }
    )


def get_meal_plan(user_id: str, week_start: str) -> dict:
    resp = _table.get_item(
        Key={"pk": _pk(user_id), "sk": f"MEALPLAN#{week_start}"}
    )
    return resp.get("Item", {}).get("data", {})


# --- Chat history --------------------------------------------------------

def load_history(user_id: str, limit: int = 20) -> list[dict]:
    resp = _table.get_item(Key={"pk": _pk(user_id), "sk": "CHAT"})
    raw = resp.get("Item", {}).get("messages_json")
    if not raw:
        return []
    messages = json.loads(raw)
    return messages[-limit:] if len(messages) > limit else messages


def save_history(user_id: str, messages: list[dict], cap: int = 30) -> None:
    """Save chat history, capped to roughly the last `cap` messages.

    The trim is "tool-aware": if the cut point would land in the middle of a
    tool_use → tool_result pair, we extend the cut backward to keep the
    matching tool_use with its tool_result. Bedrock rejects any history
    containing a toolUse without its matching toolResult, so blindly slicing
    the tail can permanently corrupt subsequent requests.
    """
    if len(messages) <= cap:
        trimmed = messages
    else:
        # Default cut point: keep the last `cap` messages.
        cut = len(messages) - cap
        # Collect tool IDs *kept* by the default cut.
        kept_tool_use_ids = set()
        kept_tool_result_ids = set()
        for msg in messages[cut:]:
            for block in msg.get("content", []) or []:
                if "toolUse" in block:
                    tu = block["toolUse"].get("toolUseId")
                    if tu:
                        kept_tool_use_ids.add(tu)
                elif "toolResult" in block:
                    tr = block["toolResult"].get("toolUseId")
                    if tr:
                        kept_tool_result_ids.add(tr)
        # If any tool_result in the kept window has no matching tool_use,
        # walk the cut point backward until each orphan finds its match.
        orphan_results = kept_tool_result_ids - kept_tool_use_ids
        while orphan_results and cut > 0:
            cut -= 1
            msg = messages[cut]
            for block in msg.get("content", []) or []:
                if "toolUse" in block:
                    tu = block["toolUse"].get("toolUseId")
                    if tu in orphan_results:
                        orphan_results.discard(tu)
                        kept_tool_use_ids.add(tu)
        trimmed = messages[cut:]

    _table.put_item(
        Item={
            "pk": _pk(user_id),
            "sk": "CHAT",
            "messages_json": json.dumps(trimmed),
            "updated_at": _now(),
        }
    )


# --- Disliked recipes -----------------------------------------------------
#
# One row per disliked recipe (same pattern as pantry items — no merge bugs).
# Storing title alongside the ID lets the agent reference the list without
# re-fetching each recipe. 'reason' is optional free text from the user.

def add_disliked_recipe(
    user_id: str,
    recipe_id: str,
    title: str | None = None,
    reason: str | None = None,
) -> None:
    data: dict[str, Any] = {}
    if title:
        data["title"] = title
    if reason:
        data["reason"] = reason
    _table.put_item(
        Item={
            "pk": _pk(user_id),
            "sk": f"DISLIKED#{recipe_id}",
            "data": data,
            "updated_at": _now(),
        }
    )


def remove_disliked_recipe(user_id: str, recipe_id: str) -> None:
    _table.delete_item(
        Key={"pk": _pk(user_id), "sk": f"DISLIKED#{recipe_id}"}
    )


def get_disliked_recipes(user_id: str) -> list[dict]:
    """Return all disliked recipes for the user as a list of dicts with
    recipe_id (string), title (if stored), and reason (if stored).
    """
    resp = _table.query(
        KeyConditionExpression=Key("pk").eq(_pk(user_id))
        & Key("sk").begins_with("DISLIKED#")
    )
    items = []
    for row in resp.get("Items", []):
        rid = row["sk"].replace("DISLIKED#", "", 1)
        items.append({"recipe_id": rid, **row.get("data", {})})
    return items


def get_disliked_recipe_ids(user_id: str) -> set[str]:
    """Faster path when we only need the set of IDs (for filtering).
    Returns string IDs — callers comparing against Spoonacular results
    must stringify the search-result IDs before doing `in` checks."""
    return {row["recipe_id"] for row in get_disliked_recipes(user_id)}



#
# Spoonacular recipe IDs are global and the recipe content doesn't change,
# so the cache is shared across all users (pk="CACHE").
#
# We store the recipe as a JSON string rather than a native DynamoDB map
# because ingredient amounts are floats (e.g. 1.5 cups) and DynamoDB's
# boto3 resource returns numbers as Decimal, which isn't JSON-serializable
# without extra plumbing. JSON-in, JSON-out sidesteps the whole issue.

def get_cached_recipe(recipe_id: str) -> dict | None:
    """Return the cached recipe dict, or None if not cached."""
    resp = _table.get_item(Key={"pk": "CACHE", "sk": f"RECIPE#{recipe_id}"})
    raw = resp.get("Item", {}).get("recipe_json")
    if not raw:
        return None
    return json.loads(raw)


def save_cached_recipe(recipe_id: str, recipe: dict) -> None:
    """Store the slimmed recipe dict in the cache."""
    _table.put_item(
        Item={
            "pk": "CACHE",
            "sk": f"RECIPE#{recipe_id}",
            "recipe_json": json.dumps(recipe),
            "cached_at": _now(),
        }
    )
