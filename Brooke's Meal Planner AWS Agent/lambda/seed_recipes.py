"""Seed the DynamoDB recipe cache with a curated set of recipes.

Run this ONCE from your local machine, ideally first thing after the
Spoonacular quota resets. It'll burn ~1 point per recipe seeded — for the
default list of 20 recipes, that's ~20 points out of your daily 50.

After seeding, these recipes will be served from DynamoDB cache on
subsequent requests, so evaluation and demo runs won't dip into quota.

Requirements:
    - AWS credentials configured locally (aws configure, or env vars)
    - SPOONACULAR_API_KEY env var set
    - Same boto3 / requests installed: pip install boto3 requests

Usage:
    export SPOONACULAR_API_KEY=...
    export AWS_REGION=us-east-2
    python seed_recipes.py
"""
import json
import os
import sys
import time

import boto3
import requests

API_KEY = os.environ.get("SPOONACULAR_API_KEY")
REGION = os.environ.get("AWS_REGION", "us-east-2")
TABLE_NAME = os.environ.get("TABLE_NAME", "grocery-agent")

# Queries tuned to the preferences stored earlier — Mediterranean, Italian,
# salmon, chicken, broccoli, rice. Each query returns a handful of recipes
# to populate a diverse cache. Adjust to taste.
SEARCH_QUERIES = [
    {"query": "salmon rice bowl", "number": 3},
    {"query": "greek chicken", "number": 3},
    {"query": "mediterranean chicken", "number": 3},
    {"query": "italian baked salmon", "number": 2},
    {"query": "chicken broccoli rice", "number": 3},
    {"query": "sheet pan chicken", "number": 2},
    {"query": "lemon garlic salmon", "number": 2},
    {"query": "chicken pasta", "number": 2},
]

# Hard dislikes to exclude. Keeping this in sync with your real dislikes
# avoids wasting API points on recipes the agent would reject anyway.
EXCLUDE = "bell pepper,mushroom,eggplant,olive,chili pepper,jalapeno,peanut,blue cheese"


def search_for_ids() -> list[int]:
    """Run search queries and collect unique recipe IDs."""
    seen: set[int] = set()
    ids: list[int] = []
    for q in SEARCH_QUERIES:
        params = {
            "apiKey": API_KEY,
            "query": q["query"],
            "number": q["number"],
            "excludeIngredients": EXCLUDE,
        }
        print(f"Searching: {q['query']!r}")
        r = requests.get(
            "https://api.spoonacular.com/recipes/complexSearch",
            params=params,
            timeout=15,
        )
        if not r.ok:
            print(f"  ! search failed: {r.status_code} {r.text[:200]}")
            continue
        for hit in r.json().get("results", []):
            rid = hit.get("id")
            if rid and rid not in seen:
                seen.add(rid)
                ids.append(rid)
        time.sleep(0.5)  # be polite to the API
    return ids


def fetch_and_slim(recipe_id: int) -> dict | None:
    r = requests.get(
        f"https://api.spoonacular.com/recipes/{recipe_id}/information",
        params={"apiKey": API_KEY, "includeNutrition": "true"},
        timeout=15,
    )
    if not r.ok:
        print(f"  ! fetch {recipe_id} failed: {r.status_code}")
        return None
    data = r.json()
    return {
        "id": data.get("id"),
        "title": data.get("title"),
        "ready_in_minutes": data.get("readyInMinutes"),
        "servings": data.get("servings"),
        "ingredients": [
            {
                "name": ing.get("name"),
                "amount": ing.get("amount"),
                "unit": ing.get("unit"),
                "original": ing.get("original"),
            }
            for ing in data.get("extendedIngredients", [])
        ],
        "instructions": data.get("instructions"),
        "source_url": data.get("sourceUrl"),
    }


def main() -> int:
    if not API_KEY:
        print("ERROR: SPOONACULAR_API_KEY not set.")
        return 1

    table = boto3.resource("dynamodb", region_name=REGION).Table(TABLE_NAME)

    print("Collecting recipe IDs...")
    ids = search_for_ids()
    print(f"Found {len(ids)} unique recipe IDs.")

    saved = 0
    for rid in ids:
        # Skip if already cached — idempotent, safe to rerun.
        existing = table.get_item(Key={"pk": "CACHE", "sk": f"RECIPE#{rid}"}).get("Item")
        if existing:
            print(f"  already cached: {rid}")
            continue

        print(f"  fetching {rid}...")
        recipe = fetch_and_slim(rid)
        if recipe is None:
            continue

        table.put_item(
            Item={
                "pk": "CACHE",
                "sk": f"RECIPE#{rid}",
                "recipe_json": json.dumps(recipe),
                "cached_at": int(time.time()),
            }
        )
        saved += 1
        time.sleep(0.3)

    print(f"\nDone. Seeded {saved} new recipes. Quota used: ~{len(ids) + len(SEARCH_QUERIES)} points.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
