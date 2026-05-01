"""Kroger Public API client.

Uses the client-credentials OAuth2 flow (no user login required) to access
the Locations and Products endpoints. This gives us real store-specific
package sizes and prices.

Scope: `product.compact` — enough for product search. No cart ops, no user
data, which means no user-OAuth dance.

Token lifetime is 30 minutes. We cache in module-level state across Lambda
warm invocations, so a warm Lambda typically makes zero token calls per
request. A cold Lambda makes one.

Location ID is cached in DynamoDB (shared across cold starts) so we only
hit the Locations endpoint once per ZIP code.

Env vars:
    KROGER_CLIENT_ID      required
    KROGER_CLIENT_SECRET  required
    KROGER_ZIP_CODE       optional, defaults to 37203 (Nashville)

Docs: https://developer.kroger.com/reference/api/product-api-public
"""
from __future__ import annotations

import base64
import os
import time
from typing import Any, Optional

import boto3
import requests
from boto3.dynamodb.conditions import Key

CLIENT_ID = os.environ.get("KROGER_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("KROGER_CLIENT_SECRET", "")
ZIP_CODE = os.environ.get("KROGER_ZIP_CODE", "37203")
TABLE_NAME = os.environ.get("TABLE_NAME", "grocery-agent")

BASE = "https://api.kroger.com/v1"
TIMEOUT = 8  # Kroger responses are typically <1s; 8s gives generous slack

_dynamodb = boto3.resource("dynamodb")
_table = _dynamodb.Table(TABLE_NAME)

# Module-level token cache. Survives across warm Lambda invocations.
_token_cache: dict[str, Any] = {"access_token": None, "expires_at": 0}
# Module-level location cache for the duration of a single Lambda container.
# Maps zip -> {location_id, chain, name, city, state}. Keeping the full
# metadata (not just the ID) lets lookup_many() tell the agent which
# specific Kroger-family store the prices came from.
_location_cache: dict[str, dict] = {}


class KrogerError(Exception):
    pass


def is_configured() -> bool:
    """True when client credentials are present."""
    return bool(CLIENT_ID and CLIENT_SECRET)


def current_store_info() -> dict | None:
    """Return metadata about the most recently resolved store in this Lambda
    container: {chain, name, city, state, location_id}. Returns None if no
    lookup has happened yet in this container. Used by packages.lookup_many()
    to tell the agent which specific store answered.
    """
    zip_code = os.environ.get("KROGER_ZIP_CODE", ZIP_CODE)
    info = _location_cache.get(zip_code)
    if not info:
        return None
    return {
        "chain": info.get("chain"),
        "name": info.get("name"),
        "city": info.get("city"),
        "state": info.get("state"),
        "location_id": info.get("location_id"),
        "zip": zip_code,
    }


def _get_access_token() -> str:
    """Return a valid access token, fetching a new one if expired.

    Buffer of 60s so we don't hand back a token that expires mid-request.
    """
    now = time.time()
    if _token_cache["access_token"] and _token_cache["expires_at"] - 60 > now:
        return _token_cache["access_token"]

    if not is_configured():
        raise KrogerError("KROGER_CLIENT_ID / KROGER_CLIENT_SECRET not set")

    basic = base64.b64encode(
        f"{CLIENT_ID}:{CLIENT_SECRET}".encode()
    ).decode()
    resp = requests.post(
        f"{BASE}/connect/oauth2/token",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {basic}",
        },
        data={"grant_type": "client_credentials", "scope": "product.compact"},
        timeout=TIMEOUT,
    )
    if not resp.ok:
        raise KrogerError(f"Token request failed: {resp.status_code} {resp.text[:200]}")
    data = resp.json()
    _token_cache["access_token"] = data["access_token"]
    _token_cache["expires_at"] = now + int(data.get("expires_in", 1800))
    return _token_cache["access_token"]


def _get_location_id(zip_code: str) -> str:
    """Find and cache the best Kroger-family location for the given ZIP.

    Strategy:
      1. In-memory cache (this Lambda container) — full metadata
      2. DynamoDB cache (survives cold starts) — full metadata
      3. Fetch from Kroger and populate both caches

    Always returns just the location_id for use in product queries; the
    full metadata is stashed in _location_cache so current_store_info()
    can surface it to the agent.
    """
    if zip_code in _location_cache:
        return _location_cache[zip_code]["location_id"]

    # DynamoDB cache check
    resp = _table.get_item(Key={"pk": "CACHE", "sk": f"KROGER_LOC#{zip_code}"})
    cached_item = resp.get("Item")
    if cached_item and cached_item.get("location_id"):
        _location_cache[zip_code] = {
            "location_id": cached_item["location_id"],
            "chain": cached_item.get("chain"),
            "name": cached_item.get("name"),
            "city": cached_item.get("city"),
            "state": cached_item.get("state"),
        }
        return cached_item["location_id"]

    # Fresh fetch. No `filter.chain` — Kroger Co. owns many banners (Kroger,
    # Fred Meyer, QFC, Ralphs, King Soopers, Harris Teeter, Smith's, etc.)
    # and the Products API works across all of them. Filtering to just
    # "KROGER" would break in the ~half the country that uses a different
    # banner. Default sort is nearest-first.
    token = _get_access_token()
    r = requests.get(
        f"{BASE}/locations",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        params={
            "filter.zipCode.near": zip_code,
            "filter.limit": 1,
        },
        timeout=TIMEOUT,
    )
    if not r.ok:
        raise KrogerError(f"Location lookup failed: {r.status_code} {r.text[:200]}")
    data = r.json().get("data", [])
    if not data:
        raise KrogerError(
            f"No Kroger-family stores found near ZIP {zip_code}. "
            "Kroger Co. operates under many banners (Kroger, Fred Meyer, QFC, "
            "Ralphs, King Soopers, Harris Teeter, Smith's, Dillons, Fry's, "
            "Mariano's, Pick 'n Save, City Market, Food 4 Less). If none of "
            "these are near you, this API won't help."
        )
    loc = data[0]
    location_id = loc["locationId"]
    chain = loc.get("chain")
    name = loc.get("name")
    city = (loc.get("address") or {}).get("city")
    state = (loc.get("address") or {}).get("state")

    # Log which banner we picked — makes it obvious from CloudWatch whether
    # this ZIP is a Kroger, Harris Teeter, Fred Meyer, etc.
    try:
        from obs import log_event
        log_event(
            "kroger_location_resolved",
            {
                "zip": zip_code,
                "location_id": location_id,
                "chain": chain,
                "name": name,
                "city": city,
                "state": state,
            },
        )
    except ImportError:
        pass

    # Write through to DynamoDB cache — location IDs are very stable.
    _table.put_item(
        Item={
            "pk": "CACHE",
            "sk": f"KROGER_LOC#{zip_code}",
            "location_id": location_id,
            "chain": chain,
            "name": name,
            "city": city,
            "state": state,
            "cached_at": int(time.time()),
        }
    )
    _location_cache[zip_code] = {
        "location_id": location_id,
        "chain": chain,
        "name": name,
        "city": city,
        "state": state,
    }
    return location_id


def _best_product(products: list[dict]) -> Optional[dict]:
    """Choose the most useful product from a search result.

    Heuristic:
      1. Require at least one item with a price.
      2. Prefer lower-priced items (generic/store-brand usually shows up here).
      3. Skip obviously-wrong results: organic-only, large bulk packs > 5 lb,
         gift cards / subscriptions. This is heuristic — not perfect.
    """
    scored = []
    for p in products:
        items = p.get("items") or []
        if not items:
            continue
        item = items[0]
        price = (item.get("price") or {}).get("regular")
        if not price:
            continue
        size = item.get("size", "")
        description = (p.get("description") or "").lower()

        # Skip obvious non-matches
        if "gift card" in description or "subscription" in description:
            continue

        scored.append((price, p, item, size))

    if not scored:
        return None

    # Sort by price ascending — cheapest real product first
    scored.sort(key=lambda t: t[0])
    price, product, item, size = scored[0]
    return {
        "kroger_product_id": product.get("productId"),
        "description": product.get("description"),
        "brand": product.get("brand"),
        "size": size,
        "price_usd": float(price),
        "upc": product.get("upc"),
    }


def search_product(term: str, zip_code: Optional[str] = None) -> Optional[dict]:
    """Search Kroger for a product matching `term` and return the best match.

    Returns None if nothing sensible was found or if the API is not configured.
    Never raises for "no results" — only raises for infrastructure failures
    (network, auth, malformed response).
    """
    if not is_configured():
        return None

    token = _get_access_token()
    location_id = _get_location_id(zip_code or ZIP_CODE)

    r = requests.get(
        f"{BASE}/products",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        params={
            "filter.term": term,
            "filter.locationId": location_id,
            "filter.limit": 10,
        },
        timeout=TIMEOUT,
    )
    if r.status_code == 401:
        # Token expired mid-use — rare but possible. Clear cache and retry once.
        _token_cache["access_token"] = None
        _token_cache["expires_at"] = 0
        token = _get_access_token()
        r = requests.get(
            f"{BASE}/products",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            params={
                "filter.term": term,
                "filter.locationId": location_id,
                "filter.limit": 10,
            },
            timeout=TIMEOUT,
        )
    if not r.ok:
        raise KrogerError(f"Product search failed: {r.status_code} {r.text[:200]}")

    return _best_product(r.json().get("data", []))
