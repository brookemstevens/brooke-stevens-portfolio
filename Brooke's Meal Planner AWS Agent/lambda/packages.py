"""Realistic grocery package sizes and prices.

Most recipe APIs (including Spoonacular) return ingredient quantities in
cooking units — "6 oz chicken breast", "1/2 cup diced onion". These don't
match how things are actually sold at the grocery store. This module closes
that gap.

Data source is Kroger Products API (live prices at user's nearest store)
with a curated fallback catalog when Kroger returns nothing useful.
The interface is uniform so the agent doesn't care which source answered.
"""

# Import the structured-logging helper if available; otherwise noop.
# The try/except lets this module be imported in isolation for testing.
try:
    from obs import log_event as _log
except ImportError:
    def _log(event_type, payload):  # noqa: ARG001
        pass


# Ingredient name (normalized lowercase) -> package info.
# - smallest_size / smallest_unit: the smallest typical retail unit
# - price_usd: approximate per-package price
# - notes: anything the agent should know about buying this
# - bulk_alternative: mention when bulk is dramatically cheaper per-unit
PACKAGE_DATA: dict[str, dict] = {
    # --- Proteins (often bought in larger packages than single recipes need) ---
    "chicken breast": {
        "smallest_size": 1.0, "smallest_unit": "lb", "price_usd": 6.99,
        "notes": "Usually sold in 1-2 lb family packs. A single 6-8 oz breast is half a package; plan multiple recipes or freeze the rest.",
    },
    "chicken thighs": {
        "smallest_size": 1.5, "smallest_unit": "lb", "price_usd": 5.99,
        "notes": "Bone-in or boneless; both sold in 1.5-2 lb packages.",
    },
    "ground beef": {
        "smallest_size": 1.0, "smallest_unit": "lb", "price_usd": 6.49,
        "notes": "1 lb is the standard package. 85/15 is the common lean ratio.",
    },
    "ground turkey": {
        "smallest_size": 1.0, "smallest_unit": "lb", "price_usd": 5.99,
    },
    "salmon fillet": {
        "smallest_size": 0.5, "smallest_unit": "lb", "price_usd": 10.99,
        "notes": "Sold per filet (~6-8 oz) or by the pound at the counter. Frozen individually-wrapped fillets are cheaper and store well.",
    },
    "shrimp": {
        "smallest_size": 1.0, "smallest_unit": "lb", "price_usd": 9.99,
        "notes": "Frozen bagged shrimp is much cheaper than fresh. 1 lb bag is standard.",
    },
    "eggs": {
        "smallest_size": 12, "smallest_unit": "count", "price_usd": 3.99,
        "notes": "Sold by the dozen. 18-count and 24-count available but usually not cheaper per egg.",
    },
    "bacon": {
        "smallest_size": 12, "smallest_unit": "oz", "price_usd": 6.99,
    },

    # --- Dairy ---
    "milk": {
        "smallest_size": 0.5, "smallest_unit": "gallon", "price_usd": 2.99,
        "notes": "Half-gallon is smallest; full gallon often same price or marginally more.",
    },
    "butter": {
        "smallest_size": 1.0, "smallest_unit": "lb", "price_usd": 5.49,
        "notes": "Sold in 1-lb packages of 4 sticks. Buying one stick isn't possible.",
    },
    "greek yogurt": {
        "smallest_size": 32, "smallest_unit": "oz", "price_usd": 5.99,
        "notes": "Individual cups (5-6 oz) are available but much more expensive per oz than the tub.",
    },
    "feta": {
        "smallest_size": 6, "smallest_unit": "oz", "price_usd": 4.49,
    },
    "parmesan": {
        "smallest_size": 5, "smallest_unit": "oz", "price_usd": 5.99,
        "notes": "Grated wedge vs. tub of pre-grated: wedge keeps longer.",
    },
    "mozzarella": {
        "smallest_size": 8, "smallest_unit": "oz", "price_usd": 3.99,
    },
    "heavy cream": {
        "smallest_size": 1, "smallest_unit": "pint", "price_usd": 3.99,
    },

    # --- Produce (often needs realistic "whole unit" thinking) ---
    "lemon": {
        "smallest_size": 1, "smallest_unit": "count", "price_usd": 0.75,
    },
    "lime": {
        "smallest_size": 1, "smallest_unit": "count", "price_usd": 0.50,
    },
    "avocado": {
        "smallest_size": 1, "smallest_unit": "count", "price_usd": 1.50,
    },
    "tomato": {
        "smallest_size": 1, "smallest_unit": "count", "price_usd": 1.00,
        "notes": "Roma/plum tomatoes smaller and cheaper than slicing tomatoes.",
    },
    "cherry tomatoes": {
        "smallest_size": 10, "smallest_unit": "oz", "price_usd": 3.49,
        "notes": "Sold in pint or 10-oz plastic containers.",
    },
    "onion": {
        "smallest_size": 1, "smallest_unit": "count", "price_usd": 0.75,
        "notes": "Can buy 3 lb bag for ~$3.99 (~5-6 onions) if using a lot.",
    },
    "garlic": {
        "smallest_size": 1, "smallest_unit": "head", "price_usd": 0.75,
        "notes": "One head = ~10-12 cloves. Pre-peeled in a jar is more expensive per clove but lasts longer.",
    },
    "broccoli": {
        "smallest_size": 1, "smallest_unit": "head", "price_usd": 2.99,
        "notes": "One head ≈ 1 lb. Pre-cut florets in bags are 12 oz and cost ~$3.99 — convenience premium.",
    },
    "spinach": {
        "smallest_size": 5, "smallest_unit": "oz", "price_usd": 3.49,
        "notes": "Pre-washed bag. Wilts fast — buy frozen if unsure you'll use it in time.",
    },
    "carrots": {
        "smallest_size": 1, "smallest_unit": "lb", "price_usd": 1.49,
    },
    "zucchini": {
        "smallest_size": 1, "smallest_unit": "count", "price_usd": 1.00,
    },
    "cucumber": {
        "smallest_size": 1, "smallest_unit": "count", "price_usd": 1.00,
    },
    "potato": {
        "smallest_size": 5, "smallest_unit": "lb", "price_usd": 4.99,
        "notes": "Sold in bags (3-5 lb common). Loose by the pound at some stores.",
    },
    "baby potatoes": {
        "smallest_size": 1.5, "smallest_unit": "lb", "price_usd": 4.49,
    },
    "parsley": {
        "smallest_size": 1, "smallest_unit": "bunch", "price_usd": 1.49,
        "notes": "A bunch is a LOT for one person; freezes well chopped.",
    },
    "cilantro": {
        "smallest_size": 1, "smallest_unit": "bunch", "price_usd": 1.49,
        "notes": "Wilts fast. Single person will waste most of a bunch unless using repeatedly.",
    },
    "dill": {
        "smallest_size": 1, "smallest_unit": "bunch", "price_usd": 1.99,
    },

    # --- Pantry staples ---
    "rice": {
        "smallest_size": 2, "smallest_unit": "lb", "price_usd": 3.49,
        "notes": "Smallest useful size. 5-lb or 10-lb bags much cheaper per-lb. Jasmine, basmati, long-grain all similar.",
    },
    "brown rice": {
        "smallest_size": 2, "smallest_unit": "lb", "price_usd": 3.99,
    },
    "quinoa": {
        "smallest_size": 12, "smallest_unit": "oz", "price_usd": 4.49,
    },
    "pasta": {
        "smallest_size": 1, "smallest_unit": "lb", "price_usd": 1.79,
        "notes": "Standard box is 1 lb / 16 oz.",
    },
    "olive oil": {
        "smallest_size": 17, "smallest_unit": "fl oz", "price_usd": 8.99,
        "notes": "500ml bottle is smallest decent size. Lasts many months.",
    },
    "canned tomatoes": {
        "smallest_size": 14.5, "smallest_unit": "oz", "price_usd": 1.49,
        "notes": "Standard 14.5 oz can. 28 oz cans for bigger recipes.",
    },
    "tomato paste": {
        "smallest_size": 6, "smallest_unit": "oz", "price_usd": 1.29,
        "notes": "Tube is better than can for small amounts — stores in fridge after opening.",
    },
    "chicken broth": {
        "smallest_size": 32, "smallest_unit": "fl oz", "price_usd": 3.49,
    },
    "black beans": {
        "smallest_size": 15, "smallest_unit": "oz", "price_usd": 1.29,
    },
    "chickpeas": {
        "smallest_size": 15, "smallest_unit": "oz", "price_usd": 1.29,
    },
    "flour": {
        "smallest_size": 5, "smallest_unit": "lb", "price_usd": 4.49,
    },
    "sugar": {
        "smallest_size": 4, "smallest_unit": "lb", "price_usd": 3.99,
    },

    # --- Bread / grains ---
    "bread": {
        "smallest_size": 1, "smallest_unit": "loaf", "price_usd": 3.99,
    },
    "tortillas": {
        "smallest_size": 8, "smallest_unit": "count", "price_usd": 2.99,
    },
    "pita bread": {
        "smallest_size": 6, "smallest_unit": "count", "price_usd": 3.49,
    },
}


def _normalize_name(name: str) -> str:
    """Lowercase + trim, plus a few common-synonym folds."""
    n = name.strip().lower()
    # Simple singular/plural + common variant folding
    synonyms = {
        "chicken breasts": "chicken breast",
        "chicken thigh": "chicken thighs",
        "tomatoes": "tomato",
        "onions": "onion",
        "lemons": "lemon",
        "limes": "lime",
        "avocados": "avocado",
        "carrot": "carrots",
        "potatoes": "potato",
        "egg": "eggs",
        "lb ground beef": "ground beef",
        "salmon": "salmon fillet",
        "salmon fillets": "salmon fillet",
    }
    return synonyms.get(n, n)


def _curated_lookup(ingredient: str) -> dict | None:
    """Look up ingredient in the hardcoded PACKAGE_DATA dict."""
    key = _normalize_name(ingredient)
    if key in PACKAGE_DATA:
        return {"ingredient": key, "source": "curated", **PACKAGE_DATA[key]}
    return None


def _kroger_lookup(ingredient: str) -> dict | None:
    """Look up ingredient via Kroger API. Returns None if not configured,
    no results found, or on any error (never raises — caller should fall
    back to curated data).

    All failures are logged so you can diagnose why Kroger didn't answer.
    """
    try:
        import kroger  # lazy import so unit tests/curated-only mode still work
    except ImportError:
        _log("kroger_skip", {"ingredient": ingredient, "reason": "module_not_importable"})
        return None
    if not kroger.is_configured():
        _log("kroger_skip", {"ingredient": ingredient, "reason": "not_configured"})
        return None
    try:
        result = kroger.search_product(_normalize_name(ingredient))
    except kroger.KrogerError as e:
        _log("kroger_error", {"ingredient": ingredient, "error": str(e)[:300]})
        return None
    except Exception as e:
        _log("kroger_error", {
            "ingredient": ingredient,
            "error": str(e)[:300],
            "type": type(e).__name__,
        })
        return None
    if not result:
        _log("kroger_no_match", {"ingredient": ingredient})
        return None

    # Normalize Kroger's size string into smallest_size + smallest_unit where
    # possible. Kroger's size is free-text (e.g. "1 lb", "12 oz", "32 fl oz").
    size_str = (result.get("size") or "").strip()
    smallest_size, smallest_unit = _parse_size(size_str)

    return {
        "ingredient": _normalize_name(ingredient),
        "source": "kroger",
        "smallest_size": smallest_size,
        "smallest_unit": smallest_unit or size_str or "package",
        "price_usd": result["price_usd"],
        "product_description": result.get("description"),
        "brand": result.get("brand"),
        "raw_size": size_str,
    }


def _parse_size(size: str) -> tuple[float | None, str | None]:
    """Parse a size string like '1 lb' or '12 oz' into (value, unit).

    Kroger sizes are free-text and varied. This handles the common cases;
    anything exotic returns (None, None) and the caller uses raw_size."""
    import re
    if not size:
        return None, None
    m = re.match(
        r"^\s*([\d.]+)\s*(lb|lbs|pound|pounds|oz|ounce|ounces|fl\s*oz|g|gram|grams|kg|ml|l|ct|count|pack|each)\b",
        size.lower(),
    )
    if not m:
        return None, None
    try:
        val = float(m.group(1))
    except ValueError:
        return None, None
    unit = m.group(2).replace(" ", "")
    # Normalize variants
    unit_map = {
        "lbs": "lb", "pound": "lb", "pounds": "lb",
        "ounce": "oz", "ounces": "oz",
        "floz": "fl oz",
        "gram": "g", "grams": "g",
        "ct": "count", "each": "count",
    }
    return val, unit_map.get(unit, unit)


def lookup(ingredient: str) -> dict | None:
    """Return package info for an ingredient.

    Tries Kroger API first (real store prices) and falls back to the curated
    dict. Returns None only if neither source has data for this ingredient.
    """
    result = _kroger_lookup(ingredient)
    if result:
        return result
    return _curated_lookup(ingredient)


def lookup_many(ingredients: list[str]) -> dict:
    """Batch lookup. Returns {found: [...], unknown: [...], sources: {...},
    store: {...} | None}.

    Each found item includes a 'source' field ("kroger" or "curated") so
    the agent can weight confidence appropriately. When at least one Kroger
    result was returned, the response also includes a `store` dict naming
    the specific Kroger-family location so the agent can tell the user
    exactly which store these prices are from."""
    found = []
    unknown = []
    source_counts = {"kroger": 0, "curated": 0}
    for ing in ingredients:
        info = lookup(ing)
        if info:
            found.append(info)
            source_counts[info.get("source", "curated")] += 1
        else:
            unknown.append(ing)

    # Attach the resolved store when any Kroger result was included. This
    # gives the agent an unambiguous "these prices are from <X>" answer
    # instead of hedging in the natural-language reply.
    store = None
    if source_counts["kroger"] > 0:
        try:
            import kroger
            store = kroger.current_store_info()
        except Exception:
            store = None

    return {
        "found": found,
        "unknown": unknown,
        "sources": source_counts,
        "store": store,
    }
