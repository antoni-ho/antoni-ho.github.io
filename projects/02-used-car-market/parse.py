"""
Parser for one Mobil123 listing detail page.

The page exposes everything we need in two structured blocks, so no fragile
HTML table scraping is required:

  1. `<meta>` tags named `ga:cad:details:*`
       make, model, variant, year, price, transmission, body_type, fuel_type,
       engine_cc, location (province), profile_type (Dealer/Private),
       seller_id, seller badges, ad_status.

  2. One JSON-LD block, schema.org `Car`
       mileageFromOdometer (exact km), addressLocality (city), enginePower,
       color, seatingCapacity, description (free text written by the seller).

Important: `year` is the **vehicle's model/production year** (JSON-LD
`vehicleModelDate`), not the year the listing was published. It is a property
of the car, never of the dealer or the posting date.

Usage:
    from parse import parse_listing
    row = parse_listing(html_text, url="https://...")

Returns a dict ready to be written as one JSONL line. Missing required fields
do not raise: they are reported in `row["errors"]` so the crawler can keep
going and data quality stays measurable.
"""

from __future__ import annotations

import json
import re

# All listings in this dataset were collected in November 2026. Age is computed
# against this fixed snapshot year so that results stay reproducible.
SNAPSHOT_YEAR = 2026

REQUIRED = ("ad_id", "brand", "model", "year", "price", "transmission")

_RE_META = re.compile(r"<meta\s+([^>]*?)/?>", re.I | re.S)
_RE_ATTR = re.compile(r'(property|name|content)\s*=\s*["\']([^"\']*)["\']', re.I)
_RE_LD = re.compile(
    r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.S | re.I,
)


def _int(value):
    """'230000000' -> 230000000; strips currency separators and symbols."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    digits = re.sub(r"[^\d]", "", str(value))
    return int(digits) if digits else None


def _clean(value):
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def ga_meta(html: str) -> dict:
    """Collect every `ga:cad:details:*` meta tag as a dict without the prefix."""
    out = {}
    for tag in _RE_META.findall(html):
        attrs = {k.lower(): v for k, v in _RE_ATTR.findall(tag)}
        key = attrs.get("property") or attrs.get("name")
        if not key or not key.startswith("ga:cad:details:"):
            continue
        out[key.split(":")[-1]] = attrs.get("content", "")
    return out


def ld_object(html: str) -> dict:
    """Return the first JSON-LD object whose @type is Car/Vehicle/Product."""
    for raw in _RE_LD.findall(html):
        try:
            obj = json.loads(raw.strip())
        except Exception:
            continue
        candidates = obj if isinstance(obj, list) else [obj]
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get("@type") in (
                "Car", "Vehicle", "Product"
            ):
                return candidate
    return {}


def _engine_cc(raw: str | None) -> int | None:
    """
    The `engine_cc` meta actually stores displacement in litres ('1.0', '2.5').
    Values below 20 are treated as litres and converted; values that already
    look like cubic centimetres are passed through.
    """
    if not raw:
        return None
    try:
        value = float(str(raw).strip().replace(",", "."))
    except ValueError:
        return None
    if 0 < value < 20:
        return round(value * 1000)
    if 20 <= value <= 10000:
        return round(value)
    return None


def parse_listing(html: str, url: str | None = None) -> dict:
    meta = ga_meta(html)
    ld = ld_object(html)

    offers = ld.get("offers") or {}
    seller = offers.get("seller") or {}
    address = seller.get("address") or {}
    engine = ld.get("vehicleEngine") or {}
    power = engine.get("enginePower") or {}
    mileage = ld.get("mileageFromOdometer") or {}
    seats = ld.get("seatingCapacity") or {}

    year = _int(meta.get("year"))
    price = _int(meta.get("price")) or _int(offers.get("price"))

    row = {
        # identity
        "ad_id": _int(meta.get("ad_id")),
        "url": url or ld.get("url"),
        # vehicle specification
        "brand": _clean(meta.get("make")),
        "model": _clean(meta.get("model")),
        "variant": _clean(meta.get("variant")),
        "body_type": _clean(meta.get("body_type")),
        "fuel_type": _clean(meta.get("fuel_type")),
        "engine_cc": _engine_cc(meta.get("engine_cc")),
        "engine_hp": _int(power.get("value")),
        "transmission": _clean(meta.get("transmission")),
        "seats": _int(seats.get("value")),
        "color": _clean(ld.get("color")),
        # price and usage
        "year": year,
        "price": price,
        "mileage_km": _int(mileage.get("value")),
        # location
        "province": _clean(meta.get("location")) or _clean(address.get("addressRegion")),
        "city": _clean(address.get("addressLocality")),
        # seller
        "profile_type": _clean(meta.get("profile_type")),
        "seller_id": _clean(meta.get("seller_id")),
        "seller_badges": _clean(meta.get("ad_seller")),
        "listing_flags": _clean(meta.get("ad_products")),
        "ad_status": _clean(meta.get("ad_status")),
        "ad_type": _clean(meta.get("ad_type")),
        # free text
        "description": _clean(ld.get("description")),
    }

    # derived fields
    row["age"] = (SNAPSHOT_YEAR - year) if year else None
    if row["mileage_km"] is not None and row["age"]:
        row["km_per_year"] = round(row["mileage_km"] / max(row["age"], 1))
    else:
        row["km_per_year"] = None

    # light validation (reported, never raised)
    errors = []
    for field in REQUIRED:
        if row.get(field) in (None, "", 0):
            errors.append(f"missing:{field}")
    if year and not (1980 <= year <= SNAPSHOT_YEAR + 1):
        errors.append(f"year_out_of_range:{year}")
    if price is not None and price < 5_000_000:
        errors.append(f"price_too_low:{price}")
    if row["mileage_km"] is not None and row["mileage_km"] < 0:
        errors.append("mileage_negative")
    row["errors"] = errors
    return row


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for path in sorted(Path("raw_samples").glob("*.html")):
        row = parse_listing(path.read_text(encoding="utf-8", errors="replace"))
        print("=" * 60)
        print(path.name, "-> errors:", row["errors"])
        for key, value in row.items():
            if key == "errors":
                continue
            print(f"   {key:16} = {str(value)[:100]}")
