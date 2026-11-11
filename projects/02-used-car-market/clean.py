"""
Stage 2 - clean, deduplicate and enrich the crawl.

Input : data/listings.jsonl  (raw, one JSON object per listing)
Output: data/clean.csv.gz    (analysis dataset, one row per unique car)
        data/clean_report.txt

Steps
  1. drop rows that failed validation during the crawl
  2. normalise brand casing (BMW, MINI, KIA come uppercase from the source)
  3. remove duplicate listings
  4. apply plausibility filters (vehicle year, price, mileage, engine size)
  5. derive analysis features (age, mileage per year, brand origin, segment,
     free-text flags, sampling weight)

On the sampling weight
    The crawl is a *stratified* sample: brands were sampled with deliberately
    unequal quotas, so sample brand shares do not equal market shares.
    ``weight = market_share / sample_share`` re-expresses the sample as the
    marketplace. Use it for market-level descriptive statistics only; the
    hedonic regression conditions on brand and does not need it.

Usage:
    python clean.py
"""

from __future__ import annotations

import csv
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

from scrape import brand_of

RAW = Path("data/listings.jsonl")
URLS = Path("urls.txt")
CLEAN = Path("data/clean.csv.gz")
REPORT = Path("data/clean_report.txt")

SNAPSHOT_YEAR = 2026
MIN_YEAR = 2005
MIN_PRICE = 10_000_000
MAX_PRICE = 50_000_000_000
MAX_MILEAGE = 600_000          # above this the odometer reading is not credible
ENGINE_MIN, ENGINE_MAX = 600, 8000

_lines: list[str] = []


def out(msg: str = "") -> None:
    print(msg)
    _lines.append(str(msg))


# --- brand normalisation ----------------------------------------------------

_BRAND_SPECIAL = {
    "BMW": "BMW", "MINI": "MINI", "MG": "MG", "GWM": "GWM", "BYD": "BYD",
    "DFSK": "DFSK", "KIA": "Kia", "UD TRUCKS": "UD Trucks", "GAC": "GAC",
    "XPENG": "XPeng", "NETA": "Neta", "VINFAST": "VinFast", "AITO": "Aito",
    "DEEPAL": "Deepal", "JAECOO": "Jaecoo", "OMODA": "Omoda", "HAVAL": "Haval",
    "ORA": "Ora", "TANK": "Tank", "TESLA": "Tesla", "ISUZU": "Isuzu",
}

JAPAN = {"Toyota", "Honda", "Suzuki", "Daihatsu", "Mitsubishi", "Nissan",
         "Mazda", "Isuzu", "Hino", "Lexus", "Subaru", "Infiniti", "Acura"}
KOREA = {"Hyundai", "Kia", "Daewoo", "Ssangyong"}
EUROPE = {"Mercedes-Benz", "BMW", "Audi", "Volkswagen", "Porsche", "Volvo",
          "Peugeot", "Renault", "Citroen", "Fiat", "Alfa Romeo", "Abarth",
          "Ferrari", "Lamborghini", "Maserati", "Bentley", "Rolls-Royce",
          "Aston Martin", "McLaren", "Jaguar", "Land Rover", "MINI", "Smart",
          "Skoda", "Opel", "Seat"}
USA = {"Ford", "Chevrolet", "Jeep", "Dodge", "Chrysler", "Cadillac",
       "Lincoln", "Tesla", "Hummer", "Scion"}
CHINA = {"Wuling", "BYD", "Chery", "Geely", "DFSK", "Maxus", "GWM", "Haval",
         "Ora", "Tank", "Jaecoo", "Omoda", "MG", "Denza", "XPeng", "Neta",
         "Aito", "Deepal", "GAC", "Aion"}
VIETNAM = {"VinFast"}

LUXURY = {"Mercedes-Benz", "BMW", "Audi", "Lexus", "Porsche", "MINI",
          "Land Rover", "Jaguar", "Volvo", "Infiniti", "Ferrari",
          "Lamborghini", "Maserati", "Bentley", "Rolls-Royce",
          "Aston Martin", "McLaren"}


def norm_brand(raw: str | None) -> str:
    brand = (raw or "").strip()
    key = brand.upper()
    if key in _BRAND_SPECIAL:
        return _BRAND_SPECIAL[key]
    return brand.title()


def origin_of(brand: str) -> str:
    if brand in JAPAN:
        return "Japan"
    if brand in KOREA:
        return "Korea"
    if brand in EUROPE:
        return "Europe"
    if brand in USA:
        return "USA"
    if brand in CHINA:
        return "China"
    if brand in VIETNAM:
        return "Vietnam"
    return "Other"


# --- free-text flags --------------------------------------------------------

_FLAGS = {
    "flag_pajak_hidup": re.compile(r"pajak\s*(hidup|panjang|baru|on\b)", re.I),
    "flag_pajak_mati": re.compile(r"pajak\s*mati", re.I),
    "flag_tangan_pertama": re.compile(r"tangan\s*(pertama|1\b)", re.I),
    "flag_service_record": re.compile(r"servis|service|riwayat", re.I),
    "flag_km_asli": re.compile(r"km\s*(asli|original)", re.I),
    "flag_nego": re.compile(r"\bnego\b|\bnett?\b", re.I),
}


def main() -> None:
    raw = []
    with RAW.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    out("CLEANING REPORT")
    out("=" * 64)
    out(f"rows read                : {len(raw):,}")
    out()

    # 1. validation failures
    invalid = [r for r in raw if r.get("errors")]
    rows = [r for r in raw if not r.get("errors")]
    out(f"1. drop invalid rows     : -{len(invalid):,}  -> {len(rows):,}")

    # 2. brand normalisation
    for r in rows:
        r["brand"] = norm_brand(r.get("brand"))
        r["model"] = re.sub(r"\s+", " ", (r.get("model") or "").strip())
        r["city"] = re.sub(r"\s+", " ", (r.get("city") or "").strip()) or None
    out(f"2. brand normalised      : {len({r['brand'] for r in rows}):,} distinct brands")

    # 3. duplicates
    seen = set()
    deduped = []
    for r in rows:
        key = (r["brand"], r["model"], r.get("year"), r.get("price"),
               r.get("mileage_km"), r.get("city"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    removed_dup = len(rows) - len(deduped)
    rows = deduped
    out(f"3. drop duplicates       : -{removed_dup:,}  -> {len(rows):,} "
        f"({100*removed_dup/max(len(seen)+removed_dup,1):.1f}%)")

    # 4. plausibility filters
    steps = [
        ("year outside 2005-2026",
         lambda r: isinstance(r.get("year"), int) and MIN_YEAR <= r["year"] <= SNAPSHOT_YEAR),
        ("price outside 10M-50B",
         lambda r: r.get("price") and MIN_PRICE <= r["price"] <= MAX_PRICE),
        ("transmission unknown",
         lambda r: r.get("transmission") in ("Automatic", "Manual")),
    ]
    for label, predicate in steps:
        before = len(rows)
        rows = [r for r in rows if predicate(r)]
        out(f"4. drop {label:<24}: -{before-len(rows):,}  -> {len(rows):,}")

    # 5. derive features
    for r in rows:
        r["age"] = SNAPSHOT_YEAR - r["year"] if r.get("year") else None

        mileage = r.get("mileage_km")
        if mileage is None or mileage < 0 or mileage > MAX_MILEAGE:
            r["mileage_km"] = None
        engine = r.get("engine_cc")
        if engine is not None and not (ENGINE_MIN <= engine <= ENGINE_MAX):
            r["engine_cc"] = None

        if r.get("mileage_km") is not None and r.get("age"):
            r["km_per_year"] = round(r["mileage_km"] / max(r["age"], 1))
        else:
            r["km_per_year"] = None

        r["origin"] = origin_of(r["brand"])
        r["is_luxury"] = r["brand"] in LUXURY
        text = r.get("description") or ""
        for name, pattern in _FLAGS.items():
            r[name] = int(bool(pattern.search(text)))

    # 6. sampling weights (market share / sample share, by brand)
    market = Counter()
    if URLS.exists():
        for url in URLS.read_text(encoding="utf-8").splitlines():
            if url.strip():
                market[brand_of(url)] += 1
    market_total = sum(market.values())
    sample = Counter(r["brand"].lower() for r in rows)
    sample_total = sum(sample.values())
    for r in rows:
        key = r["brand"].lower()
        pop = market.get(key, market.get("other", 0))
        if pop and sample.get(key):
            r["weight"] = round((pop / market_total) / (sample[key] / sample_total), 4)
        else:
            r["weight"] = None

    # 7. write
    columns = [
        "ad_id", "brand", "model", "variant", "year", "age", "price",
        "mileage_km", "km_per_year", "transmission", "engine_cc", "engine_hp",
        "fuel_type", "body_type", "seats", "color", "origin", "is_luxury",
        "province", "city", "profile_type", "seller_badges", "listing_flags",
        "flag_pajak_hidup", "flag_pajak_mati", "flag_tangan_pertama",
        "flag_service_record", "flag_km_asli", "flag_nego", "weight",
        "description", "url",
    ]
    CLEAN.parent.mkdir(exist_ok=True)
    with gzip.open(CLEAN, "wt", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    out()
    out(f"final rows               : {len(rows):,}")
    out(f"unique sellers           : {len({r.get('seller_id') for r in rows}):,}")
    out(f"output                   : {CLEAN} "
        f"({CLEAN.stat().st_size/1024/1024:.2f} MB gzipped)")
    out()

    out("-- flag prevalence (share of listings whose description mentions it) --")
    for name in _FLAGS:
        n = sum(1 for r in rows if r.get(name))
        out(f"   {name:24} {n:>6,}  {100*n/len(rows):5.1f}%")
    out()

    out("-- composition (unweighted sample vs weighted market estimate) --")
    for label, key in (("transmission", "transmission"), ("origin", "origin")):
        counts = Counter(r[key] for r in rows)
        weighted = Counter()
        for r in rows:
            weighted[r[key]] += r["weight"] or 0
        wtotal = sum(weighted.values())
        out(f"   {label}:")
        for value, n in counts.most_common(6):
            out(f"      {str(value):14} sample {100*n/len(rows):5.1f}%   "
                f"market {100*weighted[value]/wtotal:5.1f}%")
    out()

    out("-- year coverage of the analysis set --")
    years = Counter(r["year"] for r in rows)
    for year in sorted(years, reverse=True):
        if years[year] >= 50:
            out(f"   {year}  {years[year]:>5,}")

    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    out()
    out(f"report written to {REPORT}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
