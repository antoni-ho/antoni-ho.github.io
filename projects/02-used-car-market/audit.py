"""
Stage 2a - data quality audit.

Reads data/listings.jsonl and reports:

  - field completeness
  - brand, model, year and (brand, model, year) cell distribution
  - duplicate listings
  - price and mileage distributions
  - statistical precision: how wide the confidence interval around a
    per-cell median actually is, and how much a larger sample would help

The precision section is the quantitative justification for collecting
12,000 listings rather than all 59,037 - see METHODOLOGY.md.

Usage:
    python audit.py

Output:
    prints to stdout and writes data/audit_report.txt
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

LISTINGS = Path("data/listings.jsonl")
REPORT = Path("data/audit_report.txt")

_lines: list[str] = []


def out(msg: str = "") -> None:
    print(msg)
    _lines.append(str(msg))


def quantile(values: list[float], p: float) -> float:
    """Linear-interpolated quantile."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * p
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def main() -> None:
    rows = []
    with LISTINGS.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    n = len(rows)
    out("DATA QUALITY AUDIT")
    out("=" * 62)
    out(f"rows read: {n:,}")
    out()

    # ---------- completeness ----------
    fields = [
        "brand", "model", "variant", "year", "price", "mileage_km",
        "transmission", "body_type", "fuel_type", "engine_cc", "province",
        "city", "profile_type", "description", "seller_badges",
    ]
    out("-- field completeness --")
    for field in fields:
        filled = sum(1 for r in rows if r.get(field) not in (None, "", 0))
        out(f"   {field:15} {filled:>7,}  {100*filled/n:5.1f}%")
    out()

    # ---------- brands ----------
    brands = Counter(r.get("brand") for r in rows)
    out(f"-- brands ({len(brands)}) --")
    for brand, count in brands.most_common(18):
        out(f"   {str(brand):18} {count:>6,}  {100*count/n:5.1f}%")
    if len(brands) > 18:
        out(f"   (+{len(brands)-18} more brands)")
    out()

    # ---------- year = vehicle model year ----------
    years = Counter(r.get("year") for r in rows if r.get("year"))
    out("-- vehicle model year --")
    for year in sorted(years, key=lambda v: -v):
        bar = "#" * max(1, round(60 * years[year] / max(years.values())))
        out(f"   {year}  {years[year]:>5,}  {bar}")
    out()

    # ---------- model x year cells ----------
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("model") and r.get("year"):
            cells[(r["brand"], r["model"], r["year"])].append(r)
    sizes = [len(v) for v in cells.values()]
    out("-- cells (brand, model, year) --")
    out(f"   cells                : {len(cells):,}")
    out(f"   median cell size     : {quantile(sizes, 0.50):.0f}")
    out(f"   quartiles 1 / 3      : {quantile(sizes, 0.25):.0f} / {quantile(sizes, 0.75):.0f}")
    for threshold in (5, 10, 20, 30, 50):
        ok = sum(1 for s in sizes if s >= threshold)
        obs = sum(s for s in sizes if s >= threshold)
        out(f"   cells >= {threshold:>2} obs     : {ok:>5,} cells "
            f"({100*ok/len(cells):4.1f}%)  cover {100*obs/n:4.1f}% of rows")
    out()

    # ---------- precision of a per-cell median ----------
    within = []
    for values in cells.values():
        if len(values) < 2:
            continue
        logs = [math.log(r["price"]) for r in values if r.get("price")]
        if len(logs) < 2:
            continue
        mean = sum(logs) / len(logs)
        var = sum((x - mean) ** 2 for x in logs) / (len(logs) - 1)
        within.append((len(logs), var))

    if within:
        sigma = math.sqrt(
            sum(w * var for w, var in within) / sum(w for w, _ in within)
        )
        n_median = quantile(sizes, 0.50)

        def ci_half(sample: float) -> float:
            # standard error of a median ~= 1.253 * sigma / sqrt(n)
            return 1.96 * 1.253 * sigma / math.sqrt(max(sample, 1))

        out("-- statistical precision (log price) --")
        out(f"   within-cell sigma    : {sigma:.3f} "
            f"(about +/-{100*(math.exp(sigma)-1):.0f}% between comparable cars)")
        out(f"   median cell (n={n_median:.0f})   : 95% CI of median = "
            f"+/-{100*(math.exp(ci_half(n_median))-1):.1f}%")
        out(f"   with n x 5 (n={n_median*5:.0f})   : 95% CI of median = "
            f"+/-{100*(math.exp(ci_half(n_median*5))-1):.1f}%")
        out("   -> a 5x larger sample narrows the interval by roughly 55%,")
        out("      at 5x the crawl time and bandwidth.")
    out()

    # ---------- duplicates ----------
    fingerprints = Counter(
        (r.get("brand"), r.get("model"), r.get("year"), r.get("price"),
         r.get("mileage_km"), r.get("city"))
        for r in rows
    )
    dup_groups = {k: c for k, c in fingerprints.items() if c > 1}
    dup_rows = sum(c - 1 for c in dup_groups.values())
    out("-- duplicates (brand, model, year, price, km, city identical) --")
    out(f"   duplicate groups     : {len(dup_groups):,}")
    out(f"   excess rows          : {dup_rows:,}  ({100*dup_rows/n:.1f}%)")
    sellers = Counter(r.get("seller_id") for r in rows)
    out(f"   unique sellers       : {len(sellers):,}")
    out(f"   listings per seller  : mean {n/len(sellers):.1f}, "
        f"max {max(sellers.values())}")
    out()

    # ---------- price and mileage ----------
    prices = [r["price"] for r in rows if r.get("price")]
    out("-- price (IDR) --")
    for p in (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999):
        out(f"   p{p*100:>5.1f}  {quantile(prices, p):>18,.0f}")
    kms = [r["mileage_km"] for r in rows if r.get("mileage_km") is not None]
    out(f"-- mileage: p5 {quantile(kms,0.05):,.0f} | median {quantile(kms,0.5):,.0f} "
        f"| p95 {quantile(kms,0.95):,.0f} | max {max(kms):,}")
    out()

    # ---------- transmission and seller type ----------
    transmissions = Counter(r.get("transmission") for r in rows)
    profiles = Counter(r.get("profile_type") for r in rows)
    out("-- transmission --")
    for key, count in transmissions.most_common():
        out(f"   {str(key):18} {count:>6,}  {100*count/n:5.1f}%")
    out("-- seller type --")
    for key, count in profiles.most_common():
        out(f"   {str(key):18} {count:>6,}  {100*count/n:5.1f}%")

    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    out()
    out(f"report written to {REPORT}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
