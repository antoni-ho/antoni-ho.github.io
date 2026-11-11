"""
Stage 1b - crawl Mobil123 used-car listings.

Flow:
  1. read urls.txt (produced by enumerate_urls.py)
  2. group URLs by brand using the slug, then draw a stratified sample
     (large brands get a larger quota, small brands are protected by a floor)
  3. fetch each detail page, parse it with parse.py, append one JSON line
  4. discard the HTML right after parsing - nothing large is ever stored
  5. stop at 12,000 valid rows; unused URLs stay queued for a later top-up

The job is resumable: Ctrl+C is safe, and re-running continues where it left
off instead of starting over.

Usage:
    python scrape.py --plan     # print the sampling plan, download nothing
    python scrape.py            # run the crawl
    python scrape.py --target 6000

Output:
    data/listings.jsonl    one JSON object per line, one line per car
    data/failures.jsonl    URLs that returned 404 or failed to parse
    data/attempted.txt     every URL already tried (resume bookkeeping)
    data/audit/*.html.gz   raw HTML of the first 10 listings (audit trail)
    data/run_report.txt    crawl summary
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

from parse import parse_listing

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
AUDIT = DATA / "audit"
LISTINGS = DATA / "listings.jsonl"
FAILURES = DATA / "failures.jsonl"
ATTEMPTED = DATA / "attempted.txt"
REPORT = DATA / "run_report.txt"
URLS = HERE / "urls.txt"

CURL = shutil.which("curl") or "curl.exe"
WORKERS = 5              # parallel connections
PLAN_TARGET = 15_000     # URLs prepared; deliberately above the row target so
                         # expired 404 URLs do not leave us short
STOP_ON_ROWS = 12_000    # stop once this many valid rows are collected
SEED = 42

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
HEADERS = [
    "-H", "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "-H", "Accept-Language: id-ID,id;q=0.9,en;q=0.8",
    "-H", 'sec-ch-ua: "Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "-H", "sec-ch-ua-mobile: ?0",
    "-H", 'sec-ch-ua-platform: "Windows"',
]

# Longest names first so "mercedes-benz" is matched before "mercedes".
BRANDS = [
    "mercedes-benz", "land-rover", "alfa-romeo", "aston-martin", "rolls-royce",
    "great-wall", "range-rover", "mitsubishi-fuso", "morris-garages", "ssangyong",
    "chevrolet", "volkswagen", "toyota", "honda", "suzuki", "daihatsu", "mitsubishi",
    "nissan", "mazda", "isuzu", "hino", "wuling", "hyundai", "kia", "ford", "bmw",
    "audi", "porsche", "volvo", "peugeot", "renault", "jeep", "jaguar", "lexus",
    "infiniti", "mini", "subaru", "chery", "mg", "byd", "geely", "dfsk", "maxus",
    "tesla", "neta", "vinfast", "gwm", "haval", "ora", "tank", "jaecoo", "omoda",
    "citroen", "fiat", "abarth", "dodge", "chrysler", "cadillac", "lincoln",
    "daewoo", "proton", "perodua", "smart", "ferrari", "lamborghini", "maserati",
    "bentley", "hummer", "scion", "tata", "fuso", "ud-trucks", "baic", "jac",
    "foton", "soueast", "denza", "xpeng", "gac-aion", "gac", "aito", "deepal",
]
BRANDS.sort(key=len, reverse=True)

# Quotas, in prepared URLs per brand.
CAPS = {
    "toyota": 2500, "honda": 1500,
    "mercedes-benz": 1000, "mitsubishi": 1000,
    "hyundai": 880, "bmw": 880, "daihatsu": 880,
    "suzuki": 820, "mazda": 690,
    "lexus": 500, "nissan": 500,
    "byd": 440, "chery": 380, "wuling": 320,
    "mini": 220, "land-rover": 220, "kia": 220, "porsche": 220,
    "gwm": 150, "mg": 150, "volkswagen": 150,
    "ford": 130, "jeep": 130, "geely": 130, "chevrolet": 130,
}
REST_CAP = 50        # any other brand with at least MIN_COUNT listings
MIN_COUNT = 30
UNKNOWN_BUCKET = "other"
UNKNOWN_CAP = 400


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def brand_of(url: str) -> str:
    """Infer the brand from the listing slug, e.g. /dijual/toyota-raize-.../123."""
    match = re.search(r"/dijual/(.+?)/(\d+)\s*$", url)
    if not match:
        return UNKNOWN_BUCKET
    slug = unquote(match.group(1)).lower()
    for brand in BRANDS:
        if slug == brand or slug.startswith(brand + "-"):
            return brand
    return UNKNOWN_BUCKET


def build_plan(urls: list[str]) -> tuple[list[str], list[tuple[str, int, int]]]:
    """Return (selected URLs, per-brand plan rows of (brand, available, picked))."""
    by_brand: dict[str, list[str]] = defaultdict(list)
    for url in urls:
        by_brand[brand_of(url)].append(url)

    rng = random.Random(SEED)
    caps = {}
    for brand, items in by_brand.items():
        if brand == UNKNOWN_BUCKET:
            cap = UNKNOWN_CAP
        elif brand in CAPS:
            cap = CAPS[brand]
        elif len(items) >= MIN_COUNT:
            cap = REST_CAP
        else:
            cap = 0
        caps[brand] = min(cap, len(items))

    total = sum(caps.values())
    if total > PLAN_TARGET:          # scale down so the plan fits the budget
        factor = PLAN_TARGET / total
        caps = {b: (max(20, int(c * factor)) if c else 0) for b, c in caps.items()}

    plan: list[str] = []
    table: list[tuple[str, int, int]] = []
    for brand in sorted(by_brand, key=lambda b: -len(by_brand[b])):
        items = by_brand[brand]
        cap = caps[brand]
        if cap <= 0:
            continue
        plan.extend(rng.sample(items, cap))
        table.append((brand, len(items), cap))
    rng.shuffle(plan)
    return plan, table


def load_done() -> tuple[set[str], int]:
    """Return the ad_ids already written, plus the current row count."""
    done: set[str] = set()
    rows = 0
    if LISTINGS.exists():
        with LISTINGS.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("ad_id"):
                    done.add(str(obj["ad_id"]))
    return done, rows


def load_attempted() -> set[str]:
    if not ATTEMPTED.exists():
        return set()
    with ATTEMPTED.open(encoding="utf-8") as fh:
        return {line.strip() for line in fh if line.strip()}


def fetch_html(url: str) -> tuple[int, bytes]:
    """Download one page with curl. Returns (http_status, body)."""
    cmd = [
        CURL, "-sS", "-L", "--compressed", "--max-time", "60",
        "--retry", "1", "-A", UA, *HEADERS,
        "-o", "-", "-w", "\n__CODE__%{http_code}", url,
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    out = proc.stdout or b""
    idx = out.rfind(b"\n__CODE__")
    if idx == -1:
        return 0, b""
    try:
        code = int(out[idx + 9: idx + 12])
    except ValueError:
        code = 0
    return code, out[:idx]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="print the plan only")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--target", type=int, default=STOP_ON_ROWS)
    args = ap.parse_args()

    if not URLS.exists():
        sys.exit("urls.txt not found. Run enumerate_urls.py first.")

    urls = [u.strip() for u in URLS.read_text(encoding="utf-8").splitlines() if u.strip()]
    plan, table = build_plan(urls)

    print("=" * 62)
    print("CRAWL PLAN")
    print("=" * 62)
    print(f"urls in sitemap     : {len(urls):,}")
    print(f"urls selected       : {len(plan):,}")
    print(f"valid row target    : {args.target:,}")
    print(f"workers             : {args.workers}")
    print()
    print(f"{'brand':<18}{'available':>10}{'selected':>10}")
    for brand, available, picked in table[:24]:
        print(f"{brand:<18}{available:>10,}{picked:>10,}")
    if len(table) > 24:
        rest = sum(p for _, _, p in table[24:])
        print(f"{'(+%d other brands)' % (len(table) - 24):<18}{'':>10}{rest:>10,}")
    print()
    est_gb = len(plan) * 214 / 1024 / 1024
    print(f"estimated transfer  : {est_gb:.1f} GB (gzip, 214 KB per page)")
    print(f"estimated duration  : {len(plan) / (args.workers * 1.4) / 60:.0f} minutes")
    print()

    if args.plan:
        print("(--plan: stopping here, nothing downloaded)")
        return

    DATA.mkdir(exist_ok=True)
    AUDIT.mkdir(exist_ok=True)

    done, rows = load_done()
    attempted = load_attempted()
    todo = [
        u for u in plan
        if u not in attempted
        and re.search(r"/(\d+)\s*$", u)
        and re.search(r"/(\d+)\s*$", u).group(1) not in done
    ]

    print(f"resume: {rows:,} rows already stored, {len(todo):,} urls remaining")
    if rows >= args.target:
        print("target already reached. Nothing to do.")
        return

    lock = threading.Lock()
    state = {"rows": rows, "ok": 0, "fail": 0, "start": time.time(), "stop": False}
    audit_count = {"n": 0}

    list_fh = LISTINGS.open("a", encoding="utf-8")
    fail_fh = FAILURES.open("a", encoding="utf-8")
    att_fh = ATTEMPTED.open("a", encoding="utf-8")

    def handle(url: str) -> None:
        code, body = fetch_html(url)
        with lock:
            att_fh.write(url + "\n")
        if code != 200 or not body:
            with lock:
                state["fail"] += 1
                fail_fh.write(json.dumps({
                    "url": url, "http": code, "at": now_iso()
                }) + "\n")
            return
        html = body.decode("utf-8", "replace")
        row = parse_listing(html, url=url)
        row["scraped_at"] = now_iso()
        valid = not row["errors"]
        with lock:
            if valid and state["rows"] >= args.target:
                return          # quota full, do not write beyond the target
            if valid:
                state["rows"] += 1
                state["ok"] += 1
                list_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                if audit_count["n"] < 10:
                    audit_count["n"] += 1
                    gz = AUDIT / f"{audit_count['n']:03d}_{row['ad_id']}.html.gz"
                    with gzip.open(gz, "wb") as f:
                        f.write(body)
            else:
                state["fail"] += 1
                fail_fh.write(json.dumps({
                    "url": url, "http": code, "errors": row["errors"], "at": now_iso()
                }, ensure_ascii=False) + "\n")
        if state["rows"] >= args.target:
            state["stop"] = True

    print()
    print("crawl started. Ctrl+C to stop (safe, resumable).")
    print()
    tty = sys.stdout.isatty()
    last_print = 0.0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = set()
            it = iter(todo)
            exhausted = False
            while True:
                while (
                    not exhausted
                    and len(pending) < args.workers * 3
                    and not state["stop"]
                ):
                    try:
                        url = next(it)
                    except StopIteration:
                        exhausted = True
                        break
                    pending.add(pool.submit(handle, url))
                if not pending:
                    break
                finished, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                for future in finished:
                    try:
                        future.result()
                    except Exception as exc:  # noqa: BLE001
                        with lock:
                            state["fail"] += 1
                            fail_fh.write(json.dumps({
                                "url": "?", "error": str(exc)[:200], "at": now_iso()
                            }) + "\n")
                now = time.time()
                if now - last_print >= 2 or state["stop"]:
                    last_print = now
                    elapsed = now - state["start"]
                    done_n = state["ok"] + state["fail"]
                    rate = done_n / elapsed if elapsed > 0 else 0
                    left = max(0, args.target - state["rows"])
                    eta = left / rate / 60 if rate > 0 else 0
                    pct = 100 * state["rows"] / args.target
                    msg = (f"[{state['rows']:>6,}/{args.target:,}] {pct:5.1f}%  "
                           f"ok={state['ok']:,} fail={state['fail']:,}  "
                           f"{rate:4.1f}/s  ETA {eta:4.0f}m")
                    if tty:
                        print("\r" + msg + "   ", end="", flush=True)
                    else:
                        print(msg, flush=True)
                if state["stop"]:
                    for future in pending:
                        future.cancel()
                    break
    except KeyboardInterrupt:
        print("\n\ninterrupted by user.")
        state["stop"] = True
    finally:
        list_fh.close()
        fail_fh.close()
        att_fh.close()

    elapsed = time.time() - state["start"]
    lines = [
        "CRAWL SUMMARY",
        f"finished     : {now_iso()}",
        f"duration     : {elapsed/60:.1f} minutes",
        f"valid rows   : {state['rows']:,} (target {args.target:,})",
        f"failures     : {state['fail']:,}",
        f"throughput   : {(state['ok']+state['fail'])/max(elapsed,1):.2f} urls/sec",
        f"output file  : {LISTINGS}",
    ]
    print()
    print("=" * 62)
    for line in lines:
        print(line)
    print("=" * 62)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
