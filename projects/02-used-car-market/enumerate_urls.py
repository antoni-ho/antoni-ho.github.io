"""
Stage 1 - enumerate active listing URLs from the Mobil123 sitemap.

Reads https://www.mobil123.com/sitemap.xml, follows the six
`listing_details_*.xml.gz` shards and writes every unique listing URL to
`urls.txt`. No listing content is fetched here, only the URL inventory.

Why detail pages and not the search pages?
  We tested the search pages first (`/mobil-dijual/...`). They return a
  JavaScript shell with no listing markup - the <title> still contains the
  literal template placeholder "{site_name}" and the page carries zero
  listing entries. Detail pages are server-rendered and expose both JSON-LD
  (schema.org/Car) and `ga:cad:details:*` meta tags, so they are the only
  usable source.

Note on robots.txt: only `/ajax/` and `/paths/route.js` are disallowed on
Mobil123, so listing pages may be crawled.

Usage:
    python enumerate_urls.py

Output:
    urls.txt   one listing URL per line
"""

from __future__ import annotations

import gzip
import shutil
import subprocess
import sys
import time
from pathlib import Path
from xml.etree import ElementTree as ET

BASE = "https://www.mobil123.com"
SITEMAP = f"{BASE}/sitemap.xml"
OUT = Path(__file__).resolve().parent / "urls.txt"

CURL = shutil.which("curl") or "curl.exe"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
HEADERS = [
    "-H", "Accept-Language: id-ID,id;q=0.9,en;q=0.8",
    "-H", 'sec-ch-ua: "Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "-H", "sec-ch-ua-mobile: ?0",
    "-H", 'sec-ch-ua-platform: "Windows"',
]


def fetch(url: str, tries: int = 3) -> bytes:
    """GET a URL with curl. Returns the raw body, still gzipped if it was."""
    last = "unknown"
    for attempt in range(tries):
        cmd = [
            CURL, "-sS", "-L", "--compressed", "--max-time", "90",
            "-A", UA, *HEADERS, "-o", "-", "-w", "\n__CODE__%{http_code}", url,
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=120)
        out = proc.stdout or b""
        idx = out.rfind(b"\n__CODE__")
        code = int(out[idx + 9: idx + 12]) if idx != -1 else 0
        if code == 200:
            return out[:idx]
        last = f"HTTP {code}"
        time.sleep(2 + attempt * 3)
    raise RuntimeError(f"failed to fetch {url} ({last})")


def parse_locs(xml_bytes: bytes) -> list[str]:
    """Return every <loc> text, namespace-agnostic."""
    root = ET.fromstring(xml_bytes)
    return [
        el.text.strip()
        for el in root.iter()
        if el.tag.split("}")[-1] == "loc" and el.text
    ]


def gunzip_if_needed(raw: bytes) -> bytes:
    return gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw


def main() -> None:
    print("reading sitemap index ...")
    shards = sorted(
        u for u in parse_locs(gunzip_if_needed(fetch(SITEMAP)))
        if "listing_details_" in u
    )
    print(f"found {len(shards)} listing_details shards")

    all_urls: list[str] = []
    for i, shard in enumerate(shards, 1):
        urls = [
            u for u in parse_locs(gunzip_if_needed(fetch(shard)))
            if u.startswith("http")
        ]
        print(f"  [{i}/{len(shards)}] {shard.rsplit('/', 1)[-1]}: {len(urls):,} urls")
        all_urls.extend(urls)
        time.sleep(1.0)

    unique = list(dict.fromkeys(all_urls))
    OUT.write_text("\n".join(unique), encoding="utf-8")
    print()
    print(f"total urls   : {len(all_urls):,}")
    print(f"unique urls  : {len(unique):,}")
    print(f"duplicates   : {len(all_urls) - len(unique):,}")
    print(f"written to   : {OUT}")


if __name__ == "__main__":
    sys.exit(main())
