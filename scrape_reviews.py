"""
scrape_reviews.py
=================
Phase 1 - Data Acquisition (Scraping)
Project: Sentiment Analysis Comparison of the Three Most Popular AI Applications
         (ChatGPT vs Gemini vs Claude) on the Google Play Store

Retrieves the 3,000 most recent English-language reviews for each application,
adds an ``app_name`` column to every record, and consolidates the results into
a single CSV file.

Usage:
    python scrape_reviews.py

Output:
    ai_apps_reviews.csv  (9,000 rows: 3,000 ChatGPT + 3,000 Gemini + 3,000 Claude)
"""

import sys
import time

import pandas as pd
from google_play_scraper import Sort, reviews

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
APPS = [
    {"app_name": "ChatGPT", "app_id": "com.openai.chatgpt"},
    {"app_name": "Gemini", "app_id": "com.google.android.apps.bard"},
    {"app_name": "Claude", "app_id": "com.anthropic.claude"},
]

TARGET_PER_APP = 3000   # number of reviews to collect per application
LANG = "en"             # restrict results to English-language reviews
COUNTRY = "us"          # Play Store region used for the request
PAGE_SIZE = 200         # reviews requested per call (pagination batch size)
OUTPUT_CSV = "ai_apps_reviews.csv"


# ---------------------------------------------------------------------------
# Scraping functions
# ---------------------------------------------------------------------------
def scrape_app(app_name: str, app_id: str, target: int = TARGET_PER_APP) -> list:
    """Collect `target` most recent unique reviews for a single application."""
    collected = {}
    token = None
    page = 0

    while len(collected) < target:
        page += 1
        try:
            batch, token = reviews(
                app_id,
                lang=LANG,
                country=COUNTRY,
                sort=Sort.NEWEST,
                count=PAGE_SIZE,
                continuation_token=token,
            )
        except Exception as exc:  # network or rate-limit error -> stop safely
            print(f"  [warning] page {page} failed: {exc}")
            break

        if not batch:
            break

        # Deduplicate by reviewId (Google occasionally returns repeated reviews)
        for row in batch:
            collected.setdefault(row["reviewId"], row)

        print(
            f"  page {page:>2}: +{len(batch):>3} reviews "
            f"-> {len(collected):>4}/{target} unique",
            flush=True,
        )

        # No further pages are available
        if token is None or getattr(token, "token", None) is None:
            break

        time.sleep(0.4)  # polite delay to avoid throttling

    rows = list(collected.values())[:target]
    for row in rows:
        row["app_name"] = app_name
    return rows


def main() -> int:
    all_reviews = []

    print("=" * 70)
    print("GOOGLE PLAY STORE REVIEW SCRAPING")
    print(f"Target: {TARGET_PER_APP:,} most recent English reviews per application")
    print("=" * 70)

    for app in APPS:
        print(f"\n[{app['app_name']}] {app['app_id']}")
        rows = scrape_app(app["app_name"], app["app_id"])
        print(f"  -> completed: {len(rows):,} reviews")
        if len(rows) < TARGET_PER_APP:
            print(
                f"  [note] only {len(rows):,} reviews were available/retrievable "
                f"(target {TARGET_PER_APP:,})."
            )
        all_reviews.extend(rows)

    if not all_reviews:
        print("\nFAILED: no data could be retrieved.", file=sys.stderr)
        return 1

    df = pd.DataFrame(all_reviews)

    # ------------------------------------------------------------------
    # Reorder columns: app_name first; user avatar URLs are not required
    # ------------------------------------------------------------------
    columns = ["app_name", "reviewId", "userName", "content", "score",
               "thumbsUpCount", "reviewCreatedVersion", "at", "replyContent",
               "repliedAt"]
    df = df[[k for k in columns if k in df.columns]]
    df["at"] = pd.to_datetime(df["at"], errors="coerce")
    df = df.sort_values(["app_name", "at"], ascending=[True, False]).reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULT SUMMARY")
    print("=" * 70)
    print(f"File saved       : {OUTPUT_CSV}")
    print(f"Total rows       : {len(df):,}")
    print("\nReviews per application:")
    print(df["app_name"].value_counts().to_string())
    print("\nReview date range:")
    print(f"  {df['at'].min()}  ->  {df['at'].max()}")
    print("\nFirst 5 rows:")
    with pd.option_context("display.max_columns", None,
                           "display.width", 200,
                           "display.max_colwidth", 60):
        print(df.head(5).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
