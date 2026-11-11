"""
clean_data.py
=============
Phase 2 - Data Cleaning & Text Preprocessing
Project: Sentiment Analysis Comparison of the Three Most Popular AI Applications
         (ChatGPT vs Gemini vs Claude) on the Google Play Store

Pipeline:
    1. Load the raw scrape result (``ai_apps_reviews.csv``).
    2. Drop rows whose review text is missing (NaN) or blank.
    3. Lowercase every review.
    4. Remove apostrophes without inserting a space, so contractions collapse to
       a single token ("it's" -> "its", "don't" -> "dont").
    5. Strip every character that is not a letter (digits, emoji, punctuation)
       using a regular expression.
    6. Drop rows that become empty after cleaning (e.g. emoji-only reviews).
    7. Save the cleaned dataset to ``ai_apps_reviews_preprocessed.csv``.

Usage:
    python clean_data.py
"""

import re

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_CSV = "ai_apps_reviews.csv"
OUTPUT_CSV = "ai_apps_reviews_preprocessed.csv"

# Apostrophes are removed WITHOUT a space so contractions stay a single token:
# "it's" -> "its", "don't" -> "dont" (instead of the noisy "it s", "don t").
# Both the straight and the typographic apostrophe are handled.
APOSTROPHE_PATTERN = re.compile(r"['\u2019\u2018`]")

# Keep only lowercase letters and whitespace; everything else becomes a space.
# A space (not an empty string) is used so that "good,thank" -> "good thank"
# instead of "goodthank".
NON_LETTER_PATTERN = re.compile(r"[^a-z\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")

# Output column order: cleaned text first, raw text kept for traceability
OUTPUT_COLUMNS = ["app_name", "review", "review_raw", "score", "at"]


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase the text, drop apostrophes, remove every non-letter character."""
    text = str(text).lower()
    text = APOSTROPHE_PATTERN.sub("", text)
    text = NON_LETTER_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def main() -> int:
    # --------------------------------------------------------------
    # 1. Load the raw dataset
    # --------------------------------------------------------------
    df = pd.read_csv(INPUT_CSV)
    print("=" * 70)
    print("DATA CLEANING & TEXT PREPROCESSING")
    print("=" * 70)
    print(f"Loaded                : {INPUT_CSV}")
    print(f"Rows before cleaning  : {len(df):,}")

    # --------------------------------------------------------------
    # 2. Drop missing / blank review text
    # --------------------------------------------------------------
    df = df.rename(columns={"content": "review"})
    df["review"] = df["review"].astype("string").str.strip()
    missing = df["review"].isna() | (df["review"] == "")
    n_missing = int(missing.sum())
    df = df.loc[~missing].copy()
    print(f"Dropped (NaN/blank)   : {n_missing:,}")

    # --------------------------------------------------------------
    # 3. Preserve the raw text, then clean it (lowercase + regex)
    # --------------------------------------------------------------
    df["review_raw"] = df["review"]
    df["review"] = df["review"].map(clean_text)

    # --------------------------------------------------------------
    # 4. Drop rows that lost all content (e.g. emoji-only reviews)
    # --------------------------------------------------------------
    empty_after = df["review"] == ""
    n_empty_after = int(empty_after.sum())
    df = df.loc[~empty_after].copy()
    print(f"Dropped (empty after) : {n_empty_after:,}")

    # --------------------------------------------------------------
    # 5. Save the cleaned dataset
    # --------------------------------------------------------------
    df["at"] = pd.to_datetime(df["at"], errors="coerce")
    df = df[OUTPUT_COLUMNS].reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"Rows after cleaning   : {len(df):,}")
    print(f"File saved            : {OUTPUT_CSV}")

    # --------------------------------------------------------------
    # 6. Summary
    # --------------------------------------------------------------
    print("\nRows per application:")
    print(df["app_name"].value_counts().to_string())

    print("\nBefore vs after (first 5 rows):")
    for i, row in df.head(5).iterrows():
        print(f"\n  [{i}] {row['app_name']}")
        print(f"      before: {row['review_raw']}")
        print(f"      after : {row['review']}")

    print("\nFirst 5 rows of the cleaned dataset (review, app_name):")
    with pd.option_context("display.max_colwidth", 90, "display.width", 160):
        print(df[["review", "app_name"]].head(5).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
