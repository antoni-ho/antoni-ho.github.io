"""
sentiment_analysis.py
=====================
Phase 3 - Sentiment Analysis
Project: Sentiment Analysis Comparison of the Three Most Popular AI Applications
         (ChatGPT vs Gemini vs Claude) on the Google Play Store

Method:
    Lexicon-based sentiment scoring with VADER (Valence Aware Dictionary and
    sEntiment Reasoner), which is designed for short, informal user-generated
    text such as app store reviews.

    Each review receives a ``compound`` score in [-1, 1] and is then labelled:
        compound >=  0.05  -> positive
        compound <= -0.05  -> negative
        otherwise          -> neutral

    The analysis is executed twice, on two text versions, to quantify how much
    the preprocessing choices from Phase 2 affect the result:
        * ``review``      : heavily normalised (lowercase, no punctuation/emoji)
        * ``review_raw``  : original text (keeps punctuation, emphasis, emoji)

    Finally, the labels are validated against the star rating, which acts as an
    independent proxy for the user's true sentiment.

Usage:
    python sentiment_analysis.py

Output:
    ai_apps_reviews_sentiment.csv
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_CSV = "ai_apps_reviews_preprocessed.csv"
OUTPUT_CSV = "ai_apps_reviews_sentiment.csv"

POSITIVE_THRESHOLD = 0.05    # VADER recommended cut-off
NEGATIVE_THRESHOLD = -0.05   # VADER recommended cut-off

# Star rating used as the reference label for validation
STAR_TO_SENTIMENT = {1: "negative", 2: "negative", 3: "neutral",
                     4: "positive", 5: "positive"}

LABELS = ["positive", "neutral", "negative"]

ANALYZER = SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compound_of(text: str) -> float:
    """Return the VADER compound score for a single review."""
    return ANALYZER.polarity_scores(str(text))["compound"]


def to_label(compound: float) -> str:
    """Map a compound score to positive / neutral / negative."""
    if compound >= POSITIVE_THRESHOLD:
        return "positive"
    if compound <= NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def distribution(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Count and percentage of each sentiment label per application."""
    counts = (
        df.groupby(["app_name", column])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=LABELS, fill_value=0)
    )
    totals = counts.sum(axis=1)
    pct = (counts.div(totals, axis=0) * 100).round(1)
    out = counts.add_suffix("_n").join(pct.add_suffix("_%"))
    out.insert(0, "total", totals)
    return out


def main() -> int:
    print("=" * 78)
    print("SENTIMENT ANALYSIS - VADER LEXICON")
    print("=" * 78)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded                : {INPUT_CSV}")
    print(f"Rows analysed         : {len(df):,}")

    # --------------------------------------------------------------
    # 1. Score both text versions
    # --------------------------------------------------------------
    df["compound"] = df["review"].map(compound_of)
    df["sentiment"] = df["compound"].map(to_label)

    df["compound_raw"] = df["review_raw"].map(compound_of)
    df["sentiment_raw"] = df["compound_raw"].map(to_label)

    # --------------------------------------------------------------
    # 2. Sentiment distribution per application
    # --------------------------------------------------------------
    print("\n--- Sentiment distribution (cleaned text) ---")
    print(distribution(df, "sentiment").to_string())

    print("\n--- Sentiment distribution (raw text) ---")
    print(distribution(df, "sentiment_raw").to_string())

    # --------------------------------------------------------------
    # 3. Mean compound score per application
    # --------------------------------------------------------------
    print("\n--- Mean compound score per application ---")
    mean_tbl = df.groupby("app_name")[["compound", "compound_raw"]].mean().round(3)
    mean_tbl["mean_star"] = df.groupby("app_name")["score"].mean().round(2)
    mean_tbl = mean_tbl.sort_values("compound", ascending=False)
    print(mean_tbl.to_string())

    # --------------------------------------------------------------
    # 4. Validation against star rating
    # --------------------------------------------------------------
    df["star_sentiment"] = df["score"].map(STAR_TO_SENTIMENT)
    acc_clean = (df["sentiment"] == df["star_sentiment"]).mean()
    acc_raw = (df["sentiment_raw"] == df["star_sentiment"]).mean()
    print("\n--- Agreement with star rating (reference label) ---")
    print(f"  cleaned text : {acc_clean:.1%}")
    print(f"  raw text     : {acc_raw:.1%}")

    # --------------------------------------------------------------
    # 5. Save the enriched dataset
    # --------------------------------------------------------------
    columns = ["app_name", "review", "review_raw", "score", "at",
               "compound", "sentiment", "compound_raw", "sentiment_raw"]
    df[columns].to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nFile saved            : {OUTPUT_CSV}")
    print(f"Columns               : {', '.join(columns)}")

    # --------------------------------------------------------------
    # 6. Sample rows
    # --------------------------------------------------------------
    print("\n--- Sample: 3 most positive and 3 most negative (cleaned text) ---")
    cols = ["app_name", "review", "compound", "sentiment"]
    with pd.option_context("display.max_colwidth", 70, "display.width", 170):
        print("\nTop positive:")
        print(df.nlargest(3, "compound")[cols].to_string(index=False))
        print("\nTop negative:")
        print(df.nsmallest(3, "compound")[cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
