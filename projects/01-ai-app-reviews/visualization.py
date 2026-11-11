"""
visualization.py
================
Phase 4 - Noise Filtering, Sensitivity Analysis & Visualisation
Project: Sentiment Analysis Comparison of the Three Most Popular AI Applications
         (ChatGPT vs Gemini vs Claude) on the Google Play Store

Pipeline:
    1. Load the sentiment-labelled dataset produced in Phase 3.
    2. Flag repetitive / gibberish reviews with a unique-word ratio below the
       threshold. Unlike a word-count rule, this catches spam such as
       "good good good" or long dumps padded with one repeated phrase, while
       keeping legitimate one-word reviews ("good", "nice") that carry real
       sentiment signal.
    3. Save the main analysis dataset.
    4. Build a sensitivity dataset that additionally drops reviews shorter than
       three words, to quantify how much the short-review rule inflates the
       results.
    5. Render:
         - horizontal sentiment bar chart for the main dataset,
         - "Sensitivity Analysis" chart comparing main vs short-filtered,
         - global positive and negative word clouds,
         - one negative word cloud per application.
       Every figure is written to ``images/`` at 300 dpi.

Usage:
    python visualization.py
"""

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend: render to file, no GUI window

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import STOPWORDS, WordCloud

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_CSV = "ai_apps_reviews_sentiment.csv"
MAIN_OUTPUT_CSV = "ai_apps_reviews_clean.csv"
SENSITIVITY_OUTPUT_CSV = "ai_apps_reviews_filtered_short.csv"
IMAGE_DIR = Path("images")

UNIQUE_RATIO_THRESHOLD = 0.35   # below this -> repetitive / gibberish
MIN_WORDS = 3                   # short-review rule used for the sensitivity run
DPI = 300                       # print-quality resolution

APP_ORDER = ["ChatGPT", "Gemini", "Claude"]
SENTIMENT_ORDER = ["positive", "neutral", "negative"]

# High-contrast palette: green / grey / red
SENTIMENT_COLORS = {
    "positive": "#2E7D32",
    "neutral": "#9E9E9E",
    "negative": "#C62828",
}

# Fixed palettes keep every word dark enough to read against a white background
# (matplotlib's default "Greens"/"Reds" ramp fades to a near-invisible tint).
POSITIVE_PALETTE = ["#1B5E20", "#2E7D32", "#388E3C", "#43A047", "#66BB6A"]
NEGATIVE_PALETTE = ["#7F0000", "#B71C1C", "#C62828", "#D32F2F", "#E53935"]

# Extra stopwords: app names and generic product vocabulary would otherwise
# dominate the clouds without carrying sentiment meaning.
CUSTOM_STOPWORDS = {
    "app", "apps", "application", "chatgpt", "gemini", "claude", "gpt",
    "google", "openai", "anthropic", "ai", "use", "using", "used", "user",
    "users", "just", "like", "really", "good", "dont", "doesnt", "im",
    "get", "got", "can", "cant", "will", "now", "even", "still", "much",
    "would", "could", "make", "made", "one", "also", "ever", "every",
}
STOPWORD_SET = set(STOPWORDS) | CUSTOM_STOPWORDS


# ---------------------------------------------------------------------------
# Step 1-4: noise filtering and datasets
# ---------------------------------------------------------------------------
def add_unique_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the unique-word ratio used to detect repetitive spam."""
    tokens = df["review"].fillna("").str.split()
    return df.assign(
        unique_ratio=tokens.map(lambda w: len(set(w)) / len(w) if w else 0.0)
    )


def distribution_table(df: pd.DataFrame) -> pd.DataFrame:
    """Sentiment counts and shares (%) per application."""
    counts = (
        df.groupby(["app_name", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=APP_ORDER, columns=SENTIMENT_ORDER, fill_value=0)
    )
    pct = (counts.div(counts.sum(axis=1), axis=0) * 100).round(1)
    return counts.add_suffix("_n").join(pct.add_suffix("_%"))


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def draw_bars(ax, pct: pd.DataFrame, deltas: dict = None) -> None:
    """Draw grouped horizontal bars for one sentiment distribution table."""
    bar_height = 0.26
    y_base = range(len(APP_ORDER))

    for i, sentiment in enumerate(SENTIMENT_ORDER):
        offsets = [y + (i - 1) * bar_height for y in y_base]
        values = pct[sentiment].values
        bars = ax.barh(
            offsets, values,
            height=bar_height,
            label=sentiment.capitalize(),
            color=SENTIMENT_COLORS[sentiment],
            edgecolor="white", linewidth=0.6,
        )
        for bar, value, app in zip(bars, values, pct.index):
            ax.text(
                bar.get_width() + 0.9, bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%", va="center", ha="left", fontsize=9,
            )
            if deltas and sentiment == "negative" and app in deltas:
                ax.text(
                    bar.get_width() + 12.5, bar.get_y() + bar.get_height() / 2,
                    f"({deltas[app]:+.1f} pp)", va="center", ha="left",
                    fontsize=8.5, color="#B71C1C", style="italic",
                )

    ax.set_yticks(list(y_base))
    ax.set_yticklabels(APP_ORDER, fontsize=11, fontweight="bold")
    ax.invert_yaxis()  # keep the reading order (chatgpt, gemini, claude) top to bottom
    ax.set_xlabel("Share of reviews (%)", fontsize=10.5)
    ax.set_xlim(0, 100)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_sentiment_bars(df: pd.DataFrame, out_path: Path) -> None:
    """Main analysis: sentiment share (%) per application."""
    pct = (
        df.groupby(["app_name", "sentiment"]).size()
        .unstack(fill_value=0)
        .reindex(index=APP_ORDER, columns=SENTIMENT_ORDER, fill_value=0)
    )
    pct = pct.div(pct.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(11, 6.2))
    draw_bars(ax, pct)
    ax.set_title(
        "Sentiment Distribution by Application\n"
        f"Google Play Store reviews, English, n = {len(df):,}",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.legend(title="Sentiment", loc="lower right", frameon=False, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_sensitivity(main_df: pd.DataFrame, short_df: pd.DataFrame,
                     out_path: Path) -> None:
    """Side-by-side comparison: main dataset vs short-review-filtered dataset."""
    def pct_of(frame):
        pct = (
            frame.groupby(["app_name", "sentiment"]).size()
            .unstack(fill_value=0)
            .reindex(index=APP_ORDER, columns=SENTIMENT_ORDER, fill_value=0)
        )
        return pct.div(pct.sum(axis=1), axis=0) * 100

    main_pct, short_pct = pct_of(main_df), pct_of(short_df)
    deltas = {
        app: short_pct.loc[app, "negative"] - main_pct.loc[app, "negative"]
        for app in APP_ORDER
    }

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), sharex=True)

    draw_bars(axes[0], main_pct)
    axes[0].set_title(
        f"Main dataset  (n = {len(main_df):,})", fontsize=12, fontweight="bold"
    )

    draw_bars(axes[1], short_pct, deltas=deltas)
    axes[1].set_title(
        f"After removing reviews < {MIN_WORDS} words  (n = {len(short_df):,})",
        fontsize=12, fontweight="bold",
    )

    fig.suptitle(
        "Sensitivity Analysis\nEffect of dropping short reviews on sentiment shares",
        fontsize=13.5, fontweight="bold", y=1.02,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Sentiment", loc="lower center",
               ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ---------------------------------------------------------------------------
# Word clouds
# ---------------------------------------------------------------------------
def plot_wordcloud(text: str, palette: list, title: str, out_path: Path,
                   max_words: int = 180) -> None:
    """Render a single word cloud using a fixed, high-contrast palette."""
    def pick_color(*_args, **_kwargs):
        return random.choice(palette)

    cloud = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        stopwords=STOPWORD_SET,
        min_font_size=10,
        max_words=max_words,
        min_word_length=3,
        collocations=False,
        prefer_horizontal=0.95,
        color_func=pick_color,
        random_state=42,  # reproducible layout
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    raw = pd.read_csv(INPUT_CSV)
    print("=" * 74)
    print("NOISE FILTERING & VISUALISATION")
    print("=" * 74)
    print(f"Loaded: {INPUT_CSV} ({len(raw):,} rows)")

    # ------------------------------------------------------------------
    # Step 1: flag and drop repetitive / gibberish reviews
    # ------------------------------------------------------------------
    df = add_unique_ratio(raw)
    is_spam = df["unique_ratio"] < UNIQUE_RATIO_THRESHOLD
    n_spam = int(is_spam.sum())

    print(f"\nRepetitive-spam rule : unique_ratio < {UNIQUE_RATIO_THRESHOLD}")
    print(f"Flagged as spam      : {n_spam:,} ({n_spam / len(df) * 100:.2f}%)")
    if n_spam:
        print("Examples of removed spam:")
        for text in df.loc[is_spam, "review"].head(3):
            print(f"  - {str(text)[:90]}")

    main_df = df.loc[~is_spam].copy()

    # ------------------------------------------------------------------
    # Step 2: sensitivity dataset - additionally drop short reviews
    # ------------------------------------------------------------------
    word_counts = main_df["review"].fillna("").str.split().str.len()
    short_mask = word_counts < MIN_WORDS
    short_df = main_df.loc[~short_mask].copy()

    print(f"\nRows in main dataset       : {len(main_df):,}")
    print(f"Short reviews (< {MIN_WORDS} words): {int(short_mask.sum()):,} "
          f"dropped for the sensitivity run")
    print(f"Rows in sensitivity dataset: {len(short_df):,}")

    # ------------------------------------------------------------------
    # Step 3: persist both datasets
    # ------------------------------------------------------------------
    main_columns = ["app_name", "review", "review_raw", "score", "at",
                    "compound", "sentiment", "compound_raw", "sentiment_raw",
                    "unique_ratio"]
    main_df[main_columns].to_csv(MAIN_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    short_df[main_columns].to_csv(SENSITIVITY_OUTPUT_CSV, index=False,
                                  encoding="utf-8-sig")
    print(f"\nSaved: {MAIN_OUTPUT_CSV} ({len(main_df):,} rows)")
    print(f"Saved: {SENSITIVITY_OUTPUT_CSV} ({len(short_df):,} rows)")

    # ------------------------------------------------------------------
    # Step 4: summary tables
    # ------------------------------------------------------------------
    print("\n--- Main dataset: sentiment distribution ---")
    print(distribution_table(main_df).to_string())

    print("\n--- Sensitivity dataset: sentiment distribution ---")
    print(distribution_table(short_df).to_string())

    # ------------------------------------------------------------------
    # Step 5: figures
    # ------------------------------------------------------------------
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    print("\nRendering figures:")

    plot_sentiment_bars(main_df, IMAGE_DIR / "sentiment_distribution.png")
    plot_sensitivity(main_df, short_df,
                     IMAGE_DIR / "sentiment_distribution_sensitivity.png")

    # Global word clouds (three applications combined)
    positive_text = " ".join(main_df.loc[main_df["sentiment"] == "positive", "review"])
    negative_text = " ".join(main_df.loc[main_df["sentiment"] == "negative", "review"])
    plot_wordcloud(
        positive_text, POSITIVE_PALETTE,
        "Positive Reviews - Global Word Cloud\nChatGPT, Gemini & Claude combined",
        IMAGE_DIR / "wordcloud_positive.png",
    )
    plot_wordcloud(
        negative_text, NEGATIVE_PALETTE,
        "Negative Reviews - Global Word Cloud\nChatGPT, Gemini & Claude combined",
        IMAGE_DIR / "wordcloud_negative.png",
    )

    # Per-application negative word clouds (app-specific complaints)
    for app in APP_ORDER:
        is_negative = (main_df["app_name"] == app) & (main_df["sentiment"] == "negative")
        text = " ".join(main_df.loc[is_negative, "review"])
        n_reviews = int(is_negative.sum())
        plot_wordcloud(
            text, NEGATIVE_PALETTE,
            f"Negative Reviews - {app}\n"
            f"Google Play Store, English, n = {n_reviews:,}",
            IMAGE_DIR / f"wordcloud_negative_{app.lower()}.png",
            max_words=150,
        )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
