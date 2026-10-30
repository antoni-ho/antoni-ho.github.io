# ChatGPT vs Gemini vs Claude — Google Play Review Sentiment Analysis

A small end-to-end data analyst project: scrape 9,000 Google Play reviews of the three
most popular AI assistants, clean the text, score every review with VADER, and compare
what each user base is actually complaining about.

**Result in one line:** ChatGPT users are mostly happy, Gemini users are mostly indifferent,
and Claude users are the most negative — because of access and billing, not answer quality.

Live write-up: [antoni-ho.github.io](https://antoni-ho.github.io/#projects)

---

## Business question

Star ratings tell you *how many* users are unhappy, not *why*. This project asks:

1. How does sentiment differ between ChatGPT, Gemini and Claude?
2. What is each app's complaint actually about?
3. Do the star ratings tell the same story as the reviews, or do they hide complaints?

**Short answers**

1. ChatGPT leads on positive sentiment (79.1%); Claude has the highest negative share
   (14.8%), four times ChatGPT's.
2. Complaints differ in kind, not just in volume: access and billing for Claude,
   usefulness for Gemini, output quality for ChatGPT.
3. Mostly they agree (73.2%), but not always — a five-star review can still contain a
   complaint, which is exactly why the text is worth reading.

## Dataset

| Item | Value |
| --- | --- |
| Source | Google Play Store (`google-play-scraper`) |
| Apps | ChatGPT `com.openai.chatgpt`, Gemini `com.google.android.apps.bard`, Claude `com.anthropic.claude` |
| Reviews scraped | 9,000 (3,000 newest per app) |
| Filter | English (`lang=en`), region `us`, sorted by newest |
| Collected on | 28 October 2026 |
| Review window | ≈ 1 day for ChatGPT, ≈ 2 days for Gemini, ≈ 20 days for Claude |
| Analysis set | 8,480 reviews after cleaning and noise filtering |

## Tools

Python · pandas · google-play-scraper · VADER (vaderSentiment) · matplotlib · wordcloud

## Project structure

```
.
├── scrape_reviews.py                  # Phase 1 – collect 3,000 reviews per app
├── clean_data.py                      # Phase 2 – lowercase, strip punctuation/emoji, drop empties
├── sentiment_analysis.py              # Phase 3 – VADER scores, labels, validation
├── visualization.py                   # Phase 4 – noise filter, charts, word clouds
│
├── ai_apps_reviews.csv                # 9,000 raw reviews
├── ai_apps_reviews_preprocessed.csv   # 8,489 cleaned reviews
├── ai_apps_reviews_sentiment.csv      # 8,489 + compound score and label
├── ai_apps_reviews_clean.csv          # 8,480 main analysis set
├── ai_apps_reviews_filtered_short.csv # 4,168 sensitivity set (short reviews removed)
│
├── images/                            # 300 dpi figures + web-optimised copies in images/web/
├── index.html
└── style.css
```

## How to run

```bash
python -m pip install google-play-scraper pandas vaderSentiment matplotlib wordcloud
```

Run the scripts in order:

```bash
python scrape_reviews.py        # ~3-4 minutes, hits the Play Store
python clean_data.py
python sentiment_analysis.py
python visualization.py
```

All CSV outputs are committed, so you can skip `scrape_reviews.py` and start from
`clean_data.py` if you only want to inspect the analysis.

## Methodology

### 1. Scraping

Paged 200 reviews at a time with `Sort.NEWEST`, deduplicated by `reviewId`, and added an
`app_name` column before concatenating the three apps into one file.

### 2. Cleaning

- Drop rows with missing review text (0 rows).
- Lowercase everything.
- Remove apostrophes *without* inserting a space, so contractions stay one token:
  `it's → its`, `don't → dont`.
- Remove every character that is not a letter (`[^a-z\s] → space`), which also drops
  digits, emoji and punctuation.
- Drop rows that become empty afterwards (511 rows: 239 emoji-only, 272 in non-Latin
  scripts that slipped past Play Store's English filter).

### 3. Sentiment scoring

VADER compound score in `[-1, 1]`, labelled with the recommended cut-offs:

| Compound | Label |
| --- | --- |
| `>= 0.05` | positive |
| `<= -0.05` | negative |
| otherwise | neutral |

### 4. Noise filtering and sensitivity check

Repetitive spam is detected with a unique-word ratio (`unique words / total words`)
below **0.35**, which removes 9 rows (0.1%) such as `"good good good"`.

A stricter alternative — dropping every review shorter than 3 words — was also tested and
**rejected**: it removes 50.8% of the data and inflates the negative share of every app
(Gemini 7.1% → 14.9%). That experiment is kept as `ai_apps_reviews_filtered_short.csv`
and shown as a sensitivity chart in the write-up.

### 5. Validation

The main text set and the raw text were scored separately and compared against the star
rating:

| Text version | Agreement with star rating |
| --- | --- |
| Raw text | 73.5% |
| Cleaned text | 73.2% |

A 0.3-point gap means the cleaning pipeline does not destroy the sentiment signal.
The star rating is treated as a rough proxy, not ground truth: a five-star review can
still contain a complaint.

## Key findings

| Metric | ChatGPT | Gemini | Claude |
| --- | --- | --- | --- |
| Positive | **79.1%** | 70.9% | 71.2% |
| Neutral | 17.2% | **22.1%** | 13.9% |
| Negative | 3.7% | 7.1% | **14.8%** |

- **ChatGPT** leads on satisfaction; its rare complaints are about image generation and
  waiting time.
- **Gemini** has the most neutral reviews — users are indifferent rather than angry.
- **Claude** has four times ChatGPT's negative rate, and about two in five of those
  negative reviews mention an account, limit or billing problem (38.1%, versus 8.6% for
  Gemini).

## Business implications

The answers above say *what happened*; this section says *what a product team could do
about it*. These are recommendations, not measured outcomes.

- **Claude** — people are not saying the answers are bad; they are saying they cannot get
  in or use it enough. Fixing limits, checks and billing would help more than a smarter
  model.
- **Gemini** — people are not angry, they are not convinced the app is useful. Showing
  what it can do matters more than polishing the answers.
- **ChatGPT** — people already trust the answers. The few complaints are about waiting,
  and about pictures that fail to generate.

## Limitations

- Only the 3,000 newest English reviews per app were collected, on 28 October 2026.
  Because review volume differs, that window covers roughly one day for ChatGPT but
  about three weeks for Claude, so the three samples are not equally time-balanced.
- Reviewers on Google Play are not a representative sample of all app users.
- VADER is lexicon-based and misses sarcasm, negation and context.
- Star ratings are a proxy for sentiment, not a labelled ground truth.
- The "access complaint" share is measured with a hand-built keyword list (account, limit,
  limited, subscription, verification, paid, payment, banned, usage, login, refund, price),
  so treat it as directional evidence rather than a precise rate.

---

**Antoni** — Data Analyst · [GitHub](https://github.com/antoni-ho) ·
[LinkedIn](https://www.linkedin.com/in/antoni-h20/)
