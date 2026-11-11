# Used Car Price Retention — a hedonic model of the Indonesian market

An end-to-end data project: enumerate 59,037 live listings on Mobil123, scrape 12,000,
clean them into 10,484 unique cars, and price them with a model that holds model and year
constant — to find out what the market actually charges for.

**Result in one line:** in asking prices, the model and the year decide almost everything;
the badge is worth up to +27% on top of specifications, while transmission, odometer and
city move the price by single digits — or, once you compare like with like, by almost
nothing.

Live write-up: [antoni-ho.github.io/projects/02-used-car-market](https://antoni-ho.github.io/projects/02-used-car-market/)

---

## Business question

A price prediction model answers "how much does it cost". This project asks four narrower
questions that a prediction alone cannot:

1. Which brands actually hold their value, once each model is compared against itself?
2. Is the automatic premium real, or is it measuring which segments still offer a manual?
3. What does the odometer do to the asking price, holding model and year fixed?
4. Does location change the price of the same car?

**Short answers**

1. Suzuki loses 4.4% of value per year of age, Toyota 5.6%, BMW 13.0% — a 3× spread.
2. Almost entirely the second thing: +125% pooled, **+5.8%** like-for-like.
3. Far less than it looks: −74% pooled, **−4.1%** like-for-like.
4. No. The four best-covered provinces sit within 0.7 percentage points of each other.

## Dataset

| Item | Value |
| --- | --- |
| Source | Mobil123 (`mobil123.com`), listing sitemap + detail pages |
| Listings enumerated | 59,037 unique URLs |
| Listings scraped | 12,000 (stratified by brand, seed 42) |
| After de-duplication | **10,484 unique cars** (12.3% duplicates removed) |
| Brands / model series | 57 / 503 |
| Collected on | 10 November 2026 |
| Price type | asking price, never a transaction price |
| Seller mix | 97.7% dealer listings |
| Transfer | ~3.1 GB compressed, 7.7 hours at 0.43 URL/s |

Why 12,000 of 59,037, and what `year` means: see [METHODOLOGY.md](METHODOLOGY.md).

## Tools

Python · pandas · numpy · scikit-learn · matplotlib

## Project structure

```
.
├── enumerate_urls.py      # stage 1a – sitemap → urls.txt
├── scrape.py              # stage 1b – crawl detail pages → data/listings.jsonl
├── parse.py               # stage 1c – JSON-LD + ga:cad meta → one dict per car
├── clean.py               # stage 2a – de-duplicate, filter, derive, weight → data/clean.csv.gz
├── audit.py               # stage 2b – field completeness, duplicates, cell sizes, precision
├── eda.py                 # stage 2c – the eight figures
├── verify.py              # stage 2d – every headline number recomputed another way
├── model.py               # stage 3  – baseline, ridge, random forest, gradient boosting
│
├── METHODOLOGY.md         # sampling design, what `year` means, why the odometer is weak
├── data/                  # clean.csv.gz, predictions.csv.gz, generated reports
└── images/                # the eight figures used on the write-up page
```

## How to run

```bash
python -m pip install pandas numpy matplotlib scikit-learn
```

The crawl is the slow part and is optional: `data/clean.csv.gz` is committed, so the
analysis can start from stage 2.

```bash
python enumerate_urls.py   # ~1 min   → urls.txt
python scrape.py           # ~7 hours → data/listings.jsonl   (optional, resumable)
python clean.py            #          → data/clean.csv.gz
python audit.py            #          → data/audit_report.txt
python eda.py              #          → images/*.png, data/eda_report.txt
python verify.py           #          → data/verify_report.txt
python model.py            #          → data/model_report.txt, data/predictions.csv.gz
```

`scrape.py` is resumable: Ctrl+C is safe, and re-running continues where it stopped.

## Methodology

### 1. Collection

URL inventory from six `listing_details_*.xml.gz` sitemap shards, then a stratified random
sample by brand. Brand quotas are deliberately unequal (Toyota 2,500, Porsche 220) so small
brands stay analysable, which means the sample is **not self-weighting** — every row carries
a `weight = market share / sample share`.

The search pages were tested first and abandoned: they are a JavaScript shell with no
listing markup. Detail pages are server-rendered and expose everything in two structured
blocks — `ga:cad:details:*` meta tags and one schema.org `Car` JSON-LD object — so the
parser reads structured fields, not HTML tables.

### 2. Cleaning

12,000 → 10,484: 1,471 duplicate listings (identical brand, model, year, price, mileage and
city), 45 vehicles outside 2005–2026, and implausible odometer readings (1,606 missing
mileage, 855 missing engine size) left as nulls rather than imputed.

### 3. Modelling

Target `log(price)`, so coefficients read as percentages. Features: age, odometer, km/year,
engine size and power, seats, transmission, body, fuel, province, seller type, brand, model,
plus flags extracted from the listing text (`pajak hidup`, `tangan pertama`, service record,
`nego`).

Validation is two-layered because two different questions are being asked:

| Scheme | Question | RMSE(log) | R² | MdAPE |
| --- | --- | --- | --- | --- |
| Baseline: median of the same model-year | is ML worth it at all? | 0.2325 | 0.935 | **5.9%** |
| Ridge | — | 0.2605 | 0.918 | 11.3% |
| Random forest | — | 0.1890 | 0.957 | 6.6% |
| Gradient boosting | deployment case | **0.1515** | **0.972** | **5.6%** |
| Gradient boosting, unseen model series | does it generalise? | 0.3827 | 0.824 | 22.0% |

The baseline is strong, and that is a finding, not a nuisance: a median lookup by model and
year already achieves a 5.9% median error. Gradient boosting earns its place mainly in the
tails. Withhold model identity and the error more than doubles.

## Key findings

| Comparison | Pooled | Like-for-like |
| --- | --- | --- |
| Automatic vs manual | +125.3% | **+5.8%** (95% CI +3.1 to +10.6) |
| Highest vs lowest odometer quintile | −74.5% | **−4.1%** (95% CI −4.8 to −3.2) |
| Province spread (same car) | — | **0.7 pp** |
| Badge premium over specifications | — | Land Rover +27.5% … Wuling −20.6% |

- **Retention is a brand trait.** Suzuki 4.4%/yr, Daihatsu 4.9, Toyota 5.6, Mitsubishi 5.9,
  Honda 6.9, Mazda 8.2, Mercedes-Benz 12.1, Hyundai 12.5, BMW 13.0.
- **Depreciation is not linear.** An Alphard loses half its value in about four years and
  then flattens; an Avanza declines gently and stays there.
- **The odometer is weakly priced**, but it is not ignored: the honest statement is
  "−4% for a three-fold mileage difference in asking prices", not "buyers ignore mileage".
- **Location is not a pricing dimension** on this platform, at least across Java.
- **Chinese brands are specification-rich and price-poor**: Wuling Cortez −31%, BYD Atto 1
  −26.7% against what their specifications predict.

## Business implications

These are interpretations, not measured outcomes.

- **Buyers** — the sticker is the model and the year. Mileage and gearbox are where the
  negotiation actually lives, because the sticker barely charges for them.
- **Sellers** — a service history and a low odometer do not show up in the asking price on
  this platform; they may still matter at the table, but not on paper.
- **Analysts** — most of the apparent signal in this market is composition. Holding model
  and year constant is the cheapest way to see through it, and a median-by-model-year
  baseline is the number any model must beat.

## Limitations

- One platform, one snapshot. Mobil123 is dealer-heavy (97.7%); OLX and the certified
  retailers price differently.
- Asking prices only. Negotiated transaction prices are unobservable, and mileage may be
  discounted far more during negotiation than on the sticker.
- Retention figures are **nominal**, not inflation-adjusted. With ~4% inflation, real
  depreciation is roughly double the numbers above.
- Odometer readings are plausibly rolled back, which attenuates any real mileage effect.
- Only four provinces have ≥150 listings, so the location result covers Java, not Indonesia.
- 918 sellers hold 10,484 listings (one dealer has 386), so observations are not fully
  independent.
- Three estimates are flagged as fragile in [METHODOLOGY.md](METHODOLOGY.md): Hyundai's
  retention, the transmission premium above age five, and the specification-only premiums
  for brands with few listings.

---

**Antoni** — Data Analyst · [GitHub](https://github.com/antoni-ho) ·
[LinkedIn](https://www.linkedin.com/in/antoni-h20/)
