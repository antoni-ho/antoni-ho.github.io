# Indonesian used-car prices — what the market actually charges for

An end-to-end data project: enumerate 59,037 live listings on Mobil123, scrape 12,000,
clean them into 10,484 unique cars, and model the asking price so that model and year are
held constant — to find out what the market is really charging for.

**Result in one line:** in asking prices, the model and the year decide almost everything;
the badge adds up to +27% on top of specifications, while transmission, odometer and city
move the price by single digits — or, once you compare like with like, by almost nothing.

Live write-up:
[antoni-ho.github.io/projects/02-used-car-market](https://antoni-ho.github.io/projects/02-used-car-market/)

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
| After de-duplication | **10,484 unique cars** (1,471 duplicates, 12.3%, removed) |
| Brands / model series | 57 / 503 |
| Collected on | 10 November 2026 |
| Price type | asking price, never a transaction price |
| Seller mix | 97.7% dealer listings |
| Transfer | ~3.1 GB compressed, 7.7 hours at 0.43 URL/s |

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

### 1. Collecting the listings

URL inventory from six `listing_details_*.xml.gz` sitemap shards, then a stratified random
sample by brand: quotas are deliberately unequal (Toyota 2,500, Porsche 220) so that small
brands stay analysable. That makes the sample **not self-weighting**, so every row carries a
`weight = market share ÷ sample share`. Without it, Chinese brands would look 34% larger than
they are; with it, the weighted shares reproduce the market.

The search pages were tested first and abandoned: they are a JavaScript shell with no listing
markup. Detail pages are server-rendered and expose everything in two structured blocks —
`ga:cad:details:*` meta tags and one schema.org `Car` JSON-LD object — so the parser reads
structured fields, not HTML tables.

**Why 12,000 and not 59,037.** The binding constraint is bias, not sample size. All 59,037
URLs come from the same dealer-dominated platform at asking prices; a five-times larger crawl
shrinks random error, not that bias. Within a (model, year) cell the dispersion of log price is
only σ = 0.12, so a median by model-year is already accurate at small n (5.9% median error),
and the crawl itself is rate-limited to 0.43 URLs/sec: 59,037 would take ~38 hours and 12.6 GB
against 7.7 hours and 3.1 GB. The analysis unit is a **model series**, not a single cell: 61
series have ≥50 listings across ≥5 years and cover 53% of the rows — enough for retention
curves, the transmission comparison and the regression.

**What `year` means.** It is the **vehicle's model/production year**, taken from JSON-LD
`vehicleModelDate` and the `year` meta tag. It is not the listing date and has nothing to do
with the dealer: every row was scraped in November 2026, so a "2020" row is a car
*manufactured* in 2020 and on sale now. The thin 2020 cohort (389 cars against 1,282 for 2022)
is therefore a production effect — few cars were built and sold that year — not a listing
artefact.

### 2. Cleaning

12,000 → 10,484: 1,471 duplicate listings (identical brand, model, year, price, mileage and
city), 45 vehicles outside 2005–2026, and implausible odometer readings. 1,606 rows have no
mileage and 855 no engine size; these are left as nulls rather than imputed.

### 3. Modelling

Target `log(price)`, so coefficients read as percentages. Features: age, odometer, km/year,
engine size and power, seats, transmission, body, fuel, province, seller type, brand, model,
plus flags extracted from the listing text. Validation has two layers, because two different
questions are being asked:

| Scheme | Question | RMSE(log) | R² | MdAPE |
| --- | --- | --- | --- | --- |
| Baseline: median of the same model-year | is ML worth it at all? | 0.2325 | 0.935 | **5.9%** |
| Ridge | — | 0.2605 | 0.918 | 11.3% |
| Random forest | — | 0.1890 | 0.957 | 6.6% |
| **Gradient boosting** (used in the write-up) | deployment case | **0.1515** | **0.972** | **5.6%** |
| Gradient boosting, unseen model series | does it generalise? | 0.3827 | 0.824 | 22.0% |

The baseline is strong, and that is a finding, not a nuisance: a median lookup by model and
year already achieves a 5.9% median error. Gradient boosting earns its place mainly in the
tails. Withhold model identity and the error more than doubles.

### 4. Checking the headline numbers

`verify.py` recomputes every headline figure with an independent estimator, so no result
survives only because of one modelling choice:

| Claim | Primary estimator | Independent check | Agreement |
| --- | --- | --- | --- |
| Brand retention | log-price slope per series | two-point ratio p(8)/p(3) | rank corr 0.87, median gap 1.1 pp |
| Automatic premium | median of per-cell price ratios | weighted mean of log ratios | +5.8% vs +7.2% |
| Odometer effect | model-year contrast | fixed-effect regression slope | −4.1% vs −6.3% per 5× km |

Two estimates are flagged as fragile rather than reported as solid: Hyundai's retention
(87.5% fitted, 92.9% two-point) and the automatic premium above age five, which rests on a
single model-year cell.

## Key findings

| Comparison | Pooled | Like-for-like |
| --- | --- | --- |
| Automatic vs manual | +125.3% | **+5.8%** (95% CI +3.1 to +10.6) |
| Highest vs lowest odometer quintile | −74.5% | **−4.1%** (95% CI −4.8 to −3.2) |
| Province spread (same car) | — | **0.7 pp** |
| Badge premium over specifications | — | Land Rover +27.5% … Wuling −20.6% |

- **Retention is a brand trait.** Suzuki 4.4%/yr, Daihatsu 4.9, Toyota 5.6, Mitsubishi 5.9,
  Honda 6.9, Mazda 8.2, Mercedes-Benz 12.1, Hyundai 12.5, BMW 13.0.
- **Depreciation is not linear.** An Alphard loses about half its value in four years and then
  flattens; an Avanza declines gently and stays there.
- **The odometer is weakly priced**, but it is not ignored: the honest statement is "4% for a
  three-fold mileage difference in asking prices", not "buyers ignore mileage".
- **Location is not a pricing dimension** on this platform, at least across Java.
- **Chinese brands are specification-rich and price-poor**: Wuling Cortez −31%, BYD Atto 1
  −26.7% against what their specifications predict.

## Feature glossary

Names used in `model.py` and in the model report:

| Name | Meaning |
| --- | --- |
| `brand` | manufacturer, normalised (`BMW`, `MINI`, `Kia` …) |
| `model_g` | model name; models with fewer than 20 listings are grouped as `other` |
| `age` | 2026 minus the vehicle's model year |
| `mileage_km` | odometer reading; null when the listing omits it |
| `km_per_year` | mileage divided by age |
| `engine_cc` | engine displacement in cc (the source stores litres) |
| `engine_hp` | engine power in horsepower, when stated |
| `transmission` | `Automatic` or `Manual` |
| `body_type` | SUV, MPV, Hatchback, Sedan, Coupe, Pickup … |
| `fuel_type` | Petrol, Diesel, Hybrid … |
| `province` | province of the listing |
| `profile_type` | `Dealer` or `Private` |
| `has_mileage`, `has_engine` | 1 when that field was reported |
| `flag_pajak_hidup` | description mentions road tax being paid |
| `flag_tangan_pertama` | description claims a first owner |
| `flag_service_record` | description mentions service history |
| `flag_nego` | description says the price is negotiable |
| `weight` | market share ÷ sample share, for market-level statistics |

## Business implications

These are interpretations, not measured outcomes.

- **Buyers** — the asking price is the model and the year. Mileage and gearbox are where
  the negotiation actually lives, because the asking price barely reflects them.
- **Sellers** — a service history and a low odometer do not show up in the asking price on
  this platform; they may still matter at the table, but not on paper.
- **Analysts** — most of the apparent signal in this market is composition. Holding model and
  year constant is the cheapest way to see through it, and a median-by-model-year baseline is
  the number any model must beat.

## Limitations

- **One platform, one snapshot.** These are 10,484 cars from Mobil123, scraped on 10 November
  2026, and 97.7% of them are dealer listings. OLX and the certified retailers price
  differently, and private sellers are almost unobserved.
- **Asking prices only.** Negotiated transaction prices are unobservable, and mileage may be
  discounted far more during negotiation than in the asking price.
- **Nominal, not inflation-adjusted.** With ~4% inflation, real depreciation is roughly double
  the numbers above.
- **Odometer readings are plausibly rolled back**, which attenuates any real mileage effect.
  Why the effect is small cannot be settled with this data: asking prices may simply be
  anchored on the model and year; a buyer who cannot verify the odometer will not pay much
  for it; mileage may be discounted at the negotiating table rather than in the asking price;
  condition may offset usage; and annual distances here are modest (6,000–7,000 km).
- **Coverage.** Only four provinces have ≥150 listings, so the location result describes Java,
  not Indonesia. 918 sellers hold the 10,484 listings (one dealer has 386), so observations
  are not fully independent.
- **Fragile estimates.** Hyundai's retention, the automatic premium above age five, and
  specification-only premiums for brands with few listings.

---

**Antoni** — Data Analyst · [GitHub](https://github.com/antoni-ho) ·
[LinkedIn](https://www.linkedin.com/in/antoni-h20/)
