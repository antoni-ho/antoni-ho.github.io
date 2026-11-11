# Methodology notes

Two decisions in this project are unusual enough to need a written defence:
why only 12,000 listings were collected out of 59,037 available, and what the
`year` field actually means. Both belong in the README too.

---

## 1. Why 12,000 listings, not 59,037

The sitemap exposes 59,037 active listing URLs. The crawl stopped at 12,000.
The short reason: **sample size was not the binding constraint.** Bias is, and
bias does not shrink when you add rows.

### 1.1 More data does not fix bias

All 59,037 URLs and all 12,000 rows come from the same platform, and two
limitations apply equally to both:

| Limitation | Value in this dataset | Does more data fix it? |
|---|---|---|
| Listings are dominated by dealers | **97.7% dealer**, 2.3% private | No |
| Price is an asking price, not a transaction | 100% | No |
| One platform, not the whole market | 1 of ~5 marketplaces | No |

A five-times larger crawl shrinks random error. It does nothing to systematic
error, and systematic error is far larger here.

### 1.2 Random noise is already small

Within a single (brand, model, year) cell, the dispersion of log price is only
**σ = 0.121** - comparable cars differ by about ±13%. With dispersion that
tight, a median is already precise at small n:

| Cell size (n) | 95% confidence interval of the median |
|---|---|
| 3 (dataset median) | ±18.7% |
| 10 | ±9.8% |
| 15 | ±8.0% |
| 75 | ±3.6% |

Going from n=3 to n=75 costs 25x the crawl and buys a 5x narrower interval on a
quantity we do not even report at that grain (see 1.4).

### 1.3 The cost is measured, not estimated

Observed crawl throughput was **0.43 URLs/sec**, not the 1.4/worker/sec the plan
assumed - the site rate-limits.

| | 12,000 (used) | 59,037 (all) |
|---|---|---|
| Wall-clock time | **7.7 hours** | ~38 hours |
| Transfer (gzip, 214 KB/page) | **3.1 GB** | 12.6 GB |
| Disk for parsed JSON | **21 MB** | ~95 MB |

### 1.4 The unit of analysis is a model *series*, not a single cell

The data fragments into 1,996 (brand, model, year) cells with a median of only
3 observations. But depreciation is not estimated cell by cell - it is estimated
per **model series**: one model observed across several years.

| Threshold | Series | Share of rows |
|---|---|---|
| ≥ 100 obs & ≥ 6 years | 24 | 30.6% |
| ≥ 50 obs & ≥ 5 years | **61** | **53.3%** |
| ≥ 30 obs & ≥ 5 years | 91 | 62.9% |
| ≥ 20 obs & ≥ 4 years | 118 | 69.3% |

Those 61 series support retention curves, the transmission analysis, city
effects and the hedonic model. Examples: Honda HR-V (261 obs / 12 years),
Toyota Avanza (215 / 15), Mitsubishi Xpander (232 / 10).

Crucially, thin cells are a consequence of market fragmentation (531 model
series across four decades), not of a small n. Doubling the sample to 24,000
would move the median cell from 3 to about 6 - still useless for per-cell
analysis. The fix is choosing the right unit of analysis, not collecting more.

### 1.5 Sampling design (reproducible)

- **Population**: 59,037 unique URLs from 6 `listing_details_*.xml.gz` shards
- **Frame**: every URL, grouped by brand using the listing slug
- **Stratification**: per-brand quota (Toyota 2,500, Honda 1,500, Mercedes 1,000,
  ..., small brands 50) so volume brands cannot drown out the tail
- **Randomisation**: `random.Random(42)`, 14,984 URLs prepared
- **Stopping rule**: exactly 12,000 valid rows; the remainder stays queued
- **Failures**: 93 URLs (404 or parse failure) were not retried

---

## 2. What `year` means

`year` is the **vehicle's model/production year**, taken from the JSON-LD
`vehicleModelDate` field of a `@type: Car` object and from the `year` entry of
the `ga:cad:details:*` meta tags. It describes the car.

It is **not** the year the dealer published the listing, and it has nothing to
do with the dealer. Every row in this dataset was scraped in November 2026, so a
"2020" row is a car *manufactured* in 2020 that is on sale now - six years old -
not a listing posted in 2020.

Two consequences worth keeping in mind:

- `age = 2026 - year` is a genuine vehicle age at the snapshot date.
- A thin 2020 cohort (only 408 listings, against 1,282 for 2022) reflects how few
  2020-built cars exist to be resold, consistent with the collapse in Indonesian
  car sales that year. It is a production-cohort effect, not a listing artefact.

---

## 3. Why the odometer barely moves the price

Raw, the odometer looks like the strongest price driver in the dataset: the
highest-mileage quintile lists **74% below** the lowest (8,878 listings that
report an odometer). Inside a model-year, the same comparison collapses to
**−4.1%** (median across 504 model-years; bootstrap 95% CI −4.8% to −3.2%). A
five-fold odometer difference (20,000 to 100,000 km) is worth roughly **−6%** in
the fixed-effect regression.

That gap between −74% and −4% is the whole point: the raw figure is measuring
*which cars* have high mileage, not what mileage does to a price.

### A correction worth recording

The first version of this analysis split the pooled data into mileage terciles
within age bands and reported −1%. That method is wrong for this question. It
mixes within-cell and between-cell variation: a model-year whose cars all have
high mileage still sits at its own median, which drags the pooled tercile back
towards zero. The correct contrast compares the high-mileage and low-mileage
thirds *of the same model-year*, which is what `eda.py` now plots and what
`verify.py` reproduces independently (regression slope −0.040, 95% CI
−0.050 to −0.031).

### Why the effect is small — candidate mechanisms

We cannot separate these with the data at hand, and the write-up should say so
rather than pick a favourite:

1. **Anchoring on model-year, not usage.** 97.7% of listings are dealers, who
   price from comparable asking prices and reference values. The odometer is a
   merchandising detail, not the line item that sets the number.
2. **Odometer credibility.** Rolled-back odometers are common in this market. A
   buyer who cannot verify the reading will not pay much for it, and
   measurement error in the variable attenuates any real relationship.
3. **Asking price is not the transaction price.** Negotiation may discount
   mileage far more than the sticker does, and we observe only the sticker.
4. **Condition offsets usage.** A high-mileage car with a full service history
   can be worth more than a neglected low-mileage one. If the two effects
   partly cancel, the average mileage coefficient shrinks.
5. **The distances are small.** Median usage in this dataset is roughly
   6,000–7,000 km per year, so a three-fold odometer difference is 20,000 vs
   60,000 km — modest in wear terms.
6. **Survivorship.** Cars still being advertised are the presentable ones; the
   high-mileage wrecks never reach a listing.

Only mechanism 5 is directly checkable from this data. Mechanisms 2, 3 and 4
are consistent with the result but untestable here, so the defensible claim is
**"the odometer is weakly priced in asking prices"**, not "buyers ignore
mileage".

## 4. How the headline numbers were checked

`verify.py` recomputes every headline figure with an independent estimator, so
a result cannot survive only because of one modelling choice:

| Claim | Primary estimator | Independent check | Agreement |
|---|---|---|---|
| Brand retention | log-price slope per series | two-point ratio p(8)/p(3) | rank corr 0.87, median gap 1.1 pp |
| Automatic premium | median of per-cell price ratios | weighted mean of log ratios | +5.8% vs +7.2% |
| Odometer effect | model-year contrast | fixed-effect regression slope | −4.1% vs −6.3% per 5× km |

Two estimates are flagged as fragile rather than reported as solid: **Hyundai's**
retention (87.5% fitted, 92.9% two-point — a 5.4 pp gap, driven by only five
model series) and the transmission premium above age 5, which rests on a single
model-year cell.

## 5. Limitations that must appear in the write-up

1. One platform, one snapshot (10 November 2026).
2. Asking prices only; negotiated transaction prices are unobservable.
3. 97.7% dealer listings, so private-seller pricing behaviour is barely represented.
4. 12.3% of rows are duplicates or re-listings; they are removed before analysis.
5. 918 sellers for 12,000 listings (one dealer holds 386), so observations are
   not fully independent.
6. Old cars are rare in the market (67 listings from 2010), so the far end of
   any depreciation curve is the least certain part of it.
