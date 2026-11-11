"""
Verification pass.

Every headline number in eda.py is recomputed here with a *different*
estimator. The point is to catch the case where a chart is right about the
pipeline and wrong about the world: if two unrelated methods agree, the
finding is not an artefact of one modelling choice.

  claim 1  brand retention ordering and magnitude
           fitted  : log(median price) ~ age, slope exponentiated
           check   : two-point estimator, (p(age 8) / p(age 3)) ** (1/5)
  claim 2  automatic premium inside model-year cells
           fitted  : median of per-cell price ratios
           check   : weighted mean of log ratios, plus bootstrap CI
  claim 3  odometer has almost no price effect
           fitted  : tercile medians of price relative to model-year
           check   : within-cell fixed-effect slope on all cells, on cells
                     with wide odometer spread, and bootstrap CI
           plus    : concrete model-years shown in rupiah

Usage:
    python verify.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data/clean.csv.gz")
REPORT = Path("data/verify_report.txt")

_lines: list[str] = []


def out(msg: str = "") -> None:
    print(msg)
    _lines.append(str(msg))


def spearman(a, b) -> float:
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def bootstrap_median(values, n_boot=2000, seed=11):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    draws = rng.choice(arr, size=(n_boot, len(arr)), replace=True)
    meds = np.median(draws, axis=1)
    return float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


# ---------------------------------------------------------------------------

def check_brand_retention(df: pd.DataFrame) -> None:
    out("=" * 70)
    out("CLAIM 1  brand retention")
    out("=" * 70)

    fitted, twopoint = {}, {}
    for (brand, model), g in df.groupby(["brand", "model"], dropna=True):
        g = g[(g["age"] >= 0) & (g["age"] <= 10)]
        med = g.groupby("age")["price"].median()
        cnt = g.groupby("age")["price"].size()
        keep = cnt[cnt >= 3].index
        if len(keep) >= 5:
            x = np.asarray(keep, float)
            y = np.log(med.loc[keep].astype(float).to_numpy())
            fitted.setdefault(brand, []).append(float(np.exp(np.polyfit(x, y, 1)[0])))
        if 3 in keep and 8 in keep:
            ratio = med.loc[8] / med.loc[3]
            twopoint.setdefault(brand, []).append(float(ratio ** (1 / 5)))

    brands = [b for b in fitted if b in twopoint and len(fitted[b]) >= 3]
    f = [np.median(fitted[b]) for b in brands]
    t = [np.median(twopoint[b]) for b in brands]

    out(f"brands compared        : {len(brands)}")
    out(f"rank correlation       : {spearman(f, t):.3f}")
    out(f"median absolute gap    : {100*np.median(np.abs(np.array(f)-np.array(t))):.2f} pp")
    out(f"{'brand':<16}{'fitted':>9}{'2-point':>10}{'gap':>8}")
    for b, fv, tv in sorted(zip(brands, f, t), key=lambda r: -r[1]):
        out(f"{b:<16}{100*fv:>8.1f}%{100*tv:>9.1f}%{100*(fv-tv):>+7.1f}pp")
    out()


def check_transmission(df: pd.DataFrame) -> None:
    out("=" * 70)
    out("CLAIM 2  automatic premium, like-for-like")
    out("=" * 70)

    ratios, weights = [], []
    raw_premiums = []
    for (brand, model, year), g in df.groupby(["brand", "model", "year"], dropna=True):
        at = g[g["transmission"] == "Automatic"]["price"]
        mt = g[g["transmission"] == "Manual"]["price"]
        if len(at) >= 5 and len(mt) >= 5:
            ratio = at.median() / mt.median()
            ratios.append(ratio)
            weights.append(min(len(at), len(mt)))
            raw_premiums.append(ratio - 1)

    ratios = np.array(ratios)
    weights = np.array(weights, float)
    median_premium = float(np.median(ratios) - 1)
    weighted_log = float(np.exp(np.average(np.log(ratios), weights=weights)) - 1)
    lo, hi = bootstrap_median(raw_premiums)

    at_all = df[df["transmission"] == "Automatic"]["price"].median()
    mt_all = df[df["transmission"] == "Manual"]["price"].median()

    out(f"cells                       : {len(ratios)}")
    out(f"median of ratios            : {100*median_premium:+.1f}%")
    out(f"weighted mean of log ratios : {100*weighted_log:+.1f}%")
    out(f"bootstrap 95% CI (median)   : [{100*lo:+.1f}%, {100*hi:+.1f}%]")
    out(f"raw, all listings pooled    : {100*(at_all/mt_all-1):+.1f}%")
    out(f"agreement between estimators: {abs(median_premium-weighted_log)*100:.1f} pp difference")
    out()


def within_slope(d: pd.DataFrame) -> tuple[float, int]:
    """Fixed-effect slope of log(price) on log(1+km), demeaned within cells."""
    cell = ["brand", "model", "year"]
    d = d.copy()
    d["lp"] = np.log(d["price"])
    d["lk"] = np.log1p(d["mileage_km"])
    g = d.groupby(cell)
    d["lp_r"] = d["lp"] - g["lp"].transform("mean")
    d["lk_r"] = d["lk"] - g["lk"].transform("mean")
    d = d[d["lk_r"].abs() > 1e-9]
    if len(d) < 10:
        return float("nan"), 0
    slope = float(np.polyfit(d["lk_r"], d["lp_r"], 1)[0])
    return slope, len(d)


def check_mileage(df: pd.DataFrame) -> None:
    out("=" * 70)
    out("CLAIM 3  the odometer is barely priced")
    out("=" * 70)

    d = df.dropna(subset=["mileage_km", "age"]).copy()
    d = d.groupby(["brand", "model", "year"]).filter(lambda g: len(g) >= 2)

    # (a) all cells with at least 2 listings
    slope_all, n_all = within_slope(d)
    out(f"(a) within-cell slope, all cells with >=2 rows")
    out(f"    slope {slope_all:+.4f} on log(1+km), n={n_all:,}")
    out(f"    20k -> 100k km costs {100*(np.exp(slope_all*(np.log1p(100000)-np.log1p(20000)))-1):+.1f}%")

    # (b) only cells where the odometer actually varies a lot
    wide = []
    for key, g in d.groupby(["brand", "model", "year"]):
        if len(g) >= 6:
            spread = g["mileage_km"].quantile(0.9) - g["mileage_km"].quantile(0.1)
            if spread >= 30_000:
                wide.append(key)
    d_wide = d.set_index(["brand", "model", "year"]).loc[wide].reset_index()
    slope_wide, n_wide = within_slope(d_wide)
    out(f"(b) only cells with a >=30,000 km spread")
    out(f"    {len(wide):,} cells, n={n_wide:,}, slope {slope_wide:+.4f}")
    out(f"    20k -> 100k km costs {100*(np.exp(slope_wide*(np.log1p(100000)-np.log1p(20000)))-1):+.1f}%")

    # (c) bootstrap CI on the wide-spread slope
    if len(wide) > 3:
        rng = np.random.default_rng(5)
        slopes = []
        for _ in range(300):
            pick = rng.choice(len(wide), size=len(wide), replace=True)
            keys = [wide[i] for i in pick]
            sub = d.set_index(["brand", "model", "year"]).loc[keys].reset_index()
            s, _n = within_slope(sub)
            if not np.isnan(s):
                slopes.append(s)
        if slopes:
            out(f"    bootstrap 95% CI for slope: "
                f"[{np.quantile(slopes,0.025):+.4f}, {np.quantile(slopes,0.975):+.4f}]")

    # (d) concrete model-years, in rupiah
    out("")
    out("(d) concrete cases: same model, same year, different odometer")
    out(f"    {'model-year':<34}{'low km':>22}{'high km':>22}{'gap':>8}")
    shown = 0
    for (brand, model, year), g in d.groupby(["brand", "model", "year"]):
        if len(g) < 9:
            continue
        g = g.sort_values("mileage_km")
        k = len(g) // 3
        low, high = g.iloc[:k], g.iloc[-k:]
        if high["mileage_km"].median() < 2 * low["mileage_km"].median():
            continue
        gap = high["price"].median() / low["price"].median() - 1
        out(f"    {f'{brand} {model} {year}':<34}"
            f"{low['mileage_km'].median():>9,.0f} km / {low['price'].median()/1e6:>6,.0f}M"
            f"{high['mileage_km'].median():>9,.0f} km / {high['price'].median()/1e6:>6,.0f}M"
            f"{100*gap:>7.1f}%")
        shown += 1
        if shown >= 8:
            break
    out()

    # (e) does the odometer matter more in any segment?
    out("(e) slope by segment (cells with >=2 rows)")
    for label, mask in [
        ("luxury brands", d["is_luxury"] == 1),
        ("mass market", d["is_luxury"] == 0),
        ("manual cars", d["transmission"] == "Manual"),
        ("automatic cars", d["transmission"] == "Automatic"),
        ("age 0-3", d["age"] <= 3),
        ("age 4-9", (d["age"] >= 4) & (d["age"] <= 9)),
    ]:
        s, n = within_slope(d[mask])
        out(f"    {label:<16} slope {s:+.4f}  n={n:,}")
    out()


def main() -> None:
    df = pd.read_csv(DATA)
    out("VERIFICATION REPORT")
    out(f"dataset: {len(df):,} rows")
    out()
    check_brand_retention(df)
    check_transmission(df)
    check_mileage(df)
    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    out(f"report written to {REPORT}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
