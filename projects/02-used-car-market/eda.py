"""
Stage 2b - exploratory data analysis.

Produces the figures that carry the project's argument:

  01_brand_retention.png       annual value retention by brand
  02_retention_curves.png      price index versus age, small multiples
  03_transmission_premium.png  the automatic premium: raw versus like-for-like
  04_mileage_illusion.png      the odometer: raw versus like-for-like
  05_market_structure.png      what the marketplace actually contains

Charts 3 and 4 make the same point, and it is the thesis of the project:
almost every large price difference visible in this market disappears once two
cars of the same model and year are compared.

Reads : data/clean.csv.gz
Writes: images/*.png, data/eda_report.txt

Usage:
    python eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path("data/clean.csv.gz")
IMAGES = Path("images")
REPORT = Path("data/eda_report.txt")

MIN_AGE_POINTS = 5       # a series needs this many ages with data to be fitted
MIN_CELL_N = 3           # minimum listings behind a single fitted age point
MIN_CURVE_N = 8          # minimum listings behind a plotted curve point
MIN_TRANS_N = 5          # minimum per transmission inside a cell
MIN_GROUP_N = 20
AGE_FIT_MAX = 10         # fit the decay slope on ages 0..10 only

INK = "#1f2d3d"
ACCENT = "#c2571a"
COOL = "#2b6f8f"
MUTED = "#8a94a0"
GRID = "#eceff2"
BAND_COLORS = {"0–2": "#2b6f8f", "3–5": "#c2571a", "6–9": "#5f8a5a"}
PALETTE = ["#2b6f8f", "#c2571a", "#5f8a5a", "#8c6bb1", "#c9a227",
           "#b04a5a", "#4a7fa5", "#7a6a5a"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.axisbelow": True,
    "axes.edgecolor": "#d5dae0",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.dpi": 200,
    "savefig.dpi": 200,
})

_lines: list[str] = []


def out(msg: str = "") -> None:
    print(msg)
    _lines.append(str(msg))


def add_title(fig, text: str, gap: float | None = None) -> None:
    """Centre a title just above the axes.

    fig.suptitle() reserves its own band of empty space and tight_layout's
    rect only partly overrides it, which leaves a visible gap. Placing the
    text relative to the axes bounding box, after the layout is final, keeps
    the title close; the saved canvas grows upwards to fit it.
    """
    fig.tight_layout()
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    # get_tightbbox includes each panel title, so the figure title clears them.
    boxes = [ax.get_tightbbox(fig.canvas.get_renderer()).transformed(inv)
             for ax in fig.axes if ax.get_visible()]
    left = min(b.x0 for b in boxes)
    right = max(b.x1 for b in boxes)
    top = max(b.y1 for b in boxes)
    if gap is None:
        # Panels that carry their own title need more air under the figure
        # title than a single bare plot does.
        gap = 0.075 if any(ax.get_title() for ax in fig.axes) else 0.035
    fig.text((left + right) / 2, top + gap, text, ha="center", va="bottom",
             fontsize=15, fontweight="bold")


def save(fig, name: str) -> None:
    IMAGES.mkdir(exist_ok=True)
    path = IMAGES / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.16, facecolor="white")
    plt.close(fig)
    out(f"   saved {path}")


# ---------------------------------------------------------------------------
# 1. annual retention by brand
# ---------------------------------------------------------------------------

def series_retention(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each model series, fit log(median price) ~ age over ages 0..AGE_FIT_MAX.
    The exponentiated slope is the share of value retained per year of age.
    """
    rows = []
    for (brand, model), g in df.groupby(["brand", "model"], dropna=True):
        g = g[(g["age"] >= 0) & (g["age"] <= AGE_FIT_MAX)]
        if g.empty:
            continue
        med = g.groupby("age")["price"].median()
        cnt = g.groupby("age")["price"].size()
        keep = cnt[cnt >= MIN_CELL_N].index
        if len(keep) < MIN_AGE_POINTS:
            continue
        x = np.asarray(keep, dtype=float)
        y = np.log(med.loc[keep].astype(float).to_numpy())
        slope = float(np.polyfit(x, y, 1)[0])
        rows.append({
            "brand": brand, "model": model, "n": len(g),
            "ages": len(keep), "slope": slope,
            "retention": float(np.exp(slope)),
        })
    return pd.DataFrame(rows)


def chart_brand_retention(series: pd.DataFrame) -> pd.DataFrame:
    summary = series.groupby("brand").agg(
        median=("retention", "median"),
        count=("retention", "size"),
        q1=("retention", lambda s: s.quantile(0.25)),
        q3=("retention", lambda s: s.quantile(0.75)),
    )
    summary = summary[summary["count"] >= 3].sort_values("median", ascending=False)

    # Plot the yearly loss as a positive number: "4.4% a year" reads better than
    # "−4.4%", and it removes the question of which way the minus points.
    loss = (1 - summary["median"]) * 100
    loss_low = (1 - summary["q3"]) * 100
    loss_high = (1 - summary["q1"]) * 100

    fig, ax = plt.subplots(figsize=(10, 4.2))
    y = np.arange(len(summary))
    colors = [ACCENT if i < 3 else COOL for i in range(len(summary))]
    ax.barh(
        y, loss,
        xerr=[loss - loss_low, loss_high - loss],
        color=colors, height=0.62,
        error_kw={"ecolor": "#c3cad1", "lw": 1.1, "capsize": 0},
    )
    for yi, value in enumerate(loss):
        ax.text(value + 0.35, yi, f"{value:.1f}% a year", va="center",
                ha="left", fontsize=10, color=INK, fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{b}   n={int(c)}" for b, c in zip(summary.index, summary["count"])],
        fontsize=10,
    )
    ax.invert_yaxis()
    ax.set_xticks(range(0, 19, 3))
    ax.set_xlim(0, loss_high.max() + 3.5)
    add_title(fig, "Value lost per year, by brand (%)")
    save(fig, "01_brand_retention.png")
    return summary


# ---------------------------------------------------------------------------
# 2. retention curves, small multiples
# ---------------------------------------------------------------------------

def chart_retention_curves(df: pd.DataFrame, series: pd.DataFrame) -> None:
    top = series.sort_values("n", ascending=False).head(8)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.8), sharex=True, sharey=True)

    for ax, (_, row) in zip(axes.flat, top.iterrows()):
        g = df[(df["brand"] == row["brand"]) & (df["model"] == row["model"])]
        med = g.groupby("age")["price"].median()
        cnt = g.groupby("age")["price"].size()
        keep = cnt[cnt >= MIN_CURVE_N].index
        # age 0 is a delivery-mileage car advertised as used; it is priced
        # against the new-car market and would distort the used-car curve.
        keep = keep[(keep >= 1) & (keep <= 15)]
        if len(keep) < 3:
            ax.set_visible(False)
            continue
        base_age = int(keep.min())
        index = (med.loc[keep] / med.loc[base_age] * 100).to_numpy()
        ax.plot(np.asarray(keep), index, marker="o", ms=3.5, lw=1.7, color=COOL)
        ax.fill_between(np.asarray(keep), 0, index, color=COOL, alpha=0.07)
        ax.axhline(50, color=MUTED, lw=1, ls=":")
        ax.set_title(f"{row['brand']} {row['model']}", loc="left", fontsize=11,
                     color=INK, fontweight="normal")
        ax.text(0.98, 0.92, f"n={int(row['n'])}", transform=ax.transAxes,
                ha="right", fontsize=8.5, color=MUTED)

    for ax in axes[-1]:
        ax.set_xlabel("age (years)")
    for ax in axes[:, 0]:
        ax.set_ylabel("price index")
    axes[0, 0].set_ylim(0, 145)
    axes[0, 0].set_xticks(range(1, 16, 2))

    add_title(fig, "Price index by model and vehicle age")
    save(fig, "02_retention_curves.png")


# ---------------------------------------------------------------------------
# 3. transmission premium
# ---------------------------------------------------------------------------

def age_band(age: int) -> str:
    if age <= 2:
        return "0–2"
    if age <= 5:
        return "3–5"
    return "6–9"


def transmission_cells(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (brand, model, year), g in df.groupby(["brand", "model", "year"], dropna=True):
        at = g[g["transmission"] == "Automatic"]["price"]
        mt = g[g["transmission"] == "Manual"]["price"]
        if len(at) >= MIN_TRANS_N and len(mt) >= MIN_TRANS_N:
            rows.append({
                "brand": brand, "model": model, "year": int(year),
                "age": int(g["age"].iloc[0]),
                "n_at": len(at), "n_mt": len(mt),
                "premium": at.median() / mt.median() - 1,
            })
    cells = pd.DataFrame(rows)
    cells["band"] = cells["age"].map(age_band)
    return cells


def chart_transmission(df: pd.DataFrame, cells: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [0.7, 1.5]}
    )

    # A - raw comparison
    at = df[df["transmission"] == "Automatic"]["price"].median() / 1e6
    mt = df[df["transmission"] == "Manual"]["price"].median() / 1e6
    ax1.bar(["Manual", "Automatic"], [mt, at], color=[MUTED, ACCENT], width=0.55)
    for i, value in enumerate([mt, at]):
        ax1.text(i, value * 1.02, f"{value:,.0f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")
    ax1.set_ylabel("median asking price (Rp million)")
    ax1.set_title("A. All listings pooled", loc="left",
                  pad=10, fontsize=11, color=INK, fontweight="normal")
    ax1.set_ylim(0, at * 1.16)

    # B - like-for-like strip plot, one row per age band. One colour on purpose:
    # the row already says the age band, so a second colour scale only confuses.
    ypos = {"0–2": 2, "3–5": 1, "6–9": 0}
    rng = np.random.default_rng(7)
    for band in ypos:
        part = cells[cells["band"] == band]
        if part.empty:
            continue
        y = ypos[band] + rng.uniform(-0.17, 0.17, len(part))
        ax2.scatter(part["premium"] * 100, y, s=66, color=COOL, alpha=0.9,
                    edgecolor="white", linewidth=0.8, zorder=3)

    median = cells["premium"].median() * 100
    rng_ci = np.random.default_rng(11)
    boots = [np.median(rng_ci.choice((cells["premium"] * 100).to_numpy(),
                                     len(cells), replace=True))
             for _ in range(1500)]
    ci_lo, ci_hi = (float(v) for v in np.quantile(boots, [0.025, 0.975]))
    ax2.axvline(0, color=INK, lw=1)
    ax2.axvline(median, color=ACCENT, lw=2, ls="-")
    ax2.axvspan(ci_lo, ci_hi, color=INK, alpha=0.07, lw=0, zorder=0)
    ax2.text(median + 0.6, 2.66,
             f"typical +{median:.1f}%   (likely between +{ci_lo:.1f}% and +{ci_hi:.1f}%)",
             fontsize=9.5, color=INK, va="center", ha="left")

    top_cell = cells.nlargest(1, "premium").iloc[0]
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["6–9 years", "3–5 years", "0–2 years"], fontsize=9.5)
    ax2.set_ylabel("vehicle age")
    ax2.set_ylim(-0.55, 2.85)
    ax2.set_xlabel("automatic price minus manual price (%), same model and year")
    ax2.set_title("B. Comparing the same model and year", loc="left",
                  pad=10, fontsize=11, color=INK, fontweight="normal")

    add_title(fig, "Automatic versus manual")
    save(fig, "03_transmission_premium.png")


# ---------------------------------------------------------------------------
# 4. the odometer
# ---------------------------------------------------------------------------

def chart_mileage(df: pd.DataFrame) -> dict:
    # Panel A uses every listing that reports an odometer; panel B needs cells
    # with enough listings to compare a high-mileage third against a low one.
    all_km = df.dropna(subset=["mileage_km"]).copy()
    d = all_km.dropna(subset=["age"]).copy()
    cell = ["brand", "model", "year"]
    d = d.groupby(cell).filter(lambda g: len(g) >= 5).copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.2))

    # A - raw, all listings with an odometer reading
    quint = pd.qcut(all_km["mileage_km"], 5, duplicates="drop")
    raw = all_km.groupby(quint, observed=True)["price"].median() / 1e6
    labels = [f"{int(iv.left/1000)}–{int(iv.right/1000)}k" for iv in raw.index]
    ax1.bar(np.arange(len(raw)), raw.values, color=ACCENT, width=0.64)
    for i, value in enumerate(raw.values):
        ax1.text(i, value * 1.02, f"{value:,.0f}", ha="center", va="bottom",
                 fontsize=9.5)
    ax1.set_xticks(np.arange(len(raw)))
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylim(0, raw.max() * 1.16)
    ax1.set_ylabel("median asking price (Rp million)")
    ax1.set_xlabel(f"kilometres on the clock  ·  n={len(all_km):,} listings")
    ax1.set_title(f"A. All listings together — a {100*(1-raw.iloc[-1]/raw.iloc[0]):.0f}% gap",
                  loc="left", pad=10, fontsize=11, color=INK, fontweight="normal")

    # B - proper within-cell contrast: the highest-mileage third against the
    # lowest-mileage third *of the same model-year*. Splitting the pooled data
    # into terciles would mix between-cell and within-cell variation and
    # understate the effect: a cell whose cars all have high mileage still sits
    # at its own median, which flattens the pooled picture.
    rows = []
    for (brand, model, year), g in d.groupby(cell):
        if len(g) < 6:
            continue
        g = g.sort_values("mileage_km")
        k = max(2, len(g) // 3)
        low, high = g.iloc[:k], g.iloc[-k:]
        if high["mileage_km"].median() <= low["mileage_km"].median():
            continue
        rows.append({
            "brand": brand, "model": model, "year": int(year),
            "km_low": float(low["mileage_km"].median()),
            "km_high": float(high["mileage_km"].median()),
            "gap": 100 * (high["price"].median() / low["price"].median() - 1),
        })
    gaps = pd.DataFrame(rows)

    rng = np.random.default_rng(3)
    med = float(np.median(gaps["gap"]))
    boots = [np.median(rng.choice(gaps["gap"].to_numpy(), len(gaps), replace=True))
             for _ in range(1500)]
    ci_lo, ci_hi = (float(v) for v in np.quantile(boots, [0.025, 0.975]))

    ax2.scatter(gaps["gap"], rng.uniform(-0.38, 0.38, len(gaps)), s=18,
                color=COOL, alpha=0.45, linewidth=0)
    ax2.axvspan(ci_lo, ci_hi, color=ACCENT, alpha=0.13, lw=0, zorder=0)
    ax2.axvline(0, color=INK, lw=1)
    ax2.axvline(med, color=ACCENT, lw=2, zorder=4)
    ax2.set_yticks([])
    ax2.set_ylim(-0.62, 0.78)
    ax2.set_xlabel("high-mileage price minus low-mileage price (%)\nfor the same model and year")
    ax2.set_title("B. Comparing the same model and year", loc="left",
                  pad=10, fontsize=11, color=INK, fontweight="normal")
    ax2.text(med + 1.2, 0.60, f"typical {abs(med):.1f}% cheaper", ha="left", va="center",
             fontsize=11, color=ACCENT, fontweight="bold")

    add_title(fig, "Price by mileage")
    save(fig, "04_mileage_illusion.png")
    return {"gaps": gaps, "raw": raw}


# ---------------------------------------------------------------------------
# 5. market structure
# ---------------------------------------------------------------------------

def chart_structure(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9),
                             gridspec_kw={"width_ratios": [1.6, 1, 1]})

    years = df["year"].value_counts().sort_index()
    years = years[years.index >= 2008]
    colors = [ACCENT if y == 2020 else COOL for y in years.index]
    axes[0].bar(years.index, years.values, color=colors, width=0.78)
    axes[0].set_title("Listings by model year", loc="left", pad=10,
                      fontsize=11, color=INK, fontweight="normal")
    axes[0].set_xlabel("vehicle model year")
    axes[0].set_ylabel("listings")
    axes[0].set_xticks(list(range(2008, 2027, 2)))
    axes[0].tick_params(axis="x", labelsize=9)

    body = df["body_type"].value_counts().head(6)
    y = np.arange(len(body))
    axes[1].barh(y, 100 * body.values / len(df), color=PALETTE[2], height=0.64)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(body.index, fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("% of listings")
    axes[1].set_title("Body type", loc="left", pad=10,
                      fontsize=11, color=INK, fontweight="normal")

    origin = df["origin"].value_counts()
    axes[2].barh(np.arange(len(origin)), 100 * origin.values / len(df),
                 color=PALETTE[3], height=0.64)
    axes[2].set_yticks(np.arange(len(origin)))
    axes[2].set_yticklabels(origin.index, fontsize=9)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("% of listings")
    axes[2].set_title("Brand origin", loc="left", pad=10,
                      fontsize=11, color=INK, fontweight="normal")

    add_title(fig, "Marketplace composition")
    save(fig, "05_market_structure.png")


# ---------------------------------------------------------------------------

def main() -> None:
    df = pd.read_csv(DATA)
    out("EDA")
    out("=" * 66)
    out(f"rows loaded: {len(df):,}")
    out()

    # ---- brand retention ----
    series = series_retention(df)
    out(f"model series fitted: {len(series):,}")
    rising = int((series["retention"] > 1).sum())
    out(f"series whose price rises with age (noise): {rising} "
        f"({100*rising/max(len(series),1):.1f}%)")
    out()
    out("-- annual retention by brand --")
    summary = chart_brand_retention(series)
    for brand, row in summary.iterrows():
        out(f"   {brand:16} {100*row['median']:5.1f}%   "
            f"IQR {100*row['q1']:4.1f}-{100*row['q3']:4.1f}   series={int(row['count'])}")
    top, bottom = summary.index[0], summary.index[-1]
    out(f"   spread: {top} loses {100*(1-summary.loc[top,'median']):.1f}%/yr, "
        f"{bottom} loses {100*(1-summary.loc[bottom,'median']):.1f}%/yr")
    out()

    # ---- retention curves ----
    chart_retention_curves(df, series)
    out()

    # ---- transmission ----
    cells = transmission_cells(df)
    at_raw = df[df["transmission"] == "Automatic"]["price"].median()
    mt_raw = df[df["transmission"] == "Manual"]["price"].median()
    out(f"transmission cells (>=5 each side) : {len(cells):,}")
    out(f"   raw automatic premium            : {100*(at_raw/mt_raw-1):+.1f}%")
    out(f"   like-for-like premium (median)   : {100*cells['premium'].median():+.1f}%")
    for band, part in cells.groupby("band"):
        out(f"   age {band}: {len(part):>2} cells, median {100*part['premium'].median():+.1f}%")
    chart_transmission(df, cells)
    out()

    # ---- mileage ----
    mile = chart_mileage(df)
    raw_km = mile["raw"]
    gaps = mile["gaps"]
    out(f"mileage: raw gap lowest->highest quintile : "
        f"{100*(raw_km.iloc[-1]/raw_km.iloc[0]-1):+.1f}%")
    out(f"mileage: within model-year contrast        : "
        f"{gaps['gap'].median():+.1f}% median, {gaps['gap'].mean():+.1f}% mean "
        f"({len(gaps)} model-years)")
    out(f"   median odometer, low vs high third      : "
        f"{gaps['km_low'].median():,.0f} vs {gaps['km_high'].median():,.0f} km")
    out(f"   share of model-years with a negative gap: "
        f"{100*(gaps['gap'] < 0).mean():.1f}%")
    out()

    # ---- structure ----
    chart_structure(df)
    out(f"body types: {df['body_type'].value_counts().head(4).to_dict()}")
    out(f"origins   : {df['origin'].value_counts().to_dict()}")

    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    out()
    out(f"report written to {REPORT}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
