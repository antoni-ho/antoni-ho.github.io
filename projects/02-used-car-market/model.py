"""
Stage 3 - hedonic price model.

Target
    log(price). Prices are multiplicative in this market: a coefficient of 0.05
    means "5% more expensive", which is the quantity we actually want to talk
    about.

Features
    Vehicle age, odometer, engine, body, fuel, transmission, province, seller
    type and the free-text flags. Brand and model are included: the deployed
    question is "what is this particular car worth", not "what is a car worth".

Validation
    Two schemes, because they answer different questions:
      KFold(5)              - the listing is new, the model is known. This is
                              the deployment case.
      GroupKFold by model   - the model series is unseen. A stress test that
                              says how much the score leans on model identity.

Baseline
    Median price of the same (brand, model, year) in the training fold. If the
    learned models cannot beat this clearly, they are theatre.

Usage:
    python model.py

Output:
    data/model_report.txt, images/06_feature_importance.png,
    data/predictions.csv.gz (out-of-fold predictions, for residual analysis)
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

# Pin the numeric libraries to one thread. HistGradientBoosting spawns a joblib
# thread pool during binning, which some sandboxes refuse; single-threaded runs
# are also bit-for-bit reproducible.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

DATA = Path("data/clean.csv.gz")
REPORT = Path("data/model_report.txt")
PRED = Path("data/predictions.csv.gz")
IMAGES = Path("images")

INK, ACCENT, COOL, MUTED = "#1f2d3d", "#c2571a", "#2b6f8f", "#8a94a0"
SEED = 42

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
    "grid.color": "#eceff2",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.titlesize": 12.5,
    "axes.titleweight": "bold",
    "figure.dpi": 200,
    "savefig.dpi": 200,
})

NUM = ["age", "mileage_km", "km_per_year", "engine_cc", "engine_hp", "seats"]
BIN = ["is_luxury", "has_mileage", "has_engine",
       "flag_pajak_hidup", "flag_tangan_pertama", "flag_service_record", "flag_nego"]
CAT = ["brand", "model_g", "transmission", "body_type", "fuel_type",
       "province", "profile_type"]
CAT_B = ["transmission", "body_type", "fuel_type", "province", "profile_type"]

MIN_MODEL_N = 20   # rarer models collapse into "other": HistGradientBoosting
                   # accepts at most 255 categories per feature

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


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    resid = y_true_log - y_pred_log
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true_log - y_true_log.mean()) ** 2))
    ape = np.abs(np.expm1(y_true_log) - np.expm1(y_pred_log)) / np.expm1(y_true_log)
    return {
        "rmse_log": float(np.sqrt(np.mean(resid ** 2))),
        "r2_log": 1 - ss_res / ss_tot,
        "mdape": float(np.median(ape) * 100),
        "within_10pct": float(np.mean(ape <= 0.10) * 100),
        "within_20pct": float(np.mean(ape <= 0.20) * 100),
    }


def cv_predict(X, y, fit_predict, groups=None, n_splits=5):
    splitter = (GroupKFold(n_splits=n_splits) if groups is not None
                else KFold(n_splits=n_splits, shuffle=True, random_state=SEED))
    pred = np.full(len(y), np.nan)
    for train_idx, test_idx in splitter.split(X, y, groups):
        pred[test_idx] = fit_predict(train_idx, test_idx)
    return pred


# ---------------------------------------------------------------------------
# pipelines
# ---------------------------------------------------------------------------

def onehot_pipeline(estimator) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM + BIN),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                              min_frequency=10), CAT),
    ])
    return Pipeline([("pre", pre), ("model", estimator)])


def make_ridge():
    return onehot_pipeline(Ridge(alpha=3.0))


def make_rf():
    return onehot_pipeline(RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, max_features=0.3,
        n_jobs=1, random_state=SEED))


def make_gbm():
    return HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=1.0,
        categorical_features="from_dtype", random_state=SEED)


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------

def baseline_predict(df: pd.DataFrame):
    def fit_predict(train_idx, test_idx):
        train = df.iloc[train_idx]
        by_cell = train.groupby(["brand", "model", "year"])["price"].median()
        by_model = train.groupby(["brand", "model"])["price"].median()
        by_brand = train.groupby("brand")["price"].median()
        fallback = train["price"].median()
        test = df.iloc[test_idx]
        preds = np.empty(len(test))
        for i, (_, row) in enumerate(test.iterrows()):
            value = by_cell.get((row["brand"], row["model"], row["year"]))
            if value is None or pd.isna(value):
                value = by_model.get((row["brand"], row["model"]))
            if value is None or pd.isna(value):
                value = by_brand.get(row["brand"], fallback)
            preds[i] = value
        return np.log(preds)
    return fit_predict


def model_fit_predict(factory, X):
    def fit_predict(train_idx, test_idx):
        model = factory()
        model.fit(X.iloc[train_idx], y[train_idx])
        return model.predict(X.iloc[test_idx])
    return fit_predict


# ---------------------------------------------------------------------------

def main() -> None:
    df = pd.read_csv(DATA)
    df = df[df["price"] > 0].reset_index(drop=True)

    df["has_mileage"] = df["mileage_km"].notna().astype(int)
    df["has_engine"] = df["engine_cc"].notna().astype(int)

    counts = df["model"].value_counts()
    keep = set(counts[counts >= MIN_MODEL_N].index)
    df["model_g"] = df["model"].where(df["model"].isin(keep), "other")

    global y, X
    y = np.log(df["price"].to_numpy(dtype=float))

    X = df[NUM + BIN + CAT].copy()
    X[CAT] = X[CAT].fillna("unknown").astype("category")

    out("HEDONIC PRICE MODEL")
    out("=" * 70)
    out(f"rows            : {len(df):,}")
    out(f"target          : log(price), median Rp {df['price'].median():,.0f}")
    out(f"features        : {len(NUM)} numeric + {len(BIN)} binary + {len(CAT)} categorical")
    out(f"brands / models : {df['brand'].nunique()} / {df['model'].nunique()}")
    out()

    # ---- baseline ----
    out("-- baseline: median of the same model-year --")
    base_pred = cv_predict(X, y, baseline_predict(df))
    base = metrics(y, base_pred)
    out(f"   RMSE(log) {base['rmse_log']:.4f}   R2 {base['r2_log']:.3f}   "
        f"MdAPE {base['mdape']:.1f}%   within 10%: {base['within_10pct']:.1f}%")
    out()

    # ---- models ----
    results = {"baseline": (base, base_pred)}
    for name, factory, frame in [
        ("ridge", make_ridge, X),
        ("random forest", make_rf, X),
        ("gradient boosting", make_gbm, X),
    ]:
        out(f"-- {name} --")
        pred = cv_predict(frame, y, model_fit_predict(factory, frame))
        m = metrics(y, pred)
        results[name] = (m, pred)
        out(f"   RMSE(log) {m['rmse_log']:.4f}   R2 {m['r2_log']:.3f}   "
            f"MdAPE {m['mdape']:.1f}%   within 10%: {m['within_10pct']:.1f}%")
        gain = 100 * (1 - m["rmse_log"] / base["rmse_log"])
        out(f"   improvement over baseline (RMSE): {gain:+.1f}%")
        out()

    # ---- stress test: unseen model series ----
    out("-- stress test: GroupKFold by model (an unseen model series) --")
    groups = df["model"].astype(str)
    pred_g = cv_predict(X, y, model_fit_predict(make_gbm, X), groups=groups)
    mg = metrics(y, pred_g)
    out(f"   gradient boosting  RMSE(log) {mg['rmse_log']:.4f}   R2 {mg['r2_log']:.3f}   "
        f"MdAPE {mg['mdape']:.1f}%")
    out(f"   cost of an unseen model vs KFold: "
        f"{100*(mg['rmse_log']/results['gradient boosting'][0]['rmse_log']-1):+.1f}% RMSE")
    out()

    # ---- table ----
    out("-- summary (baseline = median of the same model-year) --")
    out(f"{'model':<20}{'RMSE(log)':>11}{'vs base':>9}{'R2':>7}"
        f"{'MdAPE':>8}{'vs base':>10}{'±10%':>8}")
    for name, (m, _) in results.items():
        if name == "baseline":
            d_rmse, d_mdape = "", ""
        else:
            d_rmse = f"{100*(m['rmse_log']/base['rmse_log']-1):+.1f}%"
            d_mdape = f"{m['mdape']-base['mdape']:+.1f}pp"
        out(f"{name:<20}{m['rmse_log']:>11.4f}{d_rmse:>9}{m['r2_log']:>7.3f}"
            f"{m['mdape']:>7.1f}%{d_mdape:>10}{m['within_10pct']:>7.1f}%")
    out()

    # ---- feature importance on the winning model ----
    best_name = min(results, key=lambda k: results[k][0]["rmse_log"] if k != "baseline" else 9)
    out(f"-- permutation importance ({best_name}, on a 20% holdout) --")
    n = len(df)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)
    cut = int(0.8 * n)
    tr, te = idx[:cut], idx[cut:]
    best_factory = {"ridge": make_ridge, "random forest": make_rf,
                    "gradient boosting": make_gbm}[best_name]
    best = best_factory()
    best.fit(X.iloc[tr], y[tr])

    sample = te[: min(2500, len(te))]
    imp = permutation_importance(
        best, X.iloc[sample], y[sample], n_repeats=8,
        random_state=SEED, scoring="neg_root_mean_squared_error", n_jobs=1,
    )
    order = np.argsort(imp.importances_mean)[::-1]
    names = list(X.columns)
    for i in order[:14]:
        out(f"   {names[i]:<22}{imp.importances_mean[i]:+.4f} "
            f"(+/-{imp.importances_std[i]:.4f})")
    out()

    # ---- where the model is systematically wrong ----
    pred_best = results[best_name][1]
    resid = y - pred_best
    out("-- mean residual by brand (positive = listed above model expectation) --")
    d = df.assign(resid_pct=100 * (np.exp(-resid) - 1))
    brand = (d.groupby("brand")["resid_pct"]
             .agg(["median", "size"]).query("size >= 100")
             .sort_values("median", ascending=False))
    for b, row in brand.iterrows():
        out(f"   {b:<18}{row['median']:+6.1f}%   n={int(row['size']):,}")
    out()

    # ---- model B: specifications only ----
    out("-- model B: specifications only (brand and model withheld) --")
    Xb = df[NUM + BIN + CAT_B].copy()
    Xb[CAT_B] = Xb[CAT_B].fillna("unknown").astype("category")
    pred_b = cv_predict(Xb, y, model_fit_predict(make_gbm, Xb))
    mb = metrics(y, pred_b)
    out(f"   RMSE(log) {mb['rmse_log']:.4f}   R2 {mb['r2_log']:.3f}   "
        f"MdAPE {mb['mdape']:.1f}%   within 10%: {mb['within_10pct']:.1f}%")
    out("   residual below = how much the market pays for identity on top of specs")
    out()

    # positive = listed above what the specifications alone would justify
    df["spec_premium"] = 100 * (np.exp(y - pred_b) - 1)

    out("-- spec-adjusted premium by brand (n >= 100) --")
    brand_gap = (df.groupby("brand")["spec_premium"].agg(["median", "size"])
                 .query("size >= 100").sort_values("median", ascending=False))
    for name_, row in brand_gap.iterrows():
        out(f"   {name_:<18}{row['median']:+6.1f}%   n={int(row['size']):,}")
    out()

    model_gap = (df.groupby(["brand", "model"])["spec_premium"].agg(["median", "size"])
                 .query("size >= 25").sort_values("median"))
    out("-- priced furthest below specifications (n >= 25) --")
    for (brand_, model_), row in model_gap.head(10).iterrows():
        out(f"   {brand_} {model_:<22}{row['median']:+6.1f}%   n={int(row['size'])}")
    out("-- priced furthest above specifications --")
    for (brand_, model_), row in model_gap.tail(10).iloc[::-1].iterrows():
        out(f"   {brand_} {model_:<22}{row['median']:+6.1f}%   n={int(row['size'])}")
    out()

    # ---- geography, measured inside the full model ----
    df["loc_gap"] = 100 * (np.exp(resid) - 1)
    out("-- location effect inside the full model (same car, same year) --")
    loc = (df.groupby("province")["loc_gap"].agg(["median", "size"])
           .query("size >= 150").sort_values("median", ascending=False))
    for name_, row in loc.iterrows():
        out(f"   {name_:<20}{row['median']:+5.1f}%   n={int(row['size']):,}")
    out(f"   spread between provinces: {loc['median'].max()-loc['median'].min():.1f} pp")
    out()
    gap_span = loc["median"].max() - loc["median"].min()

    # ---- chart: location and brand premium ----
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.5, 5.0),
                                   gridspec_kw={"width_ratios": [1, 1]})
    axa.barh(np.arange(len(loc)), loc["median"], color=COOL, height=0.66)
    axa.set_yticks(np.arange(len(loc)))
    axa.set_yticklabels(loc.index, fontsize=9)
    axa.invert_yaxis()
    axa.axvline(0, color=INK, lw=1)
    # Fixed, honest scale: the point is that the differences are tiny, so the
    # axis must not zoom into them.
    axa.set_xlim(-2, 2)
    axa.set_xlabel("price difference for the same car sold elsewhere (%)")
    axa.set_title("A. Where the car is sold", loc="left", pad=10,
                  fontsize=11, color=INK, fontweight="normal")

    axb.barh(np.arange(len(brand_gap)), brand_gap["median"],
             color=[ACCENT if v > 0 else COOL for v in brand_gap["median"]],
             height=0.66)
    axb.set_yticks(np.arange(len(brand_gap)))
    axb.set_yticklabels(brand_gap.index, fontsize=9)
    axb.invert_yaxis()
    axb.axvline(0, color=INK, lw=1)
    axb.set_xlabel("asking price above or below what the specifications suggest (%)")
    axb.set_title("B. Which brand carries a premium", loc="left", pad=10,
                  fontsize=11, color=INK, fontweight="normal")

    add_title(fig, "Price by province and the brand premium")
    fig.savefig(IMAGES / "07_location_and_brand.png", bbox_inches="tight",
                pad_inches=0.16, facecolor="white")
    plt.close(fig)
    out(f"saved {IMAGES / '07_location_and_brand.png'}")

    # ---- chart: model-level bargains ----
    pick = pd.concat([model_gap.head(9), model_gap.tail(9)]).sort_values("median")
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ypos = np.arange(len(pick))
    ax.barh(ypos, pick["median"],
            color=[COOL if v < 0 else ACCENT for v in pick["median"]], height=0.66)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{b} {m}" for b, m in pick.index], fontsize=8.5)
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("price compared with what the car's specifications suggest (%)")
    add_title(fig, "Price versus specifications, by model")
    fig.savefig(IMAGES / "08_model_bargains.png", bbox_inches="tight",
                pad_inches=0.16, facecolor="white")
    plt.close(fig)
    out(f"saved {IMAGES / '08_model_bargains.png'}")
    out()

    # ---- artefacts ----
    # Readable names: `model_g` is the model name with rare models grouped.
    DISPLAY = {
        "brand": "brand", "model_g": "model", "age": "vehicle age",
        "engine_cc": "engine size", "mileage_km": "mileage",
        "body_type": "body type", "km_per_year": "km per year",
        "seats": "seats", "fuel_type": "fuel type",
        "transmission": "transmission", "province": "province",
        "engine_hp": "engine power", "is_luxury": "luxury brand",
        "has_mileage": "mileage reported", "has_engine": "engine size reported",
        "flag_pajak_hidup": "text: road tax paid",
        "flag_tangan_pertama": "text: first owner",
        "flag_service_record": "text: service history",
        "flag_nego": "text: negotiable",
    }
    fig, ax = plt.subplots(figsize=(10, 4.2))
    top = order[:12][::-1]
    ax.barh(np.arange(len(top)), imp.importances_mean[top], color=COOL, height=0.68,
            xerr=imp.importances_std[top], error_kw={"ecolor": "#c3cad1", "lw": 1})
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels([DISPLAY.get(names[i], names[i]) for i in top], fontsize=10)
    ax.set_xlabel("higher value = the price depends on it more")
    add_title(fig, "Effect of each detail on the price")
    IMAGES.mkdir(exist_ok=True)
    fig.savefig(IMAGES / "06_feature_importance.png", bbox_inches="tight",
                pad_inches=0.16, facecolor="white")
    plt.close(fig)
    out(f"saved {IMAGES / '06_feature_importance.png'}")

    pd.DataFrame({
        "ad_id": df["ad_id"], "brand": df["brand"], "model": df["model"],
        "year": df["year"], "price": df["price"],
        "pred_price": np.exp(pred_best),
        "oof_resid_log": resid,
        "weight": df["weight"],
    }).to_csv(PRED, index=False, compression="gzip")

    REPORT.write_text("\n".join(_lines), encoding="utf-8")
    out(f"report written to {REPORT}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
