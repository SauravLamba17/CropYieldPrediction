"""Time-safe India features: target encoding, lag features and a year trend.

Forecasting assumption (used everywhere in Phase 4): when predicting year t, the
actual yields of every year up to t-1 are known, and nothing from year t or later.
So every encoding and every lag for a row of year t is computed ONLY from rows of
strictly earlier years:

  * Target encoding (TimeSafeTargetEncoder) lives inside the model pipeline. It
    stores the rows it was fitted on and encodes any row from the stored rows with
    an earlier year. Training rows therefore get an expanding-window encoding (years
    before their own), CV validation rows get only the earlier training-fold years,
    and test rows only the training years 1997-2015.
  * Lags (add_lags) are pure look-ups in a table of known yields: the previous
    year's yield, the mean of the last 3 years and the number of years of history
    for the same crop x state. The caller passes the known history; nothing is fitted.
  * Trend: Crop_Year as a numeric input. Trees cannot extrapolate it: every test year
    (2016-2020) lies beyond the training years, so it falls into the last split.

Everything is on the log1p(yield) scale, the scale the India models are trained on.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src import config

CROP, STATE, YEAR, TARGET = (
    config.INDIA_CROP,
    config.INDIA_STATE,
    config.INDIA_YEAR,
    config.INDIA_TARGET,
)
PAIR = [CROP, STATE]
TE_COLUMNS = ["te_crop_state", "te_crop", "te_state"]
LAG_COLUMNS = ["lag1_yield", "rolling3_yield", "history_years"]
TREND_COLUMNS = [YEAR]
LAG_WINDOW = 3

# Inputs (beyond the base features) of each ablation feature set.
FEATURE_SETS = {
    "base": [],
    "target_encoding": TE_COLUMNS,
    "lags": LAG_COLUMNS,
    "trend": TREND_COLUMNS,
    "all": TE_COLUMNS + LAG_COLUMNS + TREND_COLUMNS,
}


def prior_stats(
    history: pd.DataFrame, values: np.ndarray, rows: pd.DataFrame, keys: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Sum and count of `values` over the history rows with the same keys and an EARLIER year.

    history / rows: frames with the key columns and Crop_Year. values: one number per
    history row. Returns (sum, count) per row of `rows`, in its order; count 0 = no history.
    """
    key_cols = keys or ["_all"]
    hist = history[keys + [YEAR]].assign(_v=np.asarray(values, dtype=float), _all=0)
    per_year = hist.groupby(key_cols + [YEAR], sort=True)["_v"].agg(["sum", "count"]).reset_index()
    per_year[["sum", "count"]] = per_year.groupby(key_cols)[["sum", "count"]].cumsum()
    query = rows[keys + [YEAR]].assign(_all=0, _row=np.arange(len(rows)))
    # Cumulative totals at the latest stored year STRICTLY before each row's year.
    merged = pd.merge_asof(
        query.sort_values(YEAR),
        per_year.sort_values(YEAR),
        on=YEAR,
        by=key_cols,
        allow_exact_matches=False,
        direction="backward",
    ).sort_values("_row")
    return merged["sum"].fillna(0).to_numpy(), merged["count"].fillna(0).to_numpy()


def smoothed(total: np.ndarray, count: np.ndarray, prior: np.ndarray, strength: float) -> np.ndarray:
    """(total + strength * prior) / (count + strength); the prior alone when count = 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (total + strength * prior) / (count + strength)
    return np.where(count > 0, out, prior)


class TimeSafeTargetEncoder(BaseEstimator, TransformerMixin):
    """Smoothed mean log-yield of the crop x state, crop and state, from EARLIER years only.

    te_crop and te_state are shrunk toward the overall mean, te_crop_state toward
    te_crop (a crop x state with little history looks like its crop elsewhere).
    `smoothing` is the number of pseudo-rows of the prior; it is tuned by CV.
    Rows with no earlier year at all (the first year of a fit) get NaN, which the
    tree models handle natively. Columns are added; the preprocessor picks inputs.
    """

    def __init__(self, smoothing: float = 10.0) -> None:
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y) -> "TimeSafeTargetEncoder":
        """Store the training rows (keys, year) and their target (already on the log scale)."""
        self.history_ = X[PAIR + [YEAR]].reset_index(drop=True)
        self.target_ = np.asarray(y, dtype=float)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add the three encodings, each from stored rows with a year before the row's year."""
        stats = {
            name: prior_stats(self.history_, self.target_, X, keys)
            for name, keys in [("all", []), ("crop", [CROP]), ("state", [STATE]), ("pair", PAIR)]
        }
        total, count = stats["all"]
        with np.errstate(invalid="ignore", divide="ignore"):
            overall = np.where(count > 0, total / count, np.nan)
        m = self.smoothing
        te_crop = smoothed(*stats["crop"], overall, m)
        return X.assign(
            te_crop_state=smoothed(*stats["pair"], te_crop, m),
            te_crop=te_crop,
            te_state=smoothed(*stats["state"], overall, m),
        )


def yearly_log_yield(history: pd.DataFrame) -> pd.Series:
    """Mean log1p(yield) per crop x state x year (seasons of the same year are averaged)."""
    return np.log1p(history[TARGET]).groupby([history[k] for k in PAIR + [YEAR]]).mean()


def add_lags(rows: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Add the lag columns to `rows`, using only `history` rows of strictly earlier years.

    lag1_yield: last year's mean log1p(yield) of the same crop x state (NaN if not grown).
    rolling3_yield: mean of the yearly values in the 3 previous years (NaN if none).
    history_years: number of earlier years with a record for this crop x state.
    `history` is what is known at prediction time: the training years when fitting, and
    for the test years either the training years only or also the earlier test years.
    """
    panel = yearly_log_yield(history)
    lags = []
    for k in range(1, LAG_WINDOW + 1):
        index = pd.MultiIndex.from_arrays([rows[CROP], rows[STATE], rows[YEAR] - k])
        lags.append(panel.reindex(index).to_numpy())
    lags = np.vstack(lags)
    years = panel.reset_index()[PAIR + [YEAR]]
    _, n_years = prior_stats(years, np.ones(len(years)), rows, PAIR)
    present = (~np.isnan(lags)).sum(axis=0)
    rolling = np.where(present > 0, np.nansum(lags, axis=0) / np.maximum(present, 1), np.nan)
    return rows.assign(lag1_yield=lags[0], rolling3_yield=rolling, history_years=n_years)


def lookup_log(history: pd.DataFrame, values: np.ndarray, rows: pd.DataFrame) -> np.ndarray:
    """The crop x state look-up (mean log1p yield) from history rows of EARLIER years.

    Same rule as src.india.lookup_baseline: the crop x state mean, else the overall mean;
    for the test years (all later than every training year) it gives exactly that baseline.
    """
    pair_total, pair_count = prior_stats(history, values, rows, PAIR)
    total, count = prior_stats(history, values, rows, [])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(pair_count > 0, pair_total / pair_count, np.where(count > 0, total / count, np.nan))
