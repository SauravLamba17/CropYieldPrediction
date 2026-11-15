"""Phase 4: time-safe India features, the hybrid, the search budget and the leakage audit."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src import config
from src.features import build_preprocessor
from src.india import lookup_baseline
from src.india_features import (
    LAG_COLUMNS,
    TE_COLUMNS,
    TimeSafeTargetEncoder,
    add_lags,
    lookup_log,
    prior_stats,
)
from src.india_models import INDIA_SEARCH_ITERATIONS, LookupResidualRegressor, india_candidate

CROP, STATE, YEAR, Y = config.INDIA_CROP, config.INDIA_STATE, config.INDIA_YEAR, config.INDIA_TARGET


def toy_panel() -> pd.DataFrame:
    """Two crops x two states, 2000-2003; Rice/S1 skips 2001; Wheat/S2 starts in 2002."""
    rows = [
        ("Rice", "S1", 2000, 1.0), ("Rice", "S1", 2002, 3.0), ("Rice", "S1", 2003, 4.0),
        ("Rice", "S2", 2000, 2.0), ("Rice", "S2", 2001, 2.0), ("Rice", "S2", 2002, 2.0),
        ("Wheat", "S1", 2000, 5.0), ("Wheat", "S1", 2001, 6.0), ("Wheat", "S1", 2001, 8.0),
        ("Wheat", "S2", 2002, 9.0), ("Wheat", "S2", 2003, 7.0),
    ]
    return pd.DataFrame(rows, columns=[CROP, STATE, YEAR, Y])


def test_prior_stats_use_strictly_earlier_years():
    panel = toy_panel()
    query = pd.DataFrame({CROP: ["Rice", "Rice", "Wheat"], STATE: ["S1", "S1", "S2"], YEAR: [2000, 2003, 2002]})
    total, count = prior_stats(panel, panel[Y].to_numpy(), query, [CROP, STATE])
    # Rice/S1 in 2000: nothing earlier; in 2003: 2000 and 2002 (not 2003 itself); Wheat/S2 2002: none.
    np.testing.assert_array_equal(count, [0, 2, 0])
    np.testing.assert_array_equal(total, [0, 4.0, 0])


def test_target_encoding_known_answers():
    panel = toy_panel()
    y = np.log1p(panel[Y].to_numpy())
    encoder = TimeSafeTargetEncoder(smoothing=2.0).fit(panel, y)
    query = pd.DataFrame({CROP: ["Rice", "Rice"], STATE: ["S1", "S1"], YEAR: [2001, 2000]})
    out = encoder.transform(query)
    # 2001: only the 2000 rows are known.
    y2000 = np.log1p([1.0, 2.0, 5.0])
    overall = y2000.mean()
    te_crop = (np.log1p(1.0) + np.log1p(2.0) + 2 * overall) / (2 + 2)
    te_pair = (np.log1p(1.0) + 2 * te_crop) / (1 + 2)
    te_state = (np.log1p(1.0) + np.log1p(5.0) + 2 * overall) / (2 + 2)
    np.testing.assert_allclose(out.loc[0, TE_COLUMNS], [te_pair, te_crop, te_state])
    assert out.loc[1, TE_COLUMNS].isna().all()  # the first year has no history at all


def test_target_encoding_is_out_of_fold_in_fit_transform():
    """A training row's own target (and its year's) never enters its own encoding."""
    panel = toy_panel()
    y = np.log1p(panel[Y].to_numpy())
    encoded = TimeSafeTargetEncoder(smoothing=1.0).fit_transform(panel, y)
    y_changed = y.copy()
    y_changed[panel[YEAR].to_numpy() == 2003] += 100.0  # change only the last year's targets
    changed = TimeSafeTargetEncoder(smoothing=1.0).fit_transform(panel, y_changed)
    pd.testing.assert_frame_equal(encoded, changed)  # every row is from 2003 or earlier, none sees 2003


def test_lags_previous_year_values_and_missing_history():
    panel = toy_panel()
    out = add_lags(panel, panel)
    rice_s1 = out[(out[CROP] == "Rice") & (out[STATE] == "S1")].set_index(YEAR)
    assert np.isnan(rice_s1.loc[2000, "lag1_yield"])  # no history
    assert np.isnan(rice_s1.loc[2002, "lag1_yield"])  # not grown in 2001
    assert rice_s1.loc[2003, "lag1_yield"] == pytest.approx(np.log1p(3.0))
    assert rice_s1.loc[2003, "rolling3_yield"] == pytest.approx(np.mean(np.log1p([3.0, 1.0])))
    assert list(rice_s1["history_years"]) == [0, 1, 2]
    wheat_s1 = out[(out[CROP] == "Wheat") & (out[STATE] == "S1") & (out[YEAR] == 2001)]
    np.testing.assert_allclose(wheat_s1["lag1_yield"], np.log1p(5.0))
    # Two 2001 Wheat/S1 rows: a 2002 query would average both seasons of 2001.
    query = pd.DataFrame({CROP: ["Wheat"], STATE: ["S1"], YEAR: [2002]})
    assert add_lags(query, panel)["lag1_yield"].iloc[0] == pytest.approx(np.log1p([6.0, 8.0]).mean())


def test_lags_ignore_the_rows_own_year_and_later():
    panel = toy_panel()
    later = panel[YEAR] >= 2002
    changed = panel.assign(**{Y: np.where(later, panel[Y] * 50, panel[Y])})
    first = add_lags(panel, panel)
    second = add_lags(panel, changed)
    rows = (panel[YEAR] <= 2002).to_numpy()  # their history is 2001 or earlier: unchanged
    pd.testing.assert_frame_equal(first.loc[rows, LAG_COLUMNS], second.loc[rows, LAG_COLUMNS])


def test_lookup_fallback_for_unseen_pairs_and_crops():
    panel = toy_panel()
    test = pd.DataFrame({CROP: ["Rice", "Wheat", "Maize"], STATE: ["S1", "S3", "S1"], YEAR: [2004] * 3})
    pred = lookup_baseline(panel, test, [CROP, STATE])
    logs = np.log1p(panel[Y])
    rice_s1 = logs[(panel[CROP] == "Rice") & (panel[STATE] == "S1")].mean()
    np.testing.assert_allclose(pred, np.expm1([rice_s1, logs.mean(), logs.mean()]))
    # The time-safe version gives exactly the same for rows later than every fitted year.
    np.testing.assert_allclose(lookup_log(panel, logs.to_numpy(), test), [rice_s1, logs.mean(), logs.mean()])


def test_hybrid_is_lookup_plus_residual_model():
    panel = toy_panel().assign(x=np.arange(11.0))
    y = np.log1p(panel[Y].to_numpy())
    residual = Pipeline([("preprocess", build_preprocessor(["x"], [])), ("model", LinearRegression())])
    hybrid = LookupResidualRegressor(residual).fit(panel, y)
    rows = pd.DataFrame({CROP: ["Rice", "Wheat"], STATE: ["S2", "S1"], YEAR: [2004, 2004], "x": [3.0, 7.0]})
    expected = hybrid.offset(rows) + hybrid.estimator_.predict(rows)
    np.testing.assert_allclose(hybrid.offset(rows), lookup_log(panel, y, rows))  # rows after every fitted year
    np.testing.assert_allclose(hybrid.predict(rows), expected)
    # The residual target is y minus the time-safe look-up; the first year (no earlier look-up)
    # falls back to the fitted mean, so a single-year fit (earliest inner fold) still works.
    mean_model = LookupResidualRegressor(DummyRegressor(strategy="mean")).fit(panel, y)
    offset = np.nan_to_num(lookup_log(panel, y, panel), nan=y.mean())
    assert mean_model.estimator_.constant_[0][0] == pytest.approx((y - offset).mean())
    first_year = panel[YEAR] == 2000
    LookupResidualRegressor(residual).fit(panel[first_year], y[first_year])  # raised before (0 rows)
    constant = LookupResidualRegressor(DummyRegressor(strategy="constant", constant=0.0)).fit(panel, y)
    np.testing.assert_allclose(constant.predict(rows), lookup_log(panel, y, rows))


@pytest.mark.parametrize(("family", "feature_set", "hybrid"), [("lightgbm", "all", False),
                                                              ("hist_gradient_boost", "lags", True)])
def test_search_budget_is_fixed_and_seeded(family, feature_set, hybrid):
    search = india_candidate(family, feature_set, hybrid)
    assert search.n_iter == INDIA_SEARCH_ITERATIONS
    assert search.random_state == config.RANDOM_STATE
    assert search.cv.n_splits == 3 and type(search.cv).__name__ == "YearBlockSplit"
    smoothing = [k for k in search.param_distributions if k.endswith("encode__smoothing")]
    assert bool(smoothing) == (feature_set == "all")  # the encoder is tuned only when it is used


def test_search_samples_the_same_candidates_every_time():
    from sklearn.model_selection import ParameterSampler

    search = india_candidate("lightgbm", "all")
    draw = lambda: list(ParameterSampler(search.param_distributions, search.n_iter,
                                         random_state=search.random_state))
    first, second = draw(), draw()
    assert len(first) == INDIA_SEARCH_ITERATIONS and first == second


@pytest.mark.skipif(not config.INDIA_TRAIN_PARQUET.exists(), reason="India split not built")
def test_leakage_audit_on_the_real_split():
    """Encodings and lags of every row use strictly earlier years; test years never reach a fit."""
    from src.india_analysis import model_inputs

    train = pd.read_parquet(config.INDIA_TRAIN_PARQUET)
    test = pd.read_parquet(config.INDIA_TEST_PARQUET)
    X_train, X_test, X_strict = model_inputs(train, test)
    assert X_train[YEAR].max() <= config.INDIA_TRAIN_LAST_YEAR
    assert config.INDIA_TARGET not in X_train.columns
    # Lags: scrambling the yields of year >= t leaves every row of year t unchanged.
    everything = pd.concat([train, test], ignore_index=True)
    for year in (2005, 2016, 2019):
        scrambled = everything.assign(**{Y: np.where(everything[YEAR] >= year, 1e6, everything[Y])})
        rows = everything[everything[YEAR] == year]
        pd.testing.assert_frame_equal(add_lags(rows, everything)[LAG_COLUMNS], add_lags(rows, scrambled)[LAG_COLUMNS])
    # Strict variant: no test-year yield at all; the 2017+ lag1 is therefore missing.
    assert X_strict.loc[test[YEAR].to_numpy() >= 2017, "lag1_yield"].isna().all()
    # Encoder fitted on the training years: test rows are encoded from <= 2015 only.
    encoder = TimeSafeTargetEncoder().fit(X_train, np.log1p(train[Y]))
    assert encoder.history_[YEAR].max() <= config.INDIA_TRAIN_LAST_YEAR
    encoded = encoder.transform(X_test)
    shifted = encoder.transform(X_test.assign(**{YEAR: config.INDIA_TEST_FIRST_YEAR}))
    pd.testing.assert_frame_equal(encoded[TE_COLUMNS], shifted[TE_COLUMNS])


def test_histgb_trains_when_every_encoding_is_missing():
    """Regression: the earliest inner fold (one training year) has all-NaN encodings and lags."""
    from src.india_features import add_lags
    from src.india_models import FillEmptyColumns

    panel = toy_panel()
    first_year = panel[panel[YEAR] == 2000]
    X = add_lags(first_year, first_year).assign(**{c: 1.0 for c in config.INDIA_NUMERIC_FEATURES},
                                                **{config.INDIA_SEASON: "Kharif"})
    search = india_candidate("hist_gradient_boost", "all", n_iter=1)
    model = search.estimator.set_params(regressor__model__max_iter=5, regressor__model__min_samples_leaf=1)
    model.fit(X, first_year[Y])  # raised "window shape cannot be larger" before FillEmptyColumns
    filler = model.regressor_.named_steps["fill_empty"]
    assert isinstance(filler, FillEmptyColumns) and filler.empty_.any()
    assert search.error_score == "raise"  # a failed fit stops the run instead of scoring NaN
