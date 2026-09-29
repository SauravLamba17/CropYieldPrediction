import numpy as np
import pandas as pd
import pytest

from src import config
from src.india import (
    YearBlockSplit,
    ablation_table,
    clean_india,
    feature_ranges,
    india_models,
    india_xy,
    load_india_raw,
    lookup_baseline,
    time_split,
)

pytestmark = pytest.mark.skipif(not config.INDIA_CSV.exists(), reason="India CSV not downloaded")


@pytest.fixture(scope="module")
def raw():
    return load_india_raw()


@pytest.fixture(scope="module")
def clean(raw):
    return clean_india(raw)[0]


@pytest.fixture(scope="module")
def split(clean):
    return time_split(clean)


def test_production_is_not_a_feature(clean, split):
    assert config.INDIA_PRODUCTION not in config.INDIA_FEATURES
    assert config.INDIA_PRODUCTION not in clean.columns
    train, _ = split
    X, y = india_xy(train.head(300))
    model = india_models(train)["multiple_lr"].fit(X, y)
    model_inputs = model.regressor_[0].get_feature_names_out()
    assert not any(config.INDIA_PRODUCTION in name for name in model_inputs)
    # Crop_Year is only for splitting; it must not reach the model directly (the
    # per-hectare inputs still encode the year implicitly -- see CODE_REVIEW.md H1).
    assert not any(config.INDIA_YEAR in name for name in model_inputs)


def test_time_split_has_no_future_in_train(split):
    train, test = split
    assert train[config.INDIA_YEAR].max() < test[config.INDIA_YEAR].min()
    assert test[config.INDIA_YEAR].min() >= config.INDIA_TEST_FIRST_YEAR


def toy_raw() -> pd.DataFrame:
    """Four hand-made rows: two valid, one with Area 0, one Coconut; text has trailing spaces."""
    return pd.DataFrame(
        {
            config.INDIA_CROP: ["Rice ", "Wheat", "Rice", "Coconut "],
            config.INDIA_YEAR: [2000, 2001, 2002, 2003],
            config.INDIA_SEASON: ["Kharif     ", "Rabi       ", "Kharif", "Whole Year "],
            config.INDIA_STATE: ["Assam", "Assam", "Assam", "Kerala"],
            config.INDIA_AREA: [100.0, 50.0, 0.0, 10.0],
            config.INDIA_PRODUCTION: [300, 100, 5, 90000],
            config.INDIA_RAINFALL: [2000.0, 2100.0, 1900.0, 3000.0],
            config.INDIA_FERTILIZER: [15000.0, 5000.0, 10.0, 1500.0],
            config.INDIA_PESTICIDE: [30.0, 10.0, 1.0, 3.0],
            config.INDIA_TARGET: [3.0, 2.0, 1.0, 9000.0],
        }
    )


def test_fertilizer_and_pesticide_are_per_hectare():
    clean, removed = clean_india(toy_raw())
    assert removed["area_or_yield_le_0"] == 1 and removed["excluded_crops"] == 1
    np.testing.assert_allclose(clean[config.INDIA_FERTILIZER_PER_HA], [150.0, 100.0])
    np.testing.assert_allclose(clean[config.INDIA_PESTICIDE_PER_HA], [0.3, 0.2])
    np.testing.assert_allclose(clean[config.INDIA_LOG_AREA], np.log1p([100.0, 50.0]))
    # The raw totals (and raw Area) never survive cleaning or enter the feature set.
    for col in (config.INDIA_FERTILIZER, config.INDIA_PESTICIDE, config.INDIA_AREA):
        assert col not in clean.columns
        assert col not in config.INDIA_FEATURES


def test_no_nonpositive_area_or_yield(clean):
    assert (clean[config.INDIA_TARGET] > 0).all()
    assert (np.expm1(clean[config.INDIA_LOG_AREA]) > 0).all()
    assert not clean[config.INDIA_CROP].isin(config.INDIA_EXCLUDED_CROPS).any()


def test_whitespace_is_stripped(clean):
    for col in config.INDIA_CATEGORICAL_FEATURES:
        assert (clean[col] == clean[col].str.strip()).all(), col
    assert list(clean_india(toy_raw())[0][config.INDIA_SEASON]) == ["Kharif", "Rabi"]


def test_year_block_split_is_time_ordered(split):
    train, _ = split
    X, _ = india_xy(train)
    for tr_idx, val_idx in YearBlockSplit(5).split(X):
        assert X[config.INDIA_YEAR].iloc[tr_idx].max() < X[config.INDIA_YEAR].iloc[val_idx].min()


def test_year_block_split_on_toy_data():
    X = pd.DataFrame({config.INDIA_YEAR: [2000, 2000, 2001, 2002, 2003, 2003]})
    folds = list(YearBlockSplit(3).split(X))
    assert [list(v) for _, v in folds] == [[2], [3], [4, 5]]
    first_train, _ = folds[0]
    assert list(first_train) == [0, 1]


def test_lookup_baseline_uses_training_group_means():
    train = pd.DataFrame(
        {config.INDIA_CROP: ["Rice", "Rice", "Wheat"], config.INDIA_TARGET: [1.0, 3.0, 2.0]}
    )
    test = pd.DataFrame({config.INDIA_CROP: ["Rice", "Wheat", "Maize"]})
    pred = lookup_baseline(train, test, [config.INDIA_CROP])
    rice = np.expm1(np.log1p([1.0, 3.0]).mean())  # geometric-style mean on the log1p scale
    overall = np.expm1(np.log1p([1.0, 3.0, 2.0]).mean())  # unseen crop -> overall mean
    np.testing.assert_allclose(pred, [rice, 2.0, overall])


def test_ablation_models_exclude_year_index_columns(split):
    train, _ = split
    reduced = [
        c for c in config.INDIA_NUMERIC_FEATURES if c not in config.INDIA_YEAR_INDEX_FEATURES
    ]
    X, y = india_xy(train.head(300))  # X still contains the columns; the model must ignore them
    for name in ("multiple_lr", "simple_lr"):
        model = india_models(train, reduced)[name].fit(X, y)
        inputs = model.regressor_[0].get_feature_names_out()
        for col in config.INDIA_YEAR_INDEX_FEATURES:
            assert col not in inputs, (name, col)


def test_ablation_table_and_ranges():
    with_cols = pd.DataFrame({"model": ["a", "b"], "rmse": [1.0, 2.0], "r2": [0.5, 0.4]})
    without = pd.DataFrame({"model": ["a", "b"], "rmse": [1.5, 1.8], "r2": [0.3, 0.45]})
    table = ablation_table(with_cols, without).set_index("model")
    np.testing.assert_allclose(table["delta_rmse"], [0.5, -0.2])  # a: columns help; b: they hurt
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    test = pd.DataFrame({"x": [2.0, 4.0, 5.0, 0.0]})
    r = feature_ranges(train, test, ["x"]).iloc[0]
    assert r["share_test_rows_above_train_max"] == 0.5
    assert r["share_test_rows_below_train_min"] == 0.25
