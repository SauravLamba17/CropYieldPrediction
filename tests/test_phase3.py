"""Phase 3: explainability tables, heatmap bounds, A vs B, per-group coverage, report export."""

import joblib
import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import load_split, split_xy
from src.eda import top_pearson_feature
from src.explain import RawFeatureExplainer
from src.explain_tables import bootstrap_coefficients, group_coverage, permutation_original
from src.models import get_linear_models
from ui import analysis
from ui.report import COMPARISON_COLUMNS, comparison_frame

RNG = np.random.default_rng(0)


@pytest.fixture(scope="module")
def train():
    return load_split()[0]


@pytest.fixture(scope="module")
def deployed():
    return joblib.load(config.MODELS_DIR / "best_overall.joblib")


def test_permutation_is_per_original_feature_for_one_hot_and_polynomial_terms(train):
    """Yield driven only by a category and by the SQUARE of a numeric input: both must show up
    under their original names (one-hot levels and polynomial terms traced back)."""
    sample = train.sample(400, random_state=1)
    X = sample[config.FEATURES]
    y = 5.0 * (X["crop_type"] == "Rice") + 2.0 * (X["soil_ph"] - X["soil_ph"].mean()) ** 2
    y = y + RNG.normal(0, 0.05, len(y))
    model = get_linear_models(top_pearson_feature(train))["polynomial"].fit(X, y)
    table = permutation_original(model, X, pd.Series(y), n_repeats=5).set_index("feature")
    assert sorted(table.index) == sorted(config.FEATURES)  # 14 rows, no one-hot / poly term names
    others = table.drop(index=["crop_type", "soil_ph"])["importance_mean"]
    assert table.loc["crop_type", "importance_mean"] > 10 * others.abs().max()
    assert table.loc["soil_ph", "importance_mean"] > 10 * others.abs().max()


def test_bootstrap_intervals_are_reproducible_with_a_fixed_seed(train):
    sample = train.sample(300, random_state=2)
    X, y = split_xy(sample)
    model = get_linear_models(top_pearson_feature(train))["multiple_lr"]
    a = bootstrap_coefficients(model, X, y, n_boot=15, seed=7)
    b = bootstrap_coefficients(model, X, y, n_boot=15, seed=7)
    c = bootstrap_coefficients(model, X, y, n_boot=15, seed=8)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["ci_low"], c["ci_low"])
    assert (a["ci_low"] <= a["ci_high"]).all()
    assert (a["crosses_zero"] == ((a["ci_low"] <= 0) & (a["ci_high"] >= 0))).all()


def test_heatmap_grid_never_leaves_the_observed_range(deployed, train):
    bounds = {
        c: {"min": float(train[c].min()), "max": float(train[c].max()),
            "is_integer": bool(pd.api.types.is_integer_dtype(train[c]))}
        for c in config.NUMERIC_FEATURES
    }
    inputs = train[config.FEATURES].iloc[0].to_dict()
    for fx, fy in [("fertilizer_amount", "total_rainfall"), ("irrigation_frequency", "soil_ph")]:
        grid = analysis.heatmap_grid(deployed, inputs, fx, fy, bounds)
        assert grid["x"].min() >= bounds[fx]["min"] and grid["x"].max() <= bounds[fx]["max"]
        assert grid["y"].min() >= bounds[fy]["min"] and grid["y"].max() <= bounds[fy]["max"]
        assert np.isfinite(grid["pred"]).all()
    assert len(analysis.heatmap_grid(deployed, inputs, "fertilizer_amount", "soil_ph", bounds)) == 25 * 25
    ints = analysis.observed_grid(1, 6, 25, is_integer=True)
    assert list(ints) == [1, 2, 3, 4, 5, 6]


def test_shap_difference_sums_to_the_prediction_gap(deployed, train):
    explainer = RawFeatureExplainer(deployed, train[config.FEATURES])
    a = train[config.FEATURES].iloc[[0]]
    b = a.assign(fertilizer_amount=a["fertilizer_amount"] * 1.5, region="North")
    ea, eb = explainer.explain(a)[0], explainer.explain(b)[0]
    gap = analysis.shap_difference(
        dict(zip(ea.feature_names, ea.values)), dict(zip(eb.feature_names, eb.values))
    )
    pred_gap = float(deployed.predict(b)[0] - deployed.predict(a)[0])
    assert gap.sum() == pytest.approx(pred_gap, abs=1e-6)
    assert gap.index[0] == "fertilizer_amount"  # largest |gap| first


def test_group_coverage_on_a_known_toy_input():
    covered = pd.Series([True, False, True, True, False, True])
    groups = {"crop": pd.Series(["A", "A", "B", "B", "B", "C"])}
    table = group_coverage(covered, groups).set_index("group")
    assert table.loc["A", "coverage"] == 0.5 and table.loc["A", "n"] == 2
    assert table.loc["B", "coverage"] == pytest.approx(2 / 3)
    assert table.loc["C", "coverage"] == 1.0
    assert set(table["group_by"]) == {"crop"}


def test_optimum_region_finds_an_interior_peak_and_rejects_an_edge():
    x = np.linspace(4, 9, 41)
    hump = analysis.optimum_region(pd.DataFrame({"x": x, "y": -((x - 6.5) ** 2)}))
    assert hump["interior"] and hump["peak_x"] == pytest.approx(6.5, abs=0.13)
    assert hump["low"] < 6.5 < hump["high"]
    edge = analysis.optimum_region(pd.DataFrame({"x": x, "y": x}))
    assert not edge["interior"] and edge["peak_x"] == 9


def test_method_agreement_is_one_for_identical_rankings():
    scores = pd.DataFrame({"permutation": [3, 2, 1], "shap": [30, 20, 10], "coefficient": [1, 2, 3]})
    agree = analysis.method_agreement(scores).set_index(["a", "b"])["rho"]
    assert agree[("permutation", "shap")] == pytest.approx(1.0)
    assert agree[("permutation", "coefficient")] == pytest.approx(-1.0)


def test_saved_explainability_tables():
    perm = pd.read_csv(config.TABLES_DIR / "explain_permutation.csv")
    shap = pd.read_csv(config.TABLES_DIR / "explain_shap.csv")
    coef = pd.read_csv(config.TABLES_DIR / "explain_coefficients.csv")
    cov = pd.read_csv(config.TABLES_DIR / "conformal_groups.csv")
    for table in (perm, shap):
        assert table.groupby("model")["feature"].apply(sorted).map(lambda f: f == sorted(config.FEATURES)).all()
    assert (perm["n_repeats"] >= 30).all() and (coef["n_boot"] >= 500).all()
    assert set(coef["model"]) == {"multiple_lr", "ridge", "lasso"}
    assert set(cov["group_by"]) == {"crop_type", "region"}
    assert cov.groupby("group_by")["n"].sum().eq(960).all()


def test_comparison_export_columns_and_gap():
    inputs = {c: (2.0 if c in config.NUMERIC_FEATURES else "Rice") for c in config.FEATURES}
    b = {**inputs, "fertilizer_amount": 3.0}
    frame = comparison_frame(inputs, b, 6.0, 6.6, 1.0, {"fertilizer_amount": 0.6}, "polynomial",
                             "t", "2.0.0", "abc")
    assert list(frame.columns) == COMPARISON_COLUMNS
    rows = frame.set_index("item")
    assert rows.loc["Predicted yield", "difference"] == "+0.60"
    assert rows.loc["Change", "difference"] == "+10.0%"
    inputs_rows = frame.query("section == 'Input'").set_index("item")
    assert inputs_rows.loc["Fertilizer", "difference"] == "+1.00"
    assert frame.query("section == 'Gap explained by (SHAP)'")["difference"].tolist() == ["+0.600"]
