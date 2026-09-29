import numpy as np

from src import config
from src.data import load_split, split_xy
from src.eda import top_pearson_feature
from src.explain import RawFeatureExplainer, original_feature, standardized_coefficients
from src.models import get_linear_models

SAMPLE_SIZE = 300


def test_original_feature_mapping():
    features = config.FEATURES
    assert original_feature("crop_type_Rice", features) == "crop_type"
    assert original_feature("soil_ph", features) == "soil_ph"


def test_raw_explainer_is_additive_and_readable():
    train, _ = load_split()
    sample = train.sample(SAMPLE_SIZE, random_state=config.RANDOM_STATE)
    X, y = split_xy(sample)
    model = get_linear_models(top_pearson_feature(train))["polynomial"].fit(X, y)
    explanation = RawFeatureExplainer(model, X).explain(X.head(3))
    # SHAP values plus the base value must reproduce each prediction exactly.
    np.testing.assert_allclose(
        explanation.values.sum(axis=1) + explanation.base_values,
        model.predict(X.head(3)),
        atol=1e-6,
    )
    assert explanation.feature_names == config.FEATURES


def test_coefficients_have_readable_baselines():
    train, _ = load_split()
    X, y = split_xy(train)
    model = get_linear_models(top_pearson_feature(train))["multiple_lr"].fit(X, y)
    table = standardized_coefficients(model, config.FEATURES)
    assert set(table["feature"]) == set(config.FEATURES)
    assert any("(vs " in label for label in table["label"])
