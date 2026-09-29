import numpy as np
import pytest

from src import config
from src.data import load_split, split_xy
from src.eda import top_pearson_feature
from src.features import build_preprocessor
from src.models import get_linear_models

SAMPLE_SIZE = 200


@pytest.fixture(scope="module")
def train():
    train_df, _ = load_split()
    return train_df


@pytest.fixture(scope="module")
def sample(train):
    return train.sample(SAMPLE_SIZE, random_state=config.RANDOM_STATE)


def all_models(train, log_target=False):
    return get_linear_models(top_pearson_feature(train), log_target=log_target)


@pytest.mark.parametrize("log_target", [False, True])
def test_every_pipeline_fits_and_predicts(train, sample, log_target):
    X, y = split_xy(sample)
    for name, model in all_models(train, log_target).items():
        pred = model.fit(X, y).predict(X)
        assert len(pred) == len(X), name
        assert np.all(np.isfinite(pred)), name
        # Predictions must be in t/ha, not log units: same order of magnitude as y.
        assert 0.1 * y.mean() < pred.mean() < 10 * y.mean(), name


def test_preprocessor_fitted_only_on_fit_data(train):
    a = train.iloc[:SAMPLE_SIZE]
    b = train.iloc[SAMPLE_SIZE : 2 * SAMPLE_SIZE]
    pre = build_preprocessor(config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES)
    pre.fit(a[config.FEATURES])
    pre.transform(b[config.FEATURES])  # transforming B must not refit anything
    scaler = pre.named_transformers_["num"][-1]
    np.testing.assert_allclose(scaler.mean_, a[config.NUMERIC_FEATURES].mean().to_numpy())
    assert not np.allclose(scaler.mean_, b[config.NUMERIC_FEATURES].mean().to_numpy())


def test_simple_lr_uses_one_feature(train, sample):
    X, y = split_xy(sample)
    model = all_models(train)["simple_lr"].fit(X, y)
    assert model[-1].coef_.shape == (1,)


def test_polynomial_expands_numeric_only(train, sample):
    X, y = split_xy(sample)
    model = all_models(train)["polynomial"].fit(X, y)
    names = model[0].get_feature_names_out()
    n = len(config.NUMERIC_FEATURES)
    n_poly = n + n * (n + 1) // 2  # linear + squares + pairwise products
    n_onehot = sum(sample[c].nunique() for c in config.CATEGORICAL_FEATURES)
    assert len(names) == n_poly + n_onehot
