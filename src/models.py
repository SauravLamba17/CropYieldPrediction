"""Model definitions. Every model is a full Pipeline: preprocessing + regressor."""

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.base import RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import BaseCrossValidator, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline

try:
    from xgboost import XGBRegressor as BoostingRegressor

    BOOSTING_LIBRARY = "xgboost"
except ImportError:  # fall back to LightGBM if xgboost is unavailable
    from lightgbm import LGBMRegressor as BoostingRegressor

    BOOSTING_LIBRARY = "lightgbm"

from src import config
from src.data import make_kfold
from src.features import build_preprocessor

RIDGE_ALPHAS = np.logspace(-3, 3, 50)
LASSO_MAX_ITER = 20000
POLY_DEGREE = 2
CUBIC_DEGREE = 3
MAX_NUMERIC_FOR_CUBIC = 10  # degree 3 on more features explodes the term count
SEARCH_ITERATIONS = 20  # light tuning: 20 random configurations per model

# Search spaces are deliberately modest ("light tuning"); distributions are
# sampled reproducibly via random_state.
RF_PARAM_DISTRIBUTIONS = {
    "n_estimators": randint(200, 601),
    "max_depth": [None, 6, 10, 16],
    "min_samples_leaf": randint(1, 21),
    "max_features": [0.3, 0.5, 0.8, 1.0],
}
BOOSTING_PARAM_DISTRIBUTIONS = {
    "n_estimators": randint(200, 1001),
    "learning_rate": loguniform(0.01, 0.2),
    "max_depth": randint(2, 7),
    "subsample": uniform(0.6, 0.4),  # U(0.6, 1.0)
    "colsample_bytree": uniform(0.6, 0.4),
    "min_child_weight": randint(1, 21),
    "reg_lambda": loguniform(0.1, 10),
}


def wrap_target(model: Pipeline, log_target: bool) -> RegressorMixin:
    """Optionally model log1p(y); predictions are always returned in original units."""
    if not log_target:
        return model
    return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)


def get_linear_models(
    simple_feature: str,
    numeric: list[str] = config.NUMERIC_FEATURES,
    categorical: list[str] = config.CATEGORICAL_FEATURES,
    log_target: bool = False,
) -> dict[str, RegressorMixin]:
    """Baseline + linear model family, each as an unfitted pipeline.

    simple_feature: the single numeric feature for simple_lr. It must be chosen
        on the TRAINING split only (see src.eda.top_pearson_feature).
    """
    models = {
        "dummy_mean": make_pipeline(
            build_preprocessor(numeric, categorical), DummyRegressor(strategy="mean")
        ),
        "simple_lr": make_pipeline(build_preprocessor([simple_feature], []), LinearRegression()),
        "multiple_lr": make_pipeline(
            build_preprocessor(numeric, categorical, drop_first=True), LinearRegression()
        ),
        # Plain OLS on polynomial terms is unstable (many correlated terms such as
        # x and x^2, high variance), so the expansion is always paired with RidgeCV.
        "polynomial": make_pipeline(
            build_preprocessor(numeric, categorical, poly_degree=POLY_DEGREE),
            RidgeCV(alphas=RIDGE_ALPHAS),
        ),
        "ridge": make_pipeline(
            build_preprocessor(numeric, categorical), RidgeCV(alphas=RIDGE_ALPHAS)
        ),
        "lasso": make_pipeline(
            build_preprocessor(numeric, categorical),
            LassoCV(cv=config.CV_FOLDS, random_state=config.RANDOM_STATE, max_iter=LASSO_MAX_ITER),
        ),
    }
    if len(numeric) <= MAX_NUMERIC_FOR_CUBIC:
        models["polynomial_deg3"] = make_pipeline(
            build_preprocessor(numeric, categorical, poly_degree=CUBIC_DEGREE),
            RidgeCV(alphas=RIDGE_ALPHAS),
        )
    return {name: wrap_target(model, log_target) for name, model in models.items()}


def get_nonlinear_models(
    numeric: list[str] = config.NUMERIC_FEATURES,
    categorical: list[str] = config.CATEGORICAL_FEATURES,
    log_target: bool = False,
    cv: BaseCrossValidator | None = None,
) -> dict[str, RandomizedSearchCV]:
    """Random forest and gradient boosting, each wrapped in a light RandomizedSearchCV.

    The search is part of the estimator, so cross-validating it gives NESTED CV:
    hyperparameters are re-tuned inside every outer training fold, exactly as
    RidgeCV/LassoCV re-tune alpha inside every fold for the linear models.
    cv: splitter for the inner search (default: the shared shuffled KFold).
    """
    # All cores are used ONCE, at the search level (n_jobs=-1 below). Also giving
    # the trees n_jobs=-1 nests a second parallel layer inside each worker, which
    # joblib caps to one core anyway and which floods the log with warnings.
    candidates = {
        "random_forest": (
            RandomForestRegressor(random_state=config.RANDOM_STATE, n_jobs=1),
            RF_PARAM_DISTRIBUTIONS,
        ),
        "gradient_boost": (
            BoostingRegressor(random_state=config.RANDOM_STATE, n_jobs=1),
            BOOSTING_PARAM_DISTRIBUTIONS,
        ),
    }
    searches = {}
    for name, (regressor, distributions) in candidates.items():
        # Trees need no scaling, but sharing the preprocessor keeps one code path;
        # scaling is monotonic so it cannot change any split.
        pipe = Pipeline(
            [("preprocess", build_preprocessor(numeric, categorical)), ("model", regressor)]
        )
        prefix = "model__"
        estimator = wrap_target(pipe, log_target)
        if log_target:
            prefix = "regressor__" + prefix
        searches[name] = RandomizedSearchCV(
            estimator,
            {prefix + key: value for key, value in distributions.items()},
            n_iter=SEARCH_ITERATIONS,
            cv=make_kfold() if cv is None else cv,
            scoring="neg_root_mean_squared_error",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
    return searches


def deployable(model: RegressorMixin) -> RegressorMixin:
    """The refitted best estimator of a search (else the model itself), for saving to disk.

    Saving the search wrapper would also pickle its CV splitter, which is not needed
    for prediction and breaks loading when the splitter is defined in a __main__ script.
    """
    return getattr(model, "best_estimator_", model)  # only fitted searches have it


def fitted_pipeline(model: RegressorMixin) -> Pipeline:
    """The fitted Pipeline inside a model that may be searched and/or target-wrapped."""
    model = deployable(model)
    if isinstance(model, TransformedTargetRegressor):
        model = model.regressor_
    return model


def final_estimator(model: RegressorMixin) -> RegressorMixin:
    """The regressor at the end of a fitted model's pipeline."""
    return fitted_pipeline(model)[-1]


def chosen_alpha(model: RegressorMixin) -> float | None:
    """Regularisation strength picked by RidgeCV/LassoCV, or None for other models."""
    return getattr(final_estimator(model), "alpha_", None)
