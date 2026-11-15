"""Model definitions. Every model is a full Pipeline: preprocessing + regressor."""

import copy

import numpy as np
from lightgbm import LGBMRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.base import RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import BaseCrossValidator, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline

try:
    from xgboost import XGBRegressor as BoostingRegressor

    BOOSTING_LIBRARY = "xgboost"
except ImportError:  # fall back to LightGBM if xgboost is unavailable
    BoostingRegressor = LGBMRegressor
    BOOSTING_LIBRARY = "lightgbm"

from src import config
from src.data import make_kfold
from src.features import build_preprocessor

RIDGE_ALPHAS = np.logspace(-3, 3, 50)
# Only used when the penalty is tuned on given folds (India): LassoCV builds its own
# data-driven grid, GridSearchCV needs an explicit one. 50 values, 1e-5 to 1.
LASSO_ALPHAS = np.logspace(-5, 0, 50)
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
LIGHTGBM_PARAM_DISTRIBUTIONS = {
    "n_estimators": randint(200, 1001),
    "learning_rate": loguniform(0.01, 0.2),
    "num_leaves": randint(8, 128),
    "min_child_samples": randint(5, 51),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "reg_lambda": loguniform(0.1, 10),
}
HISTGB_PARAM_DISTRIBUTIONS = {
    "max_iter": randint(200, 1001),
    "learning_rate": loguniform(0.01, 0.2),
    "max_leaf_nodes": randint(8, 128),
    "min_samples_leaf": randint(5, 51),
    "l2_regularization": loguniform(1e-3, 10),
    "max_features": uniform(0.6, 0.4),
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
    cv: BaseCrossValidator | None = None,
) -> dict[str, RegressorMixin]:
    """Baseline + linear model family, each as an unfitted pipeline.

    simple_feature: the single numeric feature for simple_lr. It must be chosen
        on the TRAINING split only (see src.eda.top_pearson_feature).
    cv: None (Kaggle) tunes the penalty with RidgeCV (efficient leave-one-out) and
        LassoCV (5 unshuffled folds). If a splitter is given (India: time-ordered year
        blocks), Ridge/Lasso are instead wrapped in a GridSearchCV over the same alpha
        grid on those folds, so the penalty is never chosen by scoring on past years
        with a model that saw later ones.
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
    models = {name: wrap_target(model, log_target) for name, model in models.items()}
    if cv is not None:
        for name in [n for n in PENALISED_LINEAR if n in models]:
            models[name] = time_tuned(models[name], log_target, cv)
    return models


PENALISED_LINEAR = ["polynomial", "ridge", "lasso", "polynomial_deg3"]


def time_tuned(model: RegressorMixin, log_target: bool, cv: BaseCrossValidator) -> RegressorMixin:
    """Swap RidgeCV/LassoCV for plain Ridge/Lasso and tune alpha with GridSearchCV on `cv`.

    Only the folds change: like RidgeCV/LassoCV, the search sits INSIDE the log-target
    wrapper, so alpha is still chosen by squared error on the modelling (log) scale.
    Scoring in original units instead overflows expm1 for weakly penalised candidates on
    later years (NaN scores that silently drop those candidates).
    """
    pipe = model.regressor if log_target else model
    if isinstance(pipe[-1], LassoCV):
        step, penalised, grid = "lasso", Lasso(max_iter=LASSO_MAX_ITER), LASSO_ALPHAS
    else:
        step, penalised, grid = "ridge", Ridge(), RIDGE_ALPHAS
    pipe = Pipeline([*pipe.steps[:-1], (step, penalised)])
    # n_jobs=-1: a cold-started Lasso grid is slow (about 12 min serially on India); inside an
    # already-parallel outer CV joblib runs it serially, so cores are never oversubscribed.
    search = GridSearchCV(
        pipe, {f"{step}__alpha": grid}, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1
    )
    return wrap_target(search, log_target)


def tree_regressors() -> dict[str, tuple[RegressorMixin, dict]]:
    """Every tree model with its search space (unfitted, seeded, single-threaded).

    All cores are used ONCE, at the search level (n_jobs=-1 in tuned_search). Also giving
    the trees n_jobs=-1 nests a second parallel layer inside each worker, which joblib
    caps to one core anyway and which floods the log with warnings.
    """
    return {
        "random_forest": (
            RandomForestRegressor(random_state=config.RANDOM_STATE, n_jobs=1),
            RF_PARAM_DISTRIBUTIONS,
        ),
        "gradient_boost": (
            BoostingRegressor(random_state=config.RANDOM_STATE, n_jobs=1),
            BOOSTING_PARAM_DISTRIBUTIONS,
        ),
        # Phase 4. subsample_freq=1 makes `subsample` take effect; deterministic +
        # row-wise histograms give identical results run to run.
        "lightgbm": (
            LGBMRegressor(
                random_state=config.RANDOM_STATE,
                n_jobs=1,
                subsample_freq=1,
                deterministic=True,
                force_row_wise=True,
                verbose=-1,
            ),
            LIGHTGBM_PARAM_DISTRIBUTIONS,
        ),
        # early_stopping=False: the default ("auto") holds out a random 10% of the rows
        # once there are more than 10,000, which is not a time-ordered split.
        "hist_gradient_boost": (
            HistGradientBoostingRegressor(random_state=config.RANDOM_STATE, early_stopping=False),
            HISTGB_PARAM_DISTRIBUTIONS,
        ),
    }


def tuned_search(
    estimator: RegressorMixin,
    distributions: dict,
    log_target: bool,
    cv: BaseCrossValidator | None,
    n_iter: int = SEARCH_ITERATIONS,
) -> RandomizedSearchCV:
    """RandomizedSearchCV over `distributions` (keys relative to `estimator`): fixed budget, seeded.

    The search is part of the estimator, so cross-validating it gives NESTED CV:
    hyperparameters are re-tuned inside every outer training fold.
    """
    prefix = "regressor__" if log_target else ""
    return RandomizedSearchCV(
        wrap_target(estimator, log_target),
        {prefix + key: value for key, value in distributions.items()},
        n_iter=n_iter,
        cv=make_kfold() if cv is None else cv,
        scoring="neg_root_mean_squared_error",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        # A failed fit must stop the run: the default (score NaN) lets the search pick a
        # configuration from NaN scores without saying so.
        error_score="raise",
    )


def get_nonlinear_models(
    numeric: list[str] = config.NUMERIC_FEATURES,
    categorical: list[str] = config.CATEGORICAL_FEATURES,
    log_target: bool = False,
    cv: BaseCrossValidator | None = None,
    names: tuple[str, ...] = ("random_forest", "gradient_boost"),
    n_iter: int = SEARCH_ITERATIONS,
) -> dict[str, RandomizedSearchCV]:
    """Tree models, each wrapped in a light RandomizedSearchCV (nested CV when cross-validated).

    Exactly as RidgeCV/LassoCV re-tune alpha inside every fold for the linear models.
    cv: splitter for the inner search (default: the shared shuffled KFold).
    names: which of tree_regressors() to build (the default is the original pair).
    """
    regressors = tree_regressors()
    searches = {}
    for name in names:
        regressor, distributions = regressors[name]
        # Trees need no scaling, but sharing the preprocessor keeps one code path;
        # scaling is monotonic so it cannot change any split.
        pipe = Pipeline(
            [("preprocess", build_preprocessor(numeric, categorical)), ("model", regressor)]
        )
        searches[name] = tuned_search(
            pipe, {"model__" + k: v for k, v in distributions.items()}, log_target, cv, n_iter
        )
    return searches


def deployable(model: RegressorMixin) -> RegressorMixin:
    """The refitted best estimator of a search (else the model itself), for saving to disk.

    Saving the search wrapper would also pickle its CV splitter, which is not needed
    for prediction and breaks loading when the splitter is defined in a __main__ script.
    A search inside a log-target wrapper (time_tuned) is replaced by its best estimator too.
    """
    model = getattr(model, "best_estimator_", model)  # only fitted searches have it
    inner = getattr(model, "regressor_", None)
    if isinstance(model, TransformedTargetRegressor) and hasattr(inner, "best_estimator_"):
        model = copy.copy(model)  # shallow copy: the caller's fitted model is left as it is
        model.regressor_ = inner.best_estimator_
    return model


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
    """Regularisation strength picked by RidgeCV/LassoCV or a GridSearchCV, else None."""
    estimator = final_estimator(model)
    if hasattr(estimator, "alpha_"):  # RidgeCV / LassoCV
        return estimator.alpha_
    if isinstance(estimator, (Ridge, Lasso)):  # tuned by time_tuned's GridSearchCV
        return estimator.alpha
    return None
