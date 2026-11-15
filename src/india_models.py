"""Phase 4 India candidates: tree models on time-safe features, and a look-up + residual hybrid.

Every candidate is a RandomizedSearchCV with the same fixed budget (INDIA_SEARCH_ITERATIONS
configurations, seeded) on the same time-ordered year-block folds, around a log1p-target
pipeline. Cross-validating a candidate therefore re-tunes it inside every outer fold.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src import config
from src.features import build_preprocessor
from src.india import YearBlockSplit
from src.india_features import (
    FEATURE_SETS,
    PAIR,
    TE_COLUMNS,
    YEAR,
    TimeSafeTargetEncoder,
    lookup_log,
)
from src.models import tree_regressors, tuned_search

INDIA_SEARCH_ITERATIONS = 20  # same budget as the original India RF / XGBoost searches
INNER_CV_FOLDS = 3  # as src.india.INDIA_INNER_CV_FOLDS
FAMILIES = ["random_forest", "gradient_boost", "lightgbm", "hist_gradient_boost"]
SMOOTHING_GRID = [1, 3, 10, 30, 100]  # pseudo-rows for the target encoding, tuned by CV


class FillEmptyColumns(BaseEstimator, TransformerMixin):
    """Set columns that are entirely missing at fit time to 0 (other missing values are kept).

    In the earliest inner fold (training years = 1997 only) every encoding and lag is NaN,
    and HistGradientBoosting's binning raises on an all-NaN column. A constant column offers
    no split either, so the model is unchanged; the other families handle it natively.
    """

    def fit(self, X, y=None) -> "FillEmptyColumns":
        self.empty_ = np.isnan(np.asarray(X, dtype=float)).all(axis=0)
        return self

    def transform(self, X) -> np.ndarray:
        out = np.array(X, dtype=float)
        out[:, self.empty_] = 0.0
        return out


class LookupResidualRegressor(BaseEstimator, RegressorMixin):
    """Hybrid: the crop x state look-up plus a model of what the look-up gets wrong.

    Works on the log scale (wrap it in the log1p target). The look-up for a row of year t
    uses only fitted rows of earlier years (src.india_features.lookup_log), so the training
    residuals are honest out-of-time residuals and, for the test years, the offset is
    exactly the crop x state baseline. Rows with no earlier year at all (only the first year
    of a fit; in the earliest inner CV fold that is every row) use the mean of the fitted
    targets instead, the same fallback as at prediction time.
    """

    def __init__(self, estimator: RegressorMixin) -> None:
        self.estimator = estimator

    def fit(self, X: pd.DataFrame, y) -> "LookupResidualRegressor":
        """Fit the residual model on y - look-up (both on the log scale)."""
        self.history_ = X[PAIR + [YEAR]].reset_index(drop=True)
        self.target_ = np.asarray(y, dtype=float)
        self.estimator_ = clone(self.estimator).fit(X, self.target_ - self.offset(X))
        return self

    def offset(self, X: pd.DataFrame) -> np.ndarray:
        """The look-up prediction (log scale) from fitted rows of earlier years, else the fitted mean."""
        return np.nan_to_num(lookup_log(self.history_, self.target_, X), nan=float(self.target_.mean()))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Look-up + predicted residual."""
        return self.offset(X) + self.estimator_.predict(X)


def india_candidate(
    family: str, feature_set: str, hybrid: bool = False, n_iter: int = INDIA_SEARCH_ITERATIONS
) -> RandomizedSearchCV:
    """One tuned candidate: tree `family` on the base inputs plus FEATURE_SETS[feature_set]."""
    extra = FEATURE_SETS[feature_set]
    regressor, distributions = tree_regressors()[family]
    steps, space = [], {"model__" + k: v for k, v in distributions.items()}
    if any(c in TE_COLUMNS for c in extra):
        steps.append(("encode", TimeSafeTargetEncoder()))
        space["encode__smoothing"] = SMOOTHING_GRID
    numeric = config.INDIA_NUMERIC_FEATURES + extra
    steps.append(("preprocess", build_preprocessor(numeric, config.INDIA_CATEGORICAL_FEATURES)))
    if family == "hist_gradient_boost":
        steps.append(("fill_empty", FillEmptyColumns()))
    steps.append(("model", regressor))
    estimator = Pipeline(steps)
    if hybrid:
        estimator = LookupResidualRegressor(estimator)
        space = {"estimator__" + k: v for k, v in space.items()}
    return tuned_search(estimator, space, log_target=True, cv=YearBlockSplit(INNER_CV_FOLDS), n_iter=n_iter)


def candidate_name(family: str, feature_set: str, hybrid: bool = False) -> str:
    """e.g. 'lightgbm + all' or 'hybrid: lookup + lightgbm (all)'."""
    return f"hybrid: lookup + {family} ({feature_set})" if hybrid else f"{family} + {feature_set}"
