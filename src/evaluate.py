"""Cross-validation and test-set metrics, always in the target's original units.

Kaggle: t/ha. India: the units reported per crop (mostly t/ha), never log units.
"""

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_predict, cross_validate

from src.data import make_kfold
from src.models import chosen_alpha

SCORING = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}


def cross_validate_models(
    models: dict[str, RegressorMixin],
    X: pd.DataFrame,
    y: pd.Series,
    cv: Any = None,  # any scikit-learn splitter, incl. the duck-typed YearBlockSplit
    groups: pd.Series | None = None,
    n_jobs: int | None = -1,
) -> pd.DataFrame:
    """Mean and std of RMSE, MAE and R^2 across folds, plus fit time and chosen alpha.

    Scoring uses model.predict, which already returns original units when the
    target is log-transformed (TransformedTargetRegressor inverts it).
    Pass n_jobs=None for models that parallelise internally (e.g. a search), so
    the outer folds run one at a time and the CPU is not oversubscribed.
    """
    cv = cv if cv is not None else make_kfold()
    rows = []
    for name, model in models.items():
        res = cross_validate(
            model, X, y, cv=cv, groups=groups, scoring=SCORING, return_estimator=True, n_jobs=n_jobs
        )
        alphas = [chosen_alpha(est) for est in res["estimator"]]
        row = {"model": name}
        for metric in SCORING:
            scores = res[f"test_{metric}"]
            scores = -scores if metric in ("rmse", "mae") else scores  # sklearn negates errors
            row[f"{metric}_mean"] = scores.mean()
            row[f"{metric}_std"] = scores.std()
        row["fit_time_s"] = res["fit_time"].mean()
        row["alpha_median"] = np.median(alphas) if alphas[0] is not None else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rmse_mean").reset_index(drop=True)


def leaderboard(cv: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Test results ranked by mean CV RMSE, the criterion the deployed model was chosen by.

    The test columns are kept as they are; the CV mean and std are added alongside.
    """
    board = cv[["model", "rmse_mean", "rmse_std"]].merge(test, on="model", how="inner")
    return board.sort_values("rmse_mean", kind="stable").reset_index(drop=True)


def holdout_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE, MAE and R^2 for a single held-out evaluation."""
    return {
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


CONFORMAL_ALPHA = 0.10  # 90% prediction interval


def conformal_quantile(residuals: np.ndarray, alpha: float = CONFORMAL_ALPHA) -> float:
    """Split-conformal half-width: the ceil((n + 1)(1 - alpha))-th smallest |residual|.

    With exchangeable data, prediction ± this value covers a new true value with
    probability at least 1 - alpha (finite-sample guarantee).
    """
    scores = np.sort(np.abs(np.asarray(residuals, dtype=float)))
    n = len(scores)
    rank = min(math.ceil((n + 1) * (1 - alpha)), n)
    return float(scores[rank - 1])


def conformal_interval(
    model: RegressorMixin,
    fitted: RegressorMixin,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    alpha: float = CONFORMAL_ALPHA,
) -> dict:
    """90% interval from the out-of-fold CV residuals of `model`, checked once on the test set.

    The calibration residuals come from the same 5 folds as the model comparison
    (each training field predicted by a model that never saw it); `fitted` is the
    deployed model refitted on all training rows, as used by the app.
    """
    oof = cross_val_predict(model, X_train, y_train, cv=make_kfold(), n_jobs=-1)
    q = conformal_quantile(y_train.to_numpy() - oof, alpha)
    miss = np.abs(y_test.to_numpy() - fitted.predict(X_test))
    return {
        "method": "split-conformal on out-of-fold 5-fold CV residuals",
        "level": 1 - alpha,
        "n_calibration": len(y_train),
        "quantile": q,
        "interval_width": 2 * q,
        "n_test": len(y_test),
        "test_coverage": float(np.mean(miss <= q)),
    }
