"""Paired statistical tie check on per-fold CV scores (Phase 5).

The pipelines saved only the mean and std of each CV metric, so this module re-runs ONLY the
compared models' CV with the same code, seeds and folds, saves the per-fold scores
(cv_folds.csv), and first checks that their mean and std equal the committed tables. Nothing
else is written; no model, table or selection changes.

    python -m src.tie_check        about 25 min on a laptop (the two India tree searches)

Test: for each pair, d_k = RMSE_A(fold k) - RMSE_B(fold k). Reported: mean and sample std of
d, a two-sided paired t-test and an exact two-sided Wilcoxon signed-rank test. With 5 folds the
smallest possible Wilcoxon p-value is 2 / 2^5 = 0.0625, so it can never reach 0.05; the
verdict uses the t-test (p < 0.05). CV folds share training rows, so both tests are
optimistic: "not distinguishable" is the robust conclusion, "distinguishable" a weak one.
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.model_selection import cross_validate

from src import config

logger = logging.getLogger(__name__)

ALPHA = 0.05
FOLDS_CSV = config.TABLES_DIR / "cv_folds.csv"
TIE_CSV = config.TABLES_DIR / "tie_check.csv"

# (dataset, model A, model B): A is the deployed / selected model.
KAGGLE_PAIRS = [("polynomial", "gradient_boost"), ("polynomial", "hist_gradient_boost")]
INDIA_SELECTED = "random_forest + all"
INDIA_PAIRS = [
    (INDIA_SELECTED, "last year's yield (no ML)"),
    (INDIA_SELECTED, "crop x state lookup (no ML)"),
    (INDIA_SELECTED, "hist_gradient_boost + all"),
]


def paired_test(a: np.ndarray, b: np.ndarray, alpha: float = ALPHA) -> dict:
    """Paired comparison of two per-fold score arrays (same folds, same order)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    t_p = float(ttest_rel(a, b).pvalue) if np.std(d) > 0 else (1.0 if d.mean() == 0 else 0.0)
    w_p = float(wilcoxon(a, b, method="exact").pvalue) if np.any(d != 0) else 1.0
    return {
        "n_folds": len(d),
        "mean_diff": float(d.mean()),
        "sd_diff": float(d.std(ddof=1)),
        "t_test_p": t_p,
        "wilcoxon_p": w_p,
        "distinguishable": bool(t_p < alpha),
    }


def model_folds(dataset: str, name: str, model, X, y, cv, n_jobs) -> pd.DataFrame:
    """Per-fold RMSE / MAE / R^2 of one model, exactly as src.evaluate scores it."""
    from src.evaluate import SCORING

    res = cross_validate(model, X, y, cv=cv, scoring=SCORING, n_jobs=n_jobs)
    logger.info("%s %-30s fit %.0f s per fold", dataset, name, res["fit_time"].mean())
    return pd.DataFrame(
        {
            "dataset": dataset,
            "model": name,
            "fold": range(1, len(res["test_rmse"]) + 1),
            "rmse": -res["test_rmse"],
            "mae": -res["test_mae"],
            "r2": res["test_r2"],
        }
    )


def kaggle_folds() -> pd.DataFrame:
    from src.data import load_split, make_kfold, split_xy
    from src.eda import top_pearson_feature
    from src.models import get_linear_models, get_nonlinear_models
    from src.pipeline import EXTRA_MODEL_NAMES

    train, _ = load_split()
    X, y = split_xy(train)
    poly = get_linear_models(top_pearson_feature(train))["polynomial"]
    trees = {**get_nonlinear_models(), **get_nonlinear_models(names=EXTRA_MODEL_NAMES)}
    frames = [model_folds("kaggle", "polynomial", poly, X, y, make_kfold(), n_jobs=-1)]
    for name in ("gradient_boost", "hist_gradient_boost"):  # searches parallelise internally
        frames.append(model_folds("kaggle", name, trees[name], X, y, make_kfold(), n_jobs=None))
    return pd.concat(frames, ignore_index=True)


def india_folds() -> pd.DataFrame:
    from src.evaluate import holdout_metrics
    from src.india import INDIA_CV_FOLDS, YearBlockSplit
    from src.india_analysis import REFERENCES, TARGET, model_inputs
    from src.india_models import india_candidate

    train, test = pd.read_parquet(config.INDIA_TRAIN_PARQUET), pd.read_parquet(config.INDIA_TEST_PARQUET)
    X, _, _ = model_inputs(train, test)
    y = train[TARGET]
    cv = YearBlockSplit(INDIA_CV_FOLDS)
    rows = []
    for name in ("crop x state lookup (no ML)", "last year's yield (no ML)"):  # as reference_cv
        for k, (fit_idx, val_idx) in enumerate(cv.split(X), start=1):
            pred = REFERENCES[name](X.iloc[fit_idx], y.iloc[fit_idx], X.iloc[val_idx])
            rows.append({"dataset": "india", "model": name, "fold": k,
                         **holdout_metrics(y.iloc[val_idx], pred)})
    frames = [pd.DataFrame(rows)]
    for family in ("random_forest", "hist_gradient_boost"):  # as candidate_cv
        name = f"{family} + all"
        frames.append(model_folds("india", name, india_candidate(family, "all", False), X, y, cv,
                                  n_jobs=None))
    return pd.concat(frames, ignore_index=True)


def check_against_committed(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean and std (ddof=0, as the pipelines) of the per-fold RMSE vs the committed tables."""
    committed = pd.concat(
        [
            pd.read_csv(config.TABLES_DIR / "cv_all_models.csv").assign(dataset="kaggle"),
            pd.read_csv(config.TABLES_DIR / "kaggle_extra_models.csv").assign(dataset="kaggle"),
            pd.read_csv(config.TABLES_DIR / "india_leaderboard.csv").assign(dataset="india"),
        ]
    ).drop_duplicates(["dataset", "model"]).set_index(["dataset", "model"])
    got = folds.groupby(["dataset", "model"])["rmse"].agg(
        rmse_mean="mean", rmse_std=lambda s: float(np.std(s))
    )
    diff = (got - committed.loc[got.index, ["rmse_mean", "rmse_std"]]).abs()
    return diff.rename(columns=lambda c: f"abs_diff_{c}")


def tie_table(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, pairs in (("kaggle", KAGGLE_PAIRS), ("india", INDIA_PAIRS)):
        sub = folds[folds["dataset"] == dataset]
        for a, b in pairs:
            ra = sub[sub["model"] == a].sort_values("fold")["rmse"].to_numpy()
            rb = sub[sub["model"] == b].sort_values("fold")["rmse"].to_numpy()
            rows.append({"dataset": dataset, "model_a": a, "model_b": b,
                         "cv_rmse_a": ra.mean(), "cv_rmse_b": rb.mean(), **paired_test(ra, rb)})
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    folds = pd.concat([kaggle_folds(), india_folds()], ignore_index=True)
    diff = check_against_committed(folds)
    print("Per-fold scores vs committed CV tables (absolute difference):\n" + diff.to_string())
    if diff.to_numpy().max() > 1e-9:
        raise SystemExit("Per-fold scores do not reproduce the committed CV tables; nothing saved.")
    folds.to_csv(FOLDS_CSV, index=False)
    table = tie_table(folds)
    table.to_csv(TIE_CSV, index=False)
    print("\nPaired tie check (RMSE, A - B):\n" + table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
