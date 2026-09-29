"""Real-world validation on Indian state-level crop yields, 1997-2020.

Run from the project root: python -m src.india

Leakage and scale fixes (each covered by tests/test_india.py):
  * Production is dropped: Yield = Production / Area, so it would leak the target.
  * Fertilizer and Pesticide are converted to per-hectare values (raw columns are
    totals that scale with farm size).
  * Rows with Area <= 0 or Yield <= 0 are removed; Area enters only as log1p(Area).
  * Split by TIME: train on <= 2015, test on >= 2016. CV folds are also time-ordered.
"""

import json
import logging
from collections.abc import Iterator

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.model_selection import TimeSeriesSplit

from src import config
from src.eda import top_pearson_feature
from src.evaluate import cross_validate_models, holdout_metrics
from src.models import deployable, get_linear_models, get_nonlinear_models
from src.plots import save_figure

logger = logging.getLogger(__name__)

INDIA_CV_FOLDS = 5  # outer time-ordered folds over the 19 training years
INDIA_INNER_CV_FOLDS = 3  # inner (tuning) folds: the earliest outer fold has only 4 years
TREE_MODEL_NAMES = ["random_forest", "gradient_boost"]


class YearBlockSplit:
    """TimeSeriesSplit over whole YEARS: every validation year is later than all training years.

    Years are read from the Crop_Year column of X, so the same splitter works for
    the outer CV and for the inner hyperparameter search. The column is dropped by
    the preprocessor and never reaches a model.
    """

    def __init__(self, n_splits: int) -> None:
        self.n_splits = n_splits

    def split(
        self, X: pd.DataFrame, y=None, groups=None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_rows, validation_rows) with the validation years after the training years."""
        years = X[config.INDIA_YEAR].to_numpy()
        unique_years = np.unique(years)
        for train_idx, val_idx in TimeSeriesSplit(n_splits=self.n_splits).split(unique_years):
            yield (
                np.flatnonzero(np.isin(years, unique_years[train_idx])),
                np.flatnonzero(np.isin(years, unique_years[val_idx])),
            )

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Number of folds (required by scikit-learn)."""
        return self.n_splits


def load_india_raw() -> pd.DataFrame:
    """Read the raw India CSV, with a clear message if it has not been placed yet."""
    if not config.INDIA_CSV.exists():
        raise FileNotFoundError(
            f"{config.INDIA_CSV} not found. Download it from "
            "https://data.mendeley.com/datasets/ncw2vbcgnk and place it in data/raw/india/."
        )
    return pd.read_csv(config.INDIA_CSV)


def clean_india(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Strip text, drop leakage/invalid rows, build per-hectare and log-area features.

    Returns the cleaned frame (features + Crop_Year + Yield only) and a log of
    how many rows each step removed.
    """
    df = raw.copy()
    text_columns = df.select_dtypes(include=["object", "string"]).columns
    df[text_columns] = df[text_columns].apply(lambda s: s.str.strip())  # e.g. "Kharif     "
    removed = {"rows_raw": len(df)}

    invalid = (df[config.INDIA_AREA] <= 0) | (df[config.INDIA_TARGET] <= 0)
    removed["area_or_yield_le_0"] = int(invalid.sum())
    df = df[~invalid]

    excluded = df[config.INDIA_CROP].isin(config.INDIA_EXCLUDED_CROPS)
    removed["excluded_crops"] = int(excluded.sum())
    df = df[~excluded]

    before = len(df)
    df = df.drop_duplicates()
    removed["exact_duplicates"] = before - len(df)

    df = df.assign(
        **{
            config.INDIA_FERTILIZER_PER_HA: df[config.INDIA_FERTILIZER] / df[config.INDIA_AREA],
            config.INDIA_PESTICIDE_PER_HA: df[config.INDIA_PESTICIDE] / df[config.INDIA_AREA],
            # Area spans 0.5 to 5e7 ha; log1p tames that range, and area stays useful
            # as a proxy for how established a crop is in a state.
            config.INDIA_LOG_AREA: np.log1p(df[config.INDIA_AREA]),
        }
    )
    keep = config.INDIA_FEATURES + [config.INDIA_YEAR, config.INDIA_TARGET]
    df = df[keep].reset_index(drop=True)  # Production, raw totals and raw Area are dropped here
    removed["rows_clean"] = len(df)
    for step, n in removed.items():
        logger.info("India cleaning - %s: %d", step, n)
    return df, removed


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on years <= INDIA_TRAIN_LAST_YEAR, test on years >= INDIA_TEST_FIRST_YEAR."""
    train = df[df[config.INDIA_YEAR] <= config.INDIA_TRAIN_LAST_YEAR]
    test = df[df[config.INDIA_YEAR] >= config.INDIA_TEST_FIRST_YEAR]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def make_india_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, clean and time-split the India data; save both parts to parquet."""
    clean, _ = clean_india(load_india_raw())
    train, test = time_split(clean)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(config.INDIA_TRAIN_PARQUET)
    test.to_parquet(config.INDIA_TEST_PARQUET)
    return train, test


def india_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Features plus Crop_Year (for the time splitter only; the preprocessor drops it) and target."""
    return df[config.INDIA_FEATURES + [config.INDIA_YEAR]], df[config.INDIA_TARGET]


def india_models(
    train: pd.DataFrame, numeric: list[str] = config.INDIA_NUMERIC_FEATURES
) -> dict[str, RegressorMixin]:
    """The same model suite as Kaggle, on a log1p target, with time-ordered inner CV.

    numeric: the numeric inputs to use (the ablation passes a reduced list). The
    preprocessor selects only these columns, so any other columns in X are ignored.
    """
    # Chosen on the TRAINING years, against the target on its modelling (log) scale.
    simple_feature = top_pearson_feature(train, numeric, np.log1p(train[config.INDIA_TARGET]))
    logger.info("India simple_lr feature (numeric inputs %s): %s", numeric, simple_feature)
    linear = get_linear_models(
        simple_feature,
        numeric,
        config.INDIA_CATEGORICAL_FEATURES,
        log_target=True,  # yield spans 0.01 to >1,000 across 54 crops (skew >> 1)
    )
    trees = get_nonlinear_models(
        numeric,
        config.INDIA_CATEGORICAL_FEATURES,
        log_target=True,
        cv=YearBlockSplit(INDIA_INNER_CV_FOLDS),
    )
    return {**linear, **trees}


def fit_and_test(
    models: dict[str, RegressorMixin], train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, RegressorMixin]]:
    """Fit every model on the training years and evaluate ONCE on the test years."""
    X_train, y_train = india_xy(train)
    X_test, y_test = india_xy(test)
    fitted, rows = {}, []
    for name, model in models.items():
        fitted[name] = clone(model).fit(X_train, y_train)
        rows.append({"model": name, **holdout_metrics(y_test, fitted[name].predict(X_test))})
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True), fitted


def feature_ranges(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Training vs test range of each column, and the share of test rows outside it."""
    rows = []
    for col in columns:
        lo, hi = train[col].min(), train[col].max()
        rows.append(
            {
                "feature": col,
                "train_min": lo,
                "train_max": hi,
                "test_min": test[col].min(),
                "test_max": test[col].max(),
                "share_test_rows_above_train_max": (test[col] > hi).mean(),
                "share_test_rows_below_train_min": (test[col] < lo).mean(),
            }
        )
    return pd.DataFrame(rows)


def ablation_table(with_cols: pd.DataFrame, without_cols: pd.DataFrame) -> pd.DataFrame:
    """Test RMSE/R^2 with vs without the year-index columns (delta_rmse > 0: they help)."""
    merged = with_cols[["model", "rmse", "r2"]].merge(
        without_cols[["model", "rmse", "r2"]], on="model", suffixes=("_with", "_without")
    )
    merged["delta_rmse"] = merged["rmse_without"] - merged["rmse_with"]
    merged["delta_r2"] = merged["r2_without"] - merged["r2_with"]
    return merged.sort_values("rmse_with").reset_index(drop=True)


def lookup_baseline(train: pd.DataFrame, test: pd.DataFrame, keys: list[str]) -> np.ndarray:
    """Reference 'no-ML' prediction: training-years geometric-mean yield per group of keys.

    E.g. keys=["Crop", "State"] predicts each test row with the average log-yield of
    the same crop in the same state during 1997-2015. Groups never seen in training
    fall back to the overall training mean. Uses TRAINING years only.
    """
    log_yield = np.log1p(train[config.INDIA_TARGET])
    group_means = log_yield.groupby([train[k] for k in keys]).mean()
    test_keys = pd.MultiIndex.from_frame(test[keys]) if len(keys) > 1 else pd.Index(test[keys[0]])
    pred_log = group_means.reindex(test_keys).fillna(log_yield.mean()).to_numpy()
    return np.expm1(pred_log)


def per_crop_rmse(test: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """Test RMSE for the most common crops, relative to each crop's mean yield.

    The median absolute error is added because a few physically impossible yields
    in the source data (e.g. sesamum > 30 t/ha) can dominate a crop's RMSE.
    """
    frame = test.assign(pred=pred)
    top = frame[config.INDIA_CROP].value_counts().index[: config.INDIA_TOP_CROPS_FOR_REPORT]
    rows = []
    for crop in top:
        d = frame[frame[config.INDIA_CROP] == crop]
        m = holdout_metrics(d[config.INDIA_TARGET], d["pred"])
        rows.append(
            {
                "crop": crop,
                "n_test_rows": len(d),
                "mean_yield": d[config.INDIA_TARGET].mean(),
                "rmse": m["rmse"],
                "relative_rmse": m["rmse"] / d[config.INDIA_TARGET].mean(),
                "r2": m["r2"],
                "median_abs_error": (d[config.INDIA_TARGET] - d["pred"]).abs().median(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Run the full India validation: CV on training years, one test on 2016-2020."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for folder in (config.TABLES_DIR, config.FIGURES_DIR, config.MODELS_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    train, test = make_india_split()
    logger.info(
        "train years %d-%d (%d rows), test years %d-%d (%d rows)",
        train[config.INDIA_YEAR].min(),
        train[config.INDIA_YEAR].max(),
        len(train),
        test[config.INDIA_YEAR].min(),
        test[config.INDIA_YEAR].max(),
        len(test),
    )
    models = india_models(train)
    X_train, y_train = india_xy(train)
    outer_cv = YearBlockSplit(INDIA_CV_FOLDS)
    linear = {k: v for k, v in models.items() if k not in TREE_MODEL_NAMES}
    trees = {k: v for k, v in models.items() if k in TREE_MODEL_NAMES}
    cv = pd.concat(
        [
            cross_validate_models(linear, X_train, y_train, cv=outer_cv),
            cross_validate_models(trees, X_train, y_train, cv=outer_cv, n_jobs=None),
        ]
    )
    cv = cv.sort_values("rmse_mean").reset_index(drop=True)
    cv.to_csv(config.TABLES_DIR / "india_cv.csv", index=False)
    print(cv.round(4).to_string(index=False))
    ranked = cv.set_index("model")["rmse_mean"]
    best_overall = ranked.drop("dummy_mean").idxmin()
    best_tree = ranked[TREE_MODEL_NAMES].idxmin()

    # --- The ONLY place the India test years are used (main suite + ablation). ---
    X_test, _ = india_xy(test)
    test_results, fitted = fit_and_test(models, train, test)
    test_results.to_csv(config.TABLES_DIR / "india_test.csv", index=False)
    print("\nIndia test (2016-2020):\n" + test_results.round(4).to_string(index=False))

    # Ablation (CODE_REVIEW H1): identical suite, seeds and split, minus the two
    # year-index columns. It is only compared, never used to pick or change a model.
    reduced = [
        c for c in config.INDIA_NUMERIC_FEATURES if c not in config.INDIA_YEAR_INDEX_FEATURES
    ]
    ablation_results, _ = fit_and_test(india_models(train, reduced), train, test)
    ablation = ablation_table(test_results, ablation_results)
    ablation.to_csv(config.TABLES_DIR / "india_ablation.csv", index=False)
    print("\nAblation (without year-index columns):\n" + ablation.round(4).to_string(index=False))
    ranges = feature_ranges(train, test, config.INDIA_YEAR_INDEX_FEATURES)
    ranges.to_csv(config.TABLES_DIR / "india_feature_ranges.csv", index=False)
    print("\nTrain vs test ranges:\n" + ranges.round(4).to_string(index=False))

    crops = per_crop_rmse(test, fitted[best_overall].predict(X_test))
    crops.to_csv(config.TABLES_DIR / "india_per_crop_rmse.csv", index=False)
    print(f"\nPer-crop test RMSE for {best_overall}:\n" + crops.round(3).to_string(index=False))

    params = {name: fitted[name].best_params_ for name in TREE_MODEL_NAMES}
    params = json.loads(json.dumps(params, default=float))
    (config.TABLES_DIR / "india_best_params.json").write_text(json.dumps(params, indent=2))
    joblib.dump(deployable(fitted[best_overall]), config.MODELS_DIR / "india_best_overall.joblib")
    joblib.dump(deployable(fitted[best_tree]), config.MODELS_DIR / "india_best_tree.joblib")
    (config.MODELS_DIR / "india_model_names.json").write_text(
        json.dumps({"best_overall": best_overall, "best_tree": best_tree}, indent=2)
    )

    ordered = test_results.sort_values("rmse", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["grey" if m == "dummy_mean" else "tab:orange" for m in ordered["model"]]
    ax.barh(ordered["model"], ordered["rmse"], color=colors)
    for y_pos, value in enumerate(ordered["rmse"]):
        ax.text(value, y_pos, f" {value:.2f}", va="center", fontsize=9)
    ax.set(
        xlabel="Test RMSE, 2016-2020 (yield units, mostly t/ha; lower is better)",
        title="India: held-out test RMSE by model",
    )
    save_figure(fig, "india_model_comparison")
    plt.close(fig)
    logger.info("Done. best overall=%s, best tree=%s", best_overall, best_tree)


if __name__ == "__main__":
    main()
