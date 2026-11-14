"""Global explainability tables for the app, computed once in the pipeline.

Run inside python -m src.pipeline (after the final test), or on its own from the saved
models: python -m src.explain_tables

Writes to results/tables/:
  explain_permutation.csv   permutation importance per ORIGINAL feature (test set, 30 repeats)
  explain_shap.csv          mean |SHAP| per original feature (test set)
  explain_coefficients.csv  standardised coefficients with 95% bootstrap intervals
  conformal_groups.csv      empirical coverage of the 90% interval per crop and per region
Nothing here changes a model, a split or an existing metric.
"""

import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.inspection import permutation_importance

from src import config
from src.data import load_split, split_xy
from src.explain import RawFeatureExplainer, standardized_coefficients, tree_shap

logger = logging.getLogger(__name__)

PERMUTATION_REPEATS = 30
BOOTSTRAP_RESAMPLES = 500
COEFFICIENT_MODELS = ["multiple_lr", "ridge", "lasso"]
COVERAGE_GROUPS = ["crop_type", "region"]


def permutation_original(
    model: RegressorMixin, X: pd.DataFrame, y: pd.Series, n_repeats: int = PERMUTATION_REPEATS
) -> pd.DataFrame:
    """Increase in test RMSE when each ORIGINAL input column is shuffled.

    The whole pipeline sits inside `model`, so shuffling the raw `crop_type` column moves
    all of its one-hot columns together, and shuffling a numeric input moves every
    polynomial term built from it: the importance is per original feature by construction.
    """
    result = permutation_importance(
        model,
        X,
        y,
        scoring="neg_root_mean_squared_error",
        n_repeats=n_repeats,
        random_state=config.RANDOM_STATE,
        n_jobs=None,
    )
    table = pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def shap_original_kernel(model: RegressorMixin, background: pd.DataFrame, X: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per original feature with the app's Kernel SHAP explainer (any model)."""
    explanation = RawFeatureExplainer(model, background).explain(X)
    return pd.Series(np.abs(explanation.values).mean(axis=0), index=explanation.feature_names)


def shap_original_tree(model: RegressorMixin, X: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per original feature for a tree model (one-hot terms summed back)."""
    _, _, by_feature = tree_shap(model, X, config.FEATURES, sample_size=len(X))
    return by_feature.abs().mean()


def bootstrap_coefficients(
    model: RegressorMixin,
    X: pd.DataFrame,
    y: pd.Series,
    n_boot: int = BOOTSTRAP_RESAMPLES,
    seed: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Standardised coefficients (full-data fit) with 95% percentile bootstrap intervals.

    Each resample draws len(X) training rows with replacement and refits the whole
    pipeline (scaler, encoder and, for Ridge/Lasso, the alpha search), so the interval
    includes the uncertainty of those steps too.
    """
    full = standardized_coefficients(clone(model).fit(X, y), config.FEATURES)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        rows = rng.integers(0, len(X), len(X))
        fitted = clone(model).fit(X.iloc[rows], y.iloc[rows])
        draws.append(standardized_coefficients(fitted, config.FEATURES).set_index("term")["coef"])
    samples = pd.DataFrame(draws).reindex(columns=full["term"])
    full["ci_low"] = samples.quantile(0.025).to_numpy()
    full["ci_high"] = samples.quantile(0.975).to_numpy()
    full["crosses_zero"] = (full["ci_low"] <= 0) & (full["ci_high"] >= 0)
    full["n_boot"] = n_boot
    return full.drop(columns="abs_coef")


def group_coverage(covered: pd.Series, groups: dict[str, pd.Series]) -> pd.DataFrame:
    """Share of rows inside the interval per level of each grouping column.

    covered: boolean per row (|real − predicted| <= half-width).
    groups: {group name: level per row}, aligned with `covered`.
    """
    rows = []
    for name, levels in groups.items():
        for level, hit in covered.groupby(levels.to_numpy()):
            rows.append({"group_by": name, "group": level, "n": len(hit), "coverage": float(hit.mean())})
    return pd.DataFrame(rows)


def save_explainability(
    deployed: RegressorMixin,
    deployed_name: str,
    tree: RegressorMixin,
    tree_name: str,
    linear_models: dict[str, RegressorMixin],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Compute and save every table listed in the module docstring."""
    logging.getLogger("shap").setLevel(logging.WARNING)  # it logs every Kernel SHAP row at INFO
    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)

    perm = pd.concat(
        [
            permutation_original(deployed, X_test, y_test).assign(model=deployed_name),
            permutation_original(tree, X_test, y_test).assign(model=tree_name),
        ]
    )
    perm["n_repeats"] = PERMUTATION_REPEATS
    perm[["model", "feature", "importance_mean", "importance_std", "n_repeats"]].to_csv(
        config.TABLES_DIR / "explain_permutation.csv", index=False
    )
    logger.info("Permutation importance saved (%d repeats, both models)", PERMUTATION_REPEATS)

    shap_rows = []
    for name, values in [
        (deployed_name, shap_original_kernel(deployed, X_train, X_test)),
        (tree_name, shap_original_tree(tree, X_test)),
    ]:
        shap_rows += [{"model": name, "feature": f, "mean_abs_shap": float(v)} for f, v in values.items()]
    pd.DataFrame(shap_rows).sort_values(["model", "mean_abs_shap"], ascending=[True, False]).to_csv(
        config.TABLES_DIR / "explain_shap.csv", index=False
    )
    logger.info("Mean |SHAP| saved (Kernel SHAP for %s, TreeExplainer for %s)", deployed_name, tree_name)

    coefficients = pd.concat(
        [
            bootstrap_coefficients(linear_models[name], X_train, y_train).assign(model=name)
            for name in COEFFICIENT_MODELS
        ]
    )
    columns = ["model", "term", "feature", "label", "coef", "ci_low", "ci_high", "crosses_zero", "n_boot"]
    coefficients[columns].to_csv(config.TABLES_DIR / "explain_coefficients.csv", index=False)
    logger.info("Bootstrap coefficients saved (%d resamples)", BOOTSTRAP_RESAMPLES)

    interval = json.loads((config.TABLES_DIR / "conformal.json").read_text())
    covered = pd.Series(np.abs(y_test.to_numpy() - deployed.predict(X_test)) <= interval["quantile"])
    coverage = group_coverage(covered, {g: test_df[g].reset_index(drop=True) for g in COVERAGE_GROUPS})
    coverage.to_csv(config.TABLES_DIR / "conformal_groups.csv", index=False)
    logger.info("Per-group conformal coverage saved")


def main() -> None:
    """Rebuild the tables from the saved models (same inputs as the pipeline step)."""
    from src.eda import top_pearson_feature
    from src.pipeline import all_models

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    train_df, test_df = load_split()
    names = json.loads((config.MODELS_DIR / "model_names.json").read_text())
    models = all_models(top_pearson_feature(train_df))
    save_explainability(
        joblib.load(config.MODELS_DIR / "best_overall.joblib"),
        names["best_overall"],
        joblib.load(config.MODELS_DIR / "best_tree.joblib"),
        names["best_tree"],
        models,
        train_df,
        test_df,
    )


if __name__ == "__main__":
    main()
