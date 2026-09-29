"""End-to-end Kaggle pipeline. Run from the project root: python -m src.pipeline

Steps: load -> split -> CV (linear + nested non-linear) -> GroupKFold check ->
final fit on the full training split -> ONE evaluation on the held-out test split
-> save tables, figures, models and the (local-only) submission file.
"""

import json
import logging

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.model_selection import GroupKFold

from src import config
from src.data import load_kaggle_competition_test, make_split, split_xy
from src.eda import top_pearson_feature
from src.evaluate import cross_validate_models, holdout_metrics
from src.models import (
    BOOSTING_LIBRARY,
    chosen_alpha,
    deployable,
    get_linear_models,
    get_nonlinear_models,
)
from src.plots import save_figure

logger = logging.getLogger(__name__)

LINEAR_MODEL_NAMES = ["simple_lr", "multiple_lr", "polynomial", "ridge", "lasso"]
TREE_MODEL_NAMES = ["random_forest", "gradient_boost"]


def all_models(simple_feature: str) -> dict[str, RegressorMixin]:
    """Every unfitted model; simple_feature must be chosen on the training split only."""
    return {**get_linear_models(simple_feature), **get_nonlinear_models()}


def run_cv(
    models: dict[str, RegressorMixin], train_df: pd.DataFrame, simple_feature: str
) -> pd.DataFrame:
    """Same 5-fold KFold for every model; saves cv_linear.csv and cv_all_models.csv."""
    X, y = split_xy(train_df)
    linear = {k: v for k, v in models.items() if k not in TREE_MODEL_NAMES}
    trees = {k: v for k, v in models.items() if k in TREE_MODEL_NAMES}
    cv_linear = cross_validate_models(linear, X, y)
    # Searches parallelise internally, so run their outer folds one at a time.
    cv_trees = cross_validate_models(trees, X, y, n_jobs=None)
    cv_all = pd.concat([cv_linear, cv_trees]).sort_values("rmse_mean").reset_index(drop=True)
    for table in (cv_linear, cv_all):
        table["simple_lr_feature"] = table["model"].map({"simple_lr": simple_feature})
    cv_linear.to_csv(config.TABLES_DIR / "cv_linear.csv", index=False)
    cv_all.to_csv(config.TABLES_DIR / "cv_all_models.csv", index=False)
    return cv_all


def best_names(cv_all: pd.DataFrame) -> tuple[str, str, str]:
    """(best linear, best overall, best tree) by mean CV RMSE, chosen before the test set."""
    ranked = cv_all.set_index("model")["rmse_mean"]
    best_linear = ranked[LINEAR_MODEL_NAMES].idxmin()
    best_overall = ranked.drop("dummy_mean").idxmin()
    best_tree = ranked[TREE_MODEL_NAMES].idxmin()
    return best_linear, best_overall, best_tree


def run_group_cv(
    models: dict[str, RegressorMixin], names: list[str], train_df: pd.DataFrame
) -> pd.DataFrame:
    """GroupKFold by region (leave-one-region-out) for the given models (duplicates dropped)."""
    X, y = split_xy(train_df)
    groups = train_df[config.GROUP_CV_COLUMN]
    cv = GroupKFold(n_splits=groups.nunique())
    subset = {name: models[name] for name in dict.fromkeys(names)}
    out = cross_validate_models(subset, X, y, cv=cv, groups=groups, n_jobs=None)
    out.to_csv(config.TABLES_DIR / "cv_groupkfold.csv", index=False)
    return out


def run_final_test(
    models: dict[str, RegressorMixin], train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, RegressorMixin]]:
    """Fit every model on the full training split and evaluate ONCE on the test split."""
    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)
    fitted, rows = {}, []
    for name, model in models.items():
        fitted[name] = clone(model).fit(X_train, y_train)
        rows.append({"model": name, **holdout_metrics(y_test, fitted[name].predict(X_test))})
    results = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    results.to_csv(config.TABLES_DIR / "test_results.csv", index=False)
    return results, fitted


def save_best_params(fitted: dict[str, RegressorMixin]) -> dict:
    """Tuned hyperparameters (trees) and chosen alphas (regularised linear models)."""
    params = {name: fitted[name].best_params_ for name in TREE_MODEL_NAMES}
    params["boosting_library"] = BOOSTING_LIBRARY
    for name in ("polynomial", "ridge", "lasso"):
        params[name] = {"alpha": float(chosen_alpha(fitted[name]))}
    params = json.loads(json.dumps(params, default=float))  # numpy scalars -> plain floats
    (config.TABLES_DIR / "best_params.json").write_text(json.dumps(params, indent=2))
    return params


def plot_results(
    test_results: pd.DataFrame,
    fitted: dict[str, RegressorMixin],
    test_df: pd.DataFrame,
    best_overall: str,
    best_linear: str,
) -> None:
    """Model-comparison bar chart and predicted-vs-actual plots."""
    ordered = test_results.sort_values("rmse", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["grey" if m == "dummy_mean" else "tab:blue" for m in ordered["model"]]
    ax.barh(ordered["model"], ordered["rmse"], color=colors)
    for y_pos, value in enumerate(ordered["rmse"]):
        ax.text(value, y_pos, f" {value:.3f}", va="center", fontsize=9)
    ax.set(xlabel="Test RMSE (t/ha, lower is better)", title="Held-out test RMSE by model")
    save_figure(fig, "model_comparison")
    plt.close(fig)

    X_test, y_test = split_xy(test_df)
    for name, file_name in [
        (best_overall, "pred_vs_actual_best"),
        (best_linear, "pred_vs_actual_linear"),
    ]:
        pred = fitted[name].predict(X_test)
        rmse = test_results.set_index("model").loc[name, "rmse"]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_test, pred, s=8, alpha=0.4)
        lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
        ax.plot(lims, lims, color="red", label="y = x (perfect prediction)")
        ax.set(
            xlabel="Actual yield (t/ha)",
            ylabel="Predicted yield (t/ha)",
            title=f"{name}: test RMSE = {rmse:.3f} t/ha",
        )
        ax.legend()
        save_figure(fig, file_name)
        plt.close(fig)


def save_app_metadata(train_df: pd.DataFrame) -> dict:
    """Slider ranges (training min/max/median) and dropdown levels for the Streamlit app."""
    metadata = {
        "numeric": {
            col: {
                "min": float(train_df[col].min()),
                "max": float(train_df[col].max()),
                "median": float(train_df[col].median()),
                "is_integer": bool(pd.api.types.is_integer_dtype(train_df[col])),
            }
            for col in config.NUMERIC_FEATURES
        },
        "categorical": {
            col: sorted(train_df[col].unique().tolist()) for col in config.CATEGORICAL_FEATURES
        },
    }
    config.APP_METADATA_JSON.write_text(json.dumps(metadata, indent=2))
    return metadata


def write_submission(model: RegressorMixin) -> None:
    """Predict the unlabelled competition test file in sample_submission format (NOT uploaded)."""
    competition_test = load_kaggle_competition_test()
    sample = pd.read_csv(config.KAGGLE_SAMPLE_SUBMISSION_CSV)
    submission = pd.DataFrame(
        {
            config.ID_COLUMN: competition_test[config.ID_COLUMN],
            config.TARGET: model.predict(competition_test[config.FEATURES]),
        }
    )
    if list(submission.columns) != list(sample.columns) or not submission[config.ID_COLUMN].equals(
        sample[config.ID_COLUMN]
    ):
        raise ValueError("Submission columns/ids do not match sample_submission.csv")
    config.SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(config.SUBMISSIONS_DIR / "submission.csv", index=False)


def main() -> None:
    """Run every pipeline step in order."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for folder in (config.TABLES_DIR, config.FIGURES_DIR, config.MODELS_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    train_df, test_df = make_split()
    simple_feature = top_pearson_feature(train_df)  # chosen on the training split only
    logger.info("simple_lr feature (highest |Pearson r| on train): %s", simple_feature)
    models = all_models(simple_feature)

    logger.info("Cross-validating %d models (trees use nested CV)...", len(models))
    cv_all = run_cv(models, train_df, simple_feature)
    print(cv_all.round(4).to_string(index=False))
    best_linear, best_overall, best_tree = best_names(cv_all)
    logger.info(
        "best linear=%s, best overall=%s, best tree=%s", best_linear, best_overall, best_tree
    )

    group_cv = run_group_cv(models, [best_linear, best_overall, best_tree], train_df)
    print("\nGroupKFold by region:\n" + group_cv.round(4).to_string(index=False))

    # --- The ONLY place the Kaggle test split is used. Nothing is changed after this. ---
    test_results, fitted = run_final_test(models, train_df, test_df)
    print("\nTest results:\n" + test_results.round(4).to_string(index=False))

    print("\nBest params:", json.dumps(save_best_params(fitted), indent=2))
    plot_results(test_results, fitted, test_df, best_overall, best_linear)
    joblib.dump(deployable(fitted[best_linear]), config.MODELS_DIR / "best_linear.joblib")
    joblib.dump(deployable(fitted[best_overall]), config.MODELS_DIR / "best_overall.joblib")
    joblib.dump(deployable(fitted[best_tree]), config.MODELS_DIR / "best_tree.joblib")
    (config.MODELS_DIR / "model_names.json").write_text(
        json.dumps(
            {"best_linear": best_linear, "best_overall": best_overall, "best_tree": best_tree},
            indent=2,
        )
    )
    save_app_metadata(train_df)
    write_submission(fitted[best_overall])
    logger.info("Done.")


if __name__ == "__main__":
    main()
