"""Phase 4 India analysis: can time-safe features and boosted models beat the crop x state look-up?

Called by `python -m src.india` after the original model suite. Steps:
  1. inputs: lag features for the training years (history = training years) and the test
     years (history = every earlier year, i.e. the earlier test years too: the forecasting
     assumption in src/india_features.py). A strict variant with only 1997-2015 history is
     also scored, as a sensitivity check.
  2. time-ordered CV (the same 5 year-block folds as the original suite) of the look-up and
     "last year's yield" references, of 4 tree families on all features, of an ablation of
     feature sets for the best family, and of a look-up + residual hybrid.
  3. selection by CV RMSE only; then ONE evaluation of every candidate on 2016-2020.
  4. error analysis, permutation importance and mean |SHAP| for the selected model.
Nothing here changes the original suite's tables (india_cv.csv, india_test.csv).
"""

import json
import logging
import time

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_validate

from src import config
from src.evaluate import SCORING, holdout_metrics
from src.explain import original_feature
from src.explain_tables import PERMUTATION_REPEATS, permutation_original
from src.india_features import FEATURE_SETS, LAG_COLUMNS, PAIR, TARGET, TE_COLUMNS, YEAR, add_lags
from src.india_models import (
    FAMILIES,
    INDIA_SEARCH_ITERATIONS,
    LookupResidualRegressor,
    candidate_name,
    india_candidate,
)
from src.models import deployable

logger = logging.getLogger(__name__)

INPUT_COLUMNS = config.INDIA_FEATURES + [YEAR] + LAG_COLUMNS
LOOKUP = "crop x state lookup (no ML)"
PERSISTENCE = "last year's yield (no ML)"
CROP_LOOKUP = "crop lookup (no ML)"
ORIGINAL_SELECTED = "random_forest"  # kept as the deployed model if nothing beats the look-up
SHAP_ROWS = 1000  # test rows for TreeExplainer (random, seeded)


def model_inputs(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(X_train, X_test, X_test_strict) with the lag columns; each lag uses earlier years only.

    X_test: one-step-ahead, history = training years + earlier test years (the forecasting
    assumption). X_test_strict: history = training years only (forecast origin 2015).
    """
    X_train = add_lags(train, train)[INPUT_COLUMNS]
    X_test = add_lags(test, pd.concat([train, test]))[INPUT_COLUMNS]
    X_strict = add_lags(test, train)[INPUT_COLUMNS]
    return X_train, X_test, X_strict


def lookup_prediction(fit_X: pd.DataFrame, fit_y: pd.Series, rows: pd.DataFrame, keys: list[str]) -> np.ndarray:
    """src.india.lookup_baseline on (fit_X, fit_y): training-rows mean log-yield per group."""
    from src.india import lookup_baseline

    return lookup_baseline(fit_X.assign(**{TARGET: fit_y.to_numpy()}), rows, keys)


def persistence_prediction(fit_X: pd.DataFrame, fit_y: pd.Series, rows: pd.DataFrame) -> np.ndarray:
    """Last year's yield of the same crop x state; the crop x state look-up when there is none."""
    last = np.expm1(rows["lag1_yield"].to_numpy())
    return np.where(np.isnan(last), lookup_prediction(fit_X, fit_y, rows, PAIR), last)


REFERENCES = {
    LOOKUP: lambda fx, fy, rows: lookup_prediction(fx, fy, rows, PAIR),
    CROP_LOOKUP: lambda fx, fy, rows: lookup_prediction(fx, fy, rows, [config.INDIA_CROP]),
    PERSISTENCE: persistence_prediction,
}


def summarise(name: str, folds: dict[str, np.ndarray], fit_time: float, **extra) -> dict:
    """One CV row: mean and std of each metric over the folds."""
    row = {"model": name}
    for metric, scores in folds.items():
        row[f"{metric}_mean"], row[f"{metric}_std"] = float(np.mean(scores)), float(np.std(scores))
    return {**row, "fit_time_s": fit_time, **extra}


def reference_cv(X: pd.DataFrame, y: pd.Series, cv) -> list[dict]:
    """CV rows for the no-ML references, on the same folds as the models."""
    rows = []
    for name, predict in REFERENCES.items():
        folds = {m: [] for m in SCORING}
        for fit_idx, val_idx in cv.split(X):
            pred = predict(X.iloc[fit_idx], y.iloc[fit_idx], X.iloc[val_idx])
            for metric, value in holdout_metrics(y.iloc[val_idx], pred).items():
                folds[metric].append(value)
        rows.append(summarise(name, folds, 0.0, kind="reference", feature_set="", family=""))
    return rows


def candidate_cv(model, name: str, X: pd.DataFrame, y: pd.Series, cv, **extra) -> dict:
    """Nested time-ordered CV of one candidate (its search re-tunes inside every outer fold)."""
    start = time.perf_counter()
    res = cross_validate(model, X, y, cv=cv, scoring=SCORING, n_jobs=None)
    folds = {m: (-res[f"test_{m}"] if m in ("rmse", "mae") else res[f"test_{m}"]) for m in SCORING}
    row = summarise(name, folds, float(res["fit_time"].mean()), kind="model", **extra)
    logger.info("CV %-45s RMSE %.4f +/- %.4f (%.0f s)", name, row["rmse_mean"], row["rmse_std"],
                time.perf_counter() - start)
    return row


def test_row(name: str, y_test: pd.Series, pred: np.ndarray, pred_strict: np.ndarray, train_rmse: float) -> dict:
    """Test metrics (one-step-ahead) plus the strict fixed-origin RMSE."""
    return {
        "model": name,
        **{f"test_{k}": v for k, v in holdout_metrics(y_test, pred).items()},
        "test_rmse_strict": root_mean_squared_error(y_test, pred_strict),
        "train_rmse": train_rmse,
    }


def error_by_group(test: pd.DataFrame, preds: dict[str, np.ndarray], column: str) -> pd.DataFrame:
    """Test RMSE per level of `column` for every predictor in `preds` (model first)."""
    frame = test.assign(**{f"pred_{k}": v for k, v in preds.items()})
    rows = []
    for level, d in frame.groupby(column):
        row = {"group_by": column, "group": level, "n_test_rows": len(d), "mean_yield": d[TARGET].mean()}
        for key in preds:
            row[f"rmse_{key}"] = root_mean_squared_error(d[TARGET], d[f"pred_{key}"])
        rows.append(row)
    out = pd.DataFrame(rows)
    out["model_beats_lookup"] = out["rmse_model"] < out["rmse_lookup"]
    return out.sort_values("rmse_model", ascending=False).reset_index(drop=True)


def error_segments(test: pd.DataFrame, X_test: pd.DataFrame, X_train: pd.DataFrame,
                   preds: dict[str, np.ndarray]) -> pd.DataFrame:
    """Test RMSE on rows with/without lag history and on unseen crops / crop x state pairs."""
    seen_pairs = pd.MultiIndex.from_frame(X_train[PAIR].drop_duplicates())
    pair_seen = pd.MultiIndex.from_frame(test[PAIR]).isin(seen_pairs)
    masks = {
        "all test rows": np.ones(len(test), dtype=bool),
        "with lag history": X_test["history_years"].to_numpy() > 0,
        "no lag history (crop x state never grown before)": X_test["history_years"].to_numpy() == 0,
        "no record last year (lag1 missing)": X_test["lag1_yield"].isna().to_numpy(),
        "crop x state not in training years": ~pair_seen,
        "crop not in training years": ~test[config.INDIA_CROP].isin(X_train[config.INDIA_CROP]).to_numpy(),
    }
    rows = []
    for segment, mask in masks.items():
        row = {"segment": segment, "n_test_rows": int(mask.sum())}
        for key, pred in preds.items():
            row[f"rmse_{key}"] = root_mean_squared_error(test[TARGET][mask], pred[mask]) if mask.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def raw_feature(term: str, features: list[str]) -> list[tuple[str, float]]:
    """Original input(s) behind a model input, with weights (target encodings map to Crop / State)."""
    feature = original_feature(term, features)
    return {
        "te_crop": [(config.INDIA_CROP, 1.0)],
        "te_state": [(config.INDIA_STATE, 1.0)],
        # Convention: an encoding of the pair is credited half to crop and half to state.
        "te_crop_state": [(config.INDIA_CROP, 0.5), (config.INDIA_STATE, 0.5)],
    }.get(feature, [(feature, 1.0)])


def shap_by_input(model, X: pd.DataFrame, inputs: list[str]) -> pd.Series:
    """Mean |SHAP| (log1p units) per original input column for a fitted tree candidate.

    For the hybrid only the residual model is explained (the look-up offset is not a tree).
    One-hot and encoding terms are summed back to their input row by row (SHAP is additive).
    """
    inner = deployable(model).regressor_
    if isinstance(inner, LookupResidualRegressor):
        inner = inner.estimator_
    rows = X.sample(min(SHAP_ROWS, len(X)), random_state=config.RANDOM_STATE)
    model_rows = inner[:-1].transform(rows)
    names = inner.named_steps["preprocess"].get_feature_names_out()
    values = shap.TreeExplainer(inner[-1]).shap_values(np.asarray(model_rows, dtype=float))
    terms = config.INDIA_FEATURES + TE_COLUMNS + LAG_COLUMNS + [YEAR]
    by_input = pd.DataFrame(0.0, index=rows.index, columns=inputs)
    for j, name in enumerate(names):
        for feature, weight in raw_feature(name, terms):
            by_input[feature] += weight * values[:, j]
    return by_input.abs().mean().sort_values(ascending=False)


def used_inputs(feature_set: str) -> list[str]:
    """Raw input columns a candidate reads (target encodings are built from Crop/State/Year)."""
    extra = [c for c in FEATURE_SETS[feature_set] if c not in TE_COLUMNS]
    return config.INDIA_FEATURES + extra


def run(train: pd.DataFrame, test: pd.DataFrame, base_cv: pd.DataFrame, base_test: pd.DataFrame,
        base_fitted: dict, base_params: dict) -> dict:
    """Every step in the module docstring; returns the selection summary (also saved to JSON)."""
    from src.india import INDIA_CV_FOLDS, YearBlockSplit, india_xy, per_crop_rmse

    timings, start = {}, time.perf_counter()
    X_train, X_test, X_strict = model_inputs(train, test)
    y_train, y_test = train[TARGET], test[TARGET]
    outer = YearBlockSplit(INDIA_CV_FOLDS)
    cv_rows = reference_cv(X_train, y_train, outer)

    # --- 1. four tree families on all features (CV only) ---
    specs = {candidate_name(f, "all"): (f, "all", False) for f in FAMILIES}
    for name, (family, fset, hybrid) in specs.items():
        cv_rows.append(candidate_cv(india_candidate(family, fset, hybrid), name, X_train, y_train, outer,
                                    family=family, feature_set=fset))
    timings["cv_families_s"] = time.perf_counter() - start
    family_cv = pd.DataFrame(cv_rows).query("kind == 'model'").set_index("family")["rmse_mean"]
    best_family = family_cv.idxmin()
    logger.info("Best family on all features (CV): %s", best_family)

    # --- 2. ablation of the best family + the hybrid (CV only) ---
    t0 = time.perf_counter()
    in_original_suite = best_family in base_cv["model"].values
    for fset in [s for s in FEATURE_SETS if s != "all"]:
        if fset == "base" and in_original_suite:
            continue  # the identical model (same inputs, space, budget, folds) is already scored
        specs[candidate_name(best_family, fset)] = (best_family, fset, False)
    specs[candidate_name(best_family, "all", hybrid=True)] = (best_family, "all", True)
    for name, (family, fset, hybrid) in list(specs.items())[len(FAMILIES):]:
        cv_rows.append(candidate_cv(india_candidate(family, fset, hybrid), name, X_train, y_train, outer,
                                    family=family, feature_set=f"hybrid ({fset})" if hybrid else fset))
    timings["cv_ablation_s"] = time.perf_counter() - t0
    cv_new = pd.DataFrame(cv_rows)

    # --- 3. selection by CV RMSE only, BEFORE any test score is computed ---
    base = base_cv.assign(kind="model", family="", feature_set="original suite")
    board_cv = pd.concat([base[cv_new.columns.intersection(base.columns)], cv_new], ignore_index=True)
    ml = board_cv.query("kind == 'model' and model != 'dummy_mean'").set_index("model")["rmse_mean"]
    lookup_cv = board_cv.set_index("model").loc[LOOKUP, "rmse_mean"]
    best_ml = ml.idxmin()
    selected = best_ml if ml[best_ml] < lookup_cv else ORIGINAL_SELECTED
    logger.info("Selected by CV: %s (CV %.4f; look-up CV %.4f)", selected, ml[selected], lookup_cv)

    # --- 4. the ONLY place the test years are used: every candidate once ---
    t0 = time.perf_counter()
    fitted, test_rows, params = {}, [], {}
    for name, (family, fset, hybrid) in specs.items():
        fitted[name] = india_candidate(family, fset, hybrid).fit(X_train, y_train)
        params[name] = json.loads(json.dumps(fitted[name].best_params_, default=float))
        test_rows.append(test_row(name, y_test, fitted[name].predict(X_test), fitted[name].predict(X_strict),
                                  root_mean_squared_error(y_train, fitted[name].predict(X_train))))
    timings["final_fit_and_test_s"] = time.perf_counter() - t0
    for name, predict in REFERENCES.items():
        pred, strict = predict(X_train, y_train, X_test), predict(X_train, y_train, X_strict)
        test_rows.append(test_row(name, y_test, pred, strict, np.nan))
    base_rows = base_test.rename(columns={"rmse": "test_rmse", "mae": "test_mae", "r2": "test_r2"})
    base_rows["test_rmse_strict"] = base_rows["test_rmse"]  # no lag inputs: both variants are equal
    tests = pd.concat([base_rows, pd.DataFrame(test_rows)], ignore_index=True)
    board = board_cv.merge(tests, on="model", how="left").sort_values("rmse_mean", kind="stable")
    board["selected"] = board["model"] == selected
    board.reset_index(drop=True).to_csv(config.TABLES_DIR / "india_leaderboard.csv", index=False)

    ablation = board[(board["family"] == best_family) | (board["model"].isin([LOOKUP, PERSISTENCE]))].copy()
    if in_original_suite:
        ablation = pd.concat([board[board["model"] == best_family].assign(feature_set="base"), ablation])
    order = {s: i for i, s in enumerate([*FEATURE_SETS, "hybrid (all)", ""])}
    ablation = ablation.assign(_o=ablation["feature_set"].map(order)).sort_values(["_o", "model"]).drop(columns="_o")
    ablation.to_csv(config.TABLES_DIR / "india_feature_ablation.csv", index=False)

    # --- 5. error analysis, selected model vs the references (test years) ---
    if selected in fitted:
        model, X_sel, X_sel_strict = fitted[selected], X_test, X_strict
    else:
        model = base_fitted[selected]
        X_sel = X_sel_strict = india_xy(test)[0]
    pred_model = model.predict(X_sel)
    preds = {
        "model": pred_model,
        "lookup": REFERENCES[LOOKUP](X_train, y_train, X_test),
        "persistence": REFERENCES[PERSISTENCE](X_train, y_train, X_test),
    }
    by_group = pd.concat([error_by_group(test, preds, c) for c in (config.INDIA_CROP, config.INDIA_STATE)])
    by_group.to_csv(config.TABLES_DIR / "india_error_by_group.csv", index=False)
    segments = error_segments(test, X_test, X_train, {**preds, "model_strict": model.predict(X_sel_strict)})
    segments.to_csv(config.TABLES_DIR / "india_error_segments.csv", index=False)
    per_crop_rmse(test, pred_model).to_csv(config.TABLES_DIR / "india_per_crop_rmse.csv", index=False)
    reference = pd.DataFrame(
        [{"predictor": name, **holdout_metrics(y_test, REFERENCES[name](X_train, y_train, X_test))}
         for name in (CROP_LOOKUP, LOOKUP, PERSISTENCE)]
        + [{"predictor": f"selected model ({selected})", **holdout_metrics(y_test, pred_model)}]
    )
    reference.to_csv(config.TABLES_DIR / "india_reference_baselines.csv", index=False)

    # --- 6. explainability of the selected model (test years) ---
    t0 = time.perf_counter()
    feature_set = specs[selected][1] if selected in specs else "base"
    inputs = used_inputs(feature_set)
    columns = list(dict.fromkeys(inputs + [YEAR]))  # the encoders and the hybrid read the year
    perm = permutation_original(model, X_sel[columns], y_test)
    perm = perm[perm["feature"].isin(inputs)].assign(model=selected, n_repeats=PERMUTATION_REPEATS)
    perm[["model", "feature", "importance_mean", "importance_std", "n_repeats"]].to_csv(
        config.TABLES_DIR / "india_explain_permutation.csv", index=False)
    mean_shap = shap_by_input(model, X_sel, inputs)
    shap_table = pd.DataFrame({"model": selected, "feature": mean_shap.index, "mean_abs_shap": mean_shap.to_numpy(),
                               "share": (mean_shap / mean_shap.sum()).to_numpy(),
                               "n_rows": min(SHAP_ROWS, len(X_sel)),
                               "explains": "residual model" if "hybrid" in selected else "model"})
    shap_table.to_csv(config.TABLES_DIR / "india_explain_shap.csv", index=False)
    shap_table.set_index("feature")[["mean_abs_shap", "share"]].to_csv(
        config.TABLES_DIR / "india_shap_shares.csv", index_label="feature")
    scores = pd.DataFrame({"permutation": perm.set_index("feature")["importance_mean"], "shap": mean_shap})
    rho = float(spearmanr(scores["permutation"], scores["shap"]).statistic)
    timings["explainability_s"] = time.perf_counter() - t0

    # --- 7. the deployed India model and the summary ---
    path = config.MODELS_DIR / "india_best_overall.joblib"
    joblib.dump(deployable(model), path)
    tree_names = ["random_forest", "gradient_boost"]
    best_tree = selected if selected in specs else base_cv.set_index("model").loc[tree_names, "rmse_mean"].idxmin()
    (config.MODELS_DIR / "india_model_names.json").write_text(json.dumps(
        {"best_overall": selected, "best_tree": best_tree, "feature_set": feature_set,
         "input_columns": INPUT_COLUMNS if selected in specs else config.INDIA_FEATURES + [YEAR]}, indent=2))
    all_params = {**base_params, **params}
    (config.TABLES_DIR / "india_best_params.json").write_text(json.dumps(all_params, indent=2))
    summary = {
        "selected": selected,
        "best_ml_by_cv": best_ml,
        "best_ml_cv_rmse": float(ml[best_ml]),
        "lookup_cv_rmse": float(lookup_cv),
        "beats_lookup_cv": bool(ml[best_ml] < lookup_cv),
        "best_family_all_features": best_family,
        "search_iterations": INDIA_SEARCH_ITERATIONS,
        "outer_folds": INDIA_CV_FOLDS,
        "permutation_vs_shap_spearman": rho,
        "model_file_mb": round(path.stat().st_size / 1e6, 2),
        "forecasting_assumption": "predicting year t, actual yields up to t-1 are known",
        "timings_s": {k: round(v, 1) for k, v in timings.items()},
    }
    (config.TABLES_DIR / "india_selection.json").write_text(json.dumps(summary, indent=2))
    print("\nIndia leaderboard (ranked by time-ordered CV RMSE):\n"
          + board[["model", "rmse_mean", "rmse_std", "test_rmse", "test_rmse_strict", "test_r2"]].round(4).to_string(index=False))
    print("\nSelection:", json.dumps(summary, indent=2))
    return summary

