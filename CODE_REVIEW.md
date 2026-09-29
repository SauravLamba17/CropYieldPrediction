# Code review: Crop Yield Prediction

**Scope.** Every file was re-read in full: `src/` (config, data, features, models, evaluate,
eda, explain, plots, pipeline, india), `tests/` (4 files), `app.py`, and all four notebooks.
The review was done from the perspective of a strict ML engineer and a sceptical examiner.

**Result: 0 CRITICAL · 1 HIGH · 4 MEDIUM · 12 LOW.**

- **No leakage was found.**
- **No finding changes any model or any reported number.** In the final run, after the fixes, two
  pipeline runs produced byte-identical `test_results.csv` files (hashes in
  `CROP_YIELD_BUILD_REPORT.md` §7). Every Kaggle and India number equals the pre-fix runs. The H1
  ablation *adds* results (`india_ablation.csv`) but changes none of the existing ones.

---

## 1. Findings

Each finding lists its severity, location, the problem, the fix, and its status.

### HIGH

| # | Where | Problem | Fix | Status |
|---|---|---|---|---|
| H1 | `notebooks/04_india.ipynb` §2b (last paragraph); comment in `tests/test_india.py::test_production_is_not_a_feature`; config comment on `INDIA_YEAR` | The claim that "`Crop_Year` is used only for splitting — it is never a feature" is literally true but misleading. `Fertilizer_per_ha` and `Pesticide_per_ha` take exactly one value per year, and fertilizer's value is unique for every year, so the model effectively *does* see the year through them. An examiner would rightly object. This is not leakage (the year is known when predicting), but the text overstated the isolation. It also explains why the trees cannot extrapolate to 2019–2020. | Two parts. (1) **Documentation:** reworded §2b, the config comment and the test comment to say that the year enters implicitly through the per-hectare index, and why that is legitimate but limits interpretation. (2) **Measurement (ablation, added after your review):** `python -m src.india` re-runs the identical suite, with the same split, seeds and time-ordered tuning, without `Fertilizer_per_ha` and `Pesticide_per_ha`. Results are in `results/tables/india_ablation.csv` and `india_feature_ranges.csv`, notebook 04 §8b, and 2 new tests. **Outcome:** the columns help the linear family modestly (RMSE +0.27 to +0.31 without them, ≈ 3 %), do essentially nothing for the trees (random forest +0.016, XGBoost +0.019, ≈ 0.2 %), and slightly hurt the degree-3 polynomial (−0.047). 100 % of test rows have pesticide above the training maximum, and 26 % have fertilizer above it, so the trees cannot use them. The best model, the ranking and the crop × state look-up result are unchanged either way. | FIXED (documented + measured by ablation) |

### MEDIUM

| # | Where | Problem | Fix | Status |
|---|---|---|---|---|
| M1 | `src/pipeline.py:34` `GROUP_COLUMN = "region"` | A column name is hardcoded outside `config.py`, which breaks the project's own rule. | Moved to `config.GROUP_CV_COLUMN`. | FIXED |
| M2 | `src/config.py:108` (`"Season"`, `"State"` as bare literals); notebook 04 §2a (`"Coconut"`) and §8 (`"State"`); `tests/test_india.py` toy data | Column and level names are repeated as string literals instead of named constants. | Added `config.INDIA_SEASON` and `config.INDIA_STATE`, and used them plus `config.INDIA_EXCLUDED_CROPS` everywhere. | FIXED |
| M3 | `notebooks/02_diagnostics.ipynb` §4 (Lasso interpretation) | Wrong reasoning: the cell said the CV alpha is tiny "because there is almost no noise to shrink away". The residual noise is actually large (σ ≈ 0.64 t/ha). The real reason is low *variance*: 3,840 rows, 24 terms and uncorrelated features mean the unpenalised fit barely overfits, so shrinkage has little to fix. | Reworded. | FIXED |
| M4 | `src/models.py` `get_linear_models` for India: `LassoCV(cv=5)` and `RidgeCV` (leave-one-out) | For India, alpha for Ridge/Lasso/polynomial is chosen with folds that are not time-ordered. The inner estimators receive only the transformed matrix, which has no `Crop_Year`, so they cannot use `YearBlockSplit`. Future training years can therefore influence the choice of alpha for earlier ones. The outer CV and the 2016–2020 test are unaffected, since the test years are never seen. | Not fixed. It would need scikit-learn metadata routing of year groups into the inner estimators, which is too much machinery for a student codebase. The effect is small: the tuned alphas are 0.37 / 0.0002. | FOLLOW-UP (documented as a limitation) |

### LOW

| # | Where | Problem | Fix | Status |
|---|---|---|---|---|
| L1 | `src/models.py:11` | `BaseSearchCV` is imported from the private module `sklearn.model_selection._search`, which could break on a scikit-learn upgrade. | Replaced with a public `hasattr(model, "best_estimator_")` check. | FIXED |
| L2 | `src/models.py` `deployable` / `fitted_pipeline` | Duplicated "unwrap the search" logic. | `fitted_pipeline` now calls `deployable`. | FIXED |
| L3 | `src/india.py:34` | `LINEAR_MODEL_NAMES` is defined but never used (dead code). | Removed. | FIXED |
| L4 | `src/pipeline.py` (`all_models`, `run_cv`) | `top_pearson_feature(train_df)` is computed twice. | Computed once in `main`, then passed in. | FIXED |
| L5 | `src/evaluate.py:1, 19` | The module docstring says metrics are "always t/ha", but the India units are only *mostly* t/ha. The `cv` type hint excludes the duck-typed `YearBlockSplit`. | Docstring now says "original target units"; the hint is widened to `Any` splitter with a comment. | FIXED |
| L6 | `notebooks/01_eda.ipynb` §10 | This notebook calls phosphorus "U-shaped", while notebooks 02 and 03 later show the effect is negligible (a ±0.03 t/ha threshold). The notebooks are inconsistent. | Added a forward note in notebook 01. | FIXED |
| L7 | `notebooks/03_explain.ipynb` summary | "Yield … is driven mainly by fertilizer" reads as causal. | Changed to "is most strongly associated with". | FIXED |
| L8 | `notebooks/02_diagnostics.ipynb` §3 | Breusch-Pagan is run on *out-of-fold* residuals (the spec placed it in that section), whereas the textbook test uses in-sample OLS residuals. | Added a note: the in-sample version gives the same conclusion (LM = 21.2, p = 0.45 vs p = 0.41 out-of-fold). | FIXED |
| L9 | `src/explain.py` `RawFeatureExplainer.explain` | It reseeds NumPy's *global* RNG on every call (Kernel SHAP draws from it). That is harmless in the app, but it is a side effect. | Documented in the docstring. A private generator is impossible because shap uses the global one. | DOCUMENTED |
| L10 | India `dummy_mean` | With a log target, `DummyRegressor(strategy="mean")` predicts expm1(mean log1p y), a geometric-style mean, not the arithmetic mean. | Documented in the report. Every model still beats it. | DOCUMENTED |
| L11 | Process | The Kaggle and India test splits were *evaluated* more than once, because of the reproducibility reruns and the thread-count fix. No model, hyperparameter or selection was changed in response to a test number. | Documented in the report (Deviations), including the one number that moved (XGBoost 0.6459 → 0.6461, thread-count arithmetic). | DOCUMENTED |
| L12 | `app.py` `load_artifacts() -> tuple`; `src/pipeline.py` `write_submission` uses `assert` | The return type is loose. The asserts are skipped under `python -O`. | Tightened to `tuple[RegressorMixin, dict, dict, RawFeatureExplainer]`. The asserts became explicit `ValueError`s. | FIXED |

### Checklist results (Phase 11 A–E)

**A. Correctness and leakage.**
- **Scalers, encoders and polynomial expansions** are inside Pipelines. The data flow was traced for all 8 (Kaggle) and 9 (India) models, and `features.build_preprocessor` is only ever used inside `make_pipeline` / `Pipeline`. There is no imputer, because there were no missing values.
- **The test split is never used for fitting, tuning or feature selection.** The `simple_lr` feature is chosen with `top_pearson_feature(train_df)`. Tuning happens inside the training folds (nested CV). The test split is read only in `run_final_test`, in notebook 03 (for permutation importance and SHAP, as the spec requires), and in the app's static tables.
- **Metrics are in original units everywhere.** There is no transform for Kaggle, and India uses `TransformedTargetRegressor`, whose `predict` inverts the log. A test asserts that predictions are on the t/ha scale.
- **Permutation importance and SHAP use the right model.** Permutation importance uses the saved `best_overall` pipeline on raw columns. SHAP uses `best_tree` / `india_best_overall`, with names from `get_feature_names_out()` mapped back by `original_feature`.
- **India: no path from Production or from the test years into training.** Production is dropped in `clean_india` and a test covers it. `max(train year) < min(test year)` is tested, and `YearBlockSplit` is tested for time order. See H1 for how the year enters implicitly.

**B. Statistical soundness.**
- VIF and OLS use exactly `multiple_lr`'s preprocessor (standardised numeric features, drop-first one-hot).
- The Breusch-Pagan conclusion matches its p-value (0.41, so homoscedastic).
- The claims in the markdown cells were checked against the printed outputs. Wrong ones were fixed during the build (e.g. "≈ 0.8" → 0.73, the skewness values, the phosphorus shape) and in this review (M3, L6).
- The only causal wording was L7, now fixed.

**C. Reproducibility.**
- Every random process uses `RANDOM_STATE`: the split, KFold, LassoCV, the searches, RF, XGBoost, permutation importance, the SHAP samples, and Kernel SHAP (via the global seed).
- There are no absolute paths; a grep of `src`, `tests`, `app.py` and the notebooks returned nothing.
- `requirements-lock.txt` is identical to the current `pip freeze`.

**D. Code quality.**
- Hardcoded column names: M1 and M2 fixed.
- Dead code: L3 fixed.
- Docstrings and type hints are present on every public function.
- The notebooks import from `src/` and don't reimplement it.
- Magic numbers are named constants.
- Error messages clearly name the missing data file or model and the command that creates it.

**E. Consistency between artifacts.**
- The tables are written by the same code the notebooks read back (e.g. notebook 04 reads `india_*.csv`).
- Figure units were checked; one wrong unit (pesticide "kg/ha" on the India figure) was fixed during the build.
- Model names are identical across tables, figures, `model_names.json` and the app.

---

## 2. Likely viva questions, with answers grounded in the results

1. **"Your Kaggle data is synthetic. Why should we believe anything?"**
   Phase 1 found strong evidence: every feature is uniform (kurtosis ≈ −1.2), features that
   should be related are uncorrelated (max |r| = 0.028), and crop type has no effect (ANOVA
   p = 0.71). So the Kaggle part is framed as a *controlled benchmark*, where we can check that
   the methods recover a known structure. The real-world claims come only from the India data
   (RQ4), and that is where the conclusions changed.

2. **"How do you know there's no leakage?"**
   - Every preprocessing step is inside a Pipeline, so it is refitted in each fold. A test
     checks the scaler's mean equals the training sample's mean.
   - The test split is used once, in `run_final_test`, and `simple_lr`'s feature is chosen on
     train only.
   - The best R² is 0.69, far below the 0.98 red flag.
   - For India: Production (r = 0.95 with Yield) is dropped and tested, the split is by time
     (tested), and CV is time-ordered.

3. **"Why Ridge on the polynomial features rather than plain OLS?"**
   A degree-2 expansion of 11 features creates 77 highly correlated terms (x and x²). OLS on
   them has high variance. RidgeCV picked α ≈ 45, a strong shrinkage, and gave the best CV RMSE
   (0.635).

4. **"Why didn't Random Forest beat linear regression on Kaggle?"**
   The relationship is mostly linear: fertilizer alone gives R² = 0.59, and there are no
   interactions (F-test p = 0.81 and 0.50). RF approximates a straight line with steps, which
   loses accuracy (test RMSE 0.663 vs 0.653 for OLS). XGBoost with depth-2 trees and a
   polynomial Ridge both capture the small curves, and they tie at 0.646.

5. **"Is the relationship linear? (RQ3)"**
   Mostly. Pearson ≈ Spearman everywhere, but quadratic tests (Bonferroni-corrected) and SHAP
   dependence plots show:
   - diminishing returns for fertilizer,
   - saturation for rainfall above about 600 mm,
   - a flat-topped pH optimum between about 6 and 7.5.

   These effects are only 0.1–0.3 t/ha, against fertilizer's roughly 3 t/ha range. So flexible
   models gain only about 0.007 t/ha RMSE.

6. **"Why do SHAP and the coefficients disagree about soil pH?"**
   pH has an optimum. A straight line through an inverted-U has a slope of about 0 (coefficient
   rank 13), while SHAP and permutation see the curve (ranks 4 and 8). The disagreement is
   evidence of non-linearity, not an error.

7. **"Why a time split for India but a random split for Kaggle?"**
   The Kaggle rows are exchangeable, with no time structure (harvest month is flat). The India
   data is a panel. A random split would let the model learn 2018 conditions from other 2018
   rows, so the test mimics real use: train on the past (≤ 2015), predict the future (≥ 2016).

8. **"Why log-transform the India target but not the Kaggle one?"**
   The skewness rule. Kaggle skewness is −0.10, and log1p would make it −0.52. India skewness
   is 21.5 and falls to 1.95 after log1p. Metrics are still reported in original units, because
   `TransformedTargetRegressor` inverts predictions before scoring.

9. **"Do the findings transfer to real data? (RQ4)"**
   Largely no.
   - On India, crop type accounts for about 83 % of SHAP importance, versus about 2 % for all
     context on Kaggle.
   - Trees clearly win (8.12 vs 9.48 RMSE).
   - A crop×state look-up table (RMSE 7.81) beats every model, so the models mainly learn yield
     levels.
   - Fertilizer can't be tested, because India's fertilizer column is a national per-year index.

   What does transfer is the methodology: baselines, honest splits and leakage checks.

10. **"Your per-crop R² is negative. Isn't the model useless?"**
    Within a crop the features (state rainfall, national indices) carry little signal, so yes:
    beyond crop and state levels the model adds little. Part of the per-crop RMSE also comes
    from 82 impossible yields in the source (e.g. maize at 990 t/ha). The median absolute error
    is 15–40 % of the mean yield, which gives a fairer picture than RMSE.

11. **Extra: "Why is Crop_Year not a feature? Doesn't the model then ignore time?"**
    Not quite (see H1). `Fertilizer_per_ha` is a one-to-one encoding of the year, so the models
    see time implicitly. The ablation measures how much that matters. Removing both per-hectare
    columns costs the linear models about 3 % RMSE (they extrapolate the trend) and the trees
    only about 0.2 %. For the trees, every test row's pesticide value, and 26 % of fertilizer
    values, lie above the training maximum, so they fall into the last split anyway. None of the
    headline conclusions depend on these columns.
