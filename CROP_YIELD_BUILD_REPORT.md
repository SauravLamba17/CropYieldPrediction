# Crop Yield Prediction: Build Report

The project predicts crop yield (t/ha) from soil, weather and farming variables. It compares
simple and regularised linear models with non-linear models, explains which variables drive the
predictions, and validates the conclusions on real Indian government data.

Every number in this report comes from code that ran in this build. The tables are in
`results/tables/`, the figures in `results/figures/`, and the command output in §7.
**Nothing has been committed or pushed, and nothing was submitted to Kaggle.**

---

## 1. Executive summary

**What was built.**
- A reproducible pipeline with a single entry point: `python -m src.pipeline`. It runs
  8 models on a Kaggle benchmark, with a locked 20 % test split and nested CV for the tree models.
- A real-data validation on 19,412 Indian state × crop × year records: `python -m src.india`.
  It uses a time split, time-ordered CV and a log target.
- Four executed notebooks: EDA, diagnostics, explainability and India.
- A Streamlit app with a per-prediction SHAP waterfall.
- 24 tests (the final run printed `24 passed`).
- A code review (`CODE_REVIEW.md`).

**Framing.** Phase 1 found the Kaggle data **likely synthetic** (§2.1). By agreement, it is
treated as a *controlled benchmark*, and the India data is the real-world test. The research
questions are therefore read as follows:

| RQ | As investigated here |
|---|---|
| RQ1 | How accurately can yield be predicted on the benchmark, and on real data? |
| RQ2 | Which variables drive the *benchmark's* yield, and do four importance methods agree? |
| RQ3 | Is the benchmark relationship linear, and does the answer change on real data? |
| RQ4 | Do the benchmark conclusions transfer to real Indian yields (1997–2020)? |

**RQ1: accuracy.**
- **Benchmark:** the best models (polynomial Ridge and XGBoost, tied) reach a **test RMSE of
  0.646 t/ha and R² ≈ 0.69**. The mean-only baseline gets 1.160 t/ha (the target's std), so the
  best models cut the error by 44 %. The residuals are Gaussian and homoscedastic with
  σ ≈ 0.64 t/ha, which looks like the generator's own noise floor.
- **India:** the best model (random forest) reaches **test RMSE 8.12 (R² 0.65)** on 2016–2020.
  The dummy gets 14.04.

**RQ2: drivers.**
- On the benchmark, **fertilizer** is the dominant driver under all four methods (standardised
  coefficient, Lasso, permutation, SHAP). **Pesticide (negative)** and **rainfall** follow;
  shuffling fertilizer alone raises test RMSE by 0.73 t/ha.
- By group, farming practices carry about **64 %** of mean |SHAP|, weather 17 %, soil 16 % and
  crop/region/season 2 %.
- The methods agree on the top 3. They disagree on **soil pH**: rank 13 by coefficient, rank 4
  by SHAP. Because pH has an optimum, a straight line through it has almost no slope.

**RQ3: linearity.**
- **Mostly linear, with mild curvature:** diminishing returns for fertilizer, saturation for
  rainfall above about 600 mm, and a flat-topped pH optimum between about 6 and 7.5.
  There are **no interactions** (F-test p = 0.81 and 0.50).
- Consequences: degree-2 polynomial ≈ XGBoost (0.646), which beat linear models by only
  **0.007 t/ha**, while random forest was worst of the flexible models (0.663).
- **On real data the answer changes.** Trees clearly win (8.12 vs 9.48 for the best linear
  model), because real yields depend on crop × state combinations, an interaction that one-hot
  linear models cannot represent.

**RQ4: transfer.**
- **Largely no.** On real data, **crop type accounts for about 83 % of SHAP importance**
  (context overall about 90 %), versus about 2 % on the benchmark.
- A **crop × state look-up table (RMSE 7.81) beats every model**, so the models mostly learn
  yield *levels*, not agronomy.
- Fertilizer's role **cannot be tested** on the India data. Its Fertilizer and Pesticide
  columns are Area × one national per-hectare rate per year (year explains 100 % of their
  variance). An ablation without them (§4.6) shows they add only a small extrapolated time
  trend for the linear models (≈ 3 % RMSE) and almost nothing for the trees (≈ 0.2 %). None of
  the conclusions depend on them.
- What transfers is the method: leakage checks, honest splits, baselines and per-group
  evaluation.

---

## 2. Data

### 2.1 Kaggle benchmark (primary)

- **Source:** Kaggle competition *crop-yield-prediction-challenge*
  (https://www.kaggle.com/competitions/crop-yield-prediction-challenge), CC BY 4.0.
- **Files:**
  - `crop_yield_train.csv`: 4,800 × 18
  - `crop_yield_test.csv`: 1,200 × 17, unlabelled
  - `sample_submission.csv`
- **Target:** `yield_tpha`, in t/ha (confirmed on the data page).
  - min 2.34, max 9.54, mean 6.27, median 6.30, std 1.15, skewness −0.09
  - 0 values ≤ 0
- **Features:** 11 numeric (5 soil, 3 weather, 3 farming) and 3 categorical (crop, region,
  season). All 14 expected features are present.
- **Missing values:** 0. **Duplicates:** 0.
- **Cleaning:** dropped `id`, plus two undocumented columns shown to be random.
  - `harvest_date`: yield is flat by month (ANOVA p = 0.56).
  - `field_id`: explains no more variance than random labels (η² 0.135 vs 0.130,
    permutation p = 0.26). Also, 99.8 % of the competition test file's field IDs appear in
    train, so keeping it would invite memorisation.
- **Split:** random 80/20 split, seed 42: train 3,840, test 960. The competition test file is
  used only to write `submissions/submission.csv`, which was **not uploaded**.

**Synthetic-data verdict: likely synthetic.** Evidence *for* synthetic:
- Every numeric feature is uniform (kurtosis ≈ −1.2; the uniform-distribution test is not
  rejected).
- Features that should correlate are unrelated: rainfall vs soil moisture r = 0.006,
  irrigation vs moisture r = −0.02, and the largest correlation between any two features
  is 0.028.
- Crop, region and season have no effect on yield (ANOVA p = 0.71, 0.24, 0.53), whereas real
  corn out-yields soybean by about 3×.
- Round ranges (pH 4.5–8.5, rainfall 100–1,200 mm), and zero feature outliers.

Evidence *against* (weak):
- Every value is physically plausible.
- The directions of the effects are sensible.
- The Kaggle page says nothing about where the data came from.

**What this means for RQ2:** the Kaggle importances describe *the data generator*, not
agronomy. They are also associations, not causal effects.

### 2.2 Indian states, 1997–2020 (real-world validation)

- **Source:** Mendeley Data `ncw2vbcgnk` (https://data.mendeley.com/datasets/ncw2vbcgnk),
  CC BY 4.0. 19,689 × 10: Crop, Crop_Year, Season, State, Area, Production, Annual_Rainfall,
  Fertilizer, Pesticide, Yield. 55 crops and 30 states.
- **Audit:** 0 missing values and 0 duplicates.
  - Every `Season` value (19,689) and 172 `Crop` values have trailing spaces; these were
    stripped.
  - Yield skewness is 12.8, the maximum is 21,105, and 112 yields are 0.
- **Leakage and scale fixes (each has a test):**
  - **Production dropped.** Production/Area correlates 0.95 with Yield.
  - **Fertilizer and Pesticide converted to per hectare.**
  - **Rows with Area ≤ 0 or Yield ≤ 0 removed** (112 rows).
  - **Area kept only as log1p(Area).**
- **Additional decisions:**
  - **Coconut excluded** (165 rows). Its "yield" is in nuts/ha (median about 8,500), not t/ha.
  - **82 physically impossible yields kept but reported.** Examples: maize at 990 t/ha,
    Delhi potato at 311 t/ha. Removing test rows after seeing them would amount to tuning on
    the test set.
- **Key discovery:** per-hectare fertilizer and pesticide have **exactly one value per year**,
  the same for every crop and state (`india_inputs_per_ha_by_year.png`). The raw columns were
  built as Area × a national rate, so these features are a **national time index**, not farm
  inputs.
- **Clean data:** 19,412 rows. Train 1997–2015 (15,186 rows); test 2016–2020 (4,226 rows).

---

## 3. Methodology

**Preprocessing** (`src/features.py`). This is always fitted inside a scikit-learn `Pipeline`,
so it never sees validation or test rows.
- `StandardScaler` on the numeric features.
- `OneHotEncoder(handle_unknown="ignore")` on the categoricals.
  - `multiple_lr` uses `drop="first"`, to avoid exact collinearity with the intercept
    (the dummy-variable trap).
- The polynomial models expand **only the numeric features** (scale → PolynomialFeatures →
  scale).
- No imputer, because there are no missing values.

**Target transform.**
- **Kaggle:** none. Skewness is −0.10, below the rule-of-thumb threshold of 1, and log1p would
  make it worse (−0.52).
- **India:** log1p, via `TransformedTargetRegressor`. Skewness is 21.5 after cleaning and 1.95
  after log1p. All metrics are in the **original units**, because `predict` inverts the log.

**Splits and cross-validation.**
- **Kaggle:** random 80/20 split. `KFold(5, shuffle=True, random_state=42)` is shared by every
  model, plus a GroupKFold check by region (leave one region out).
- **India:** split by time (≤ 2015 / ≥ 2016). CV uses `YearBlockSplit`, which is
  `TimeSeriesSplit` over whole years (5 outer folds, 3 inner folds), so every validation year
  comes after all its training years.
- **Tree models use nested CV.** The `RandomizedSearchCV` runs *inside* each outer training
  fold, just as RidgeCV and LassoCV choose alpha inside each fold.
- **Model selection:** best linear, best overall and best tree are chosen by **CV RMSE before
  the test set is touched**.

**Models and tuned hyperparameters.**

| Model | Definition | Kaggle (full training split) | India (training years) |
|---|---|---|---|
| dummy_mean | `DummyRegressor(mean)` | — | predicts expm1(mean log1p y) |
| simple_lr | OLS on the numeric feature with the highest \|Pearson r\| on train | fertilizer_amount | Fertilizer_per_ha |
| multiple_lr | OLS, drop-first one-hot | — | — |
| polynomial | degree-2 (numeric only) + RidgeCV(logspace(−3, 3, 50)); plain OLS on polynomial terms is unstable | α = 44.98 | α = 0.373 (CV median) |
| polynomial_deg3 | degree 3 + RidgeCV; used only when there are ≤ 10 numeric features | skipped (11 features → 363 or more terms) | α = 0.494 (CV median) |
| ridge | RidgeCV(logspace(−3, 3, 50)) | α = 25.60 | α = 0.373 (CV median) |
| lasso | LassoCV(cv=5, max_iter=20000) | α = 0.0029 | α = 0.0002 (CV median) |
| random_forest | RF + RandomizedSearchCV (n_iter 20) | 587 trees, max_depth 16, min_samples_leaf 8, max_features 0.5 | 513 trees, max_depth 16, min_samples_leaf 1, max_features 1.0 |
| gradient_boost | XGBoost 3.4.1 + RandomizedSearchCV (n_iter 20) | 847 trees, max_depth 2, learning rate 0.036, subsample 0.93, colsample 0.61, min_child_weight 12, λ 0.10 | 724 trees, max_depth 6, learning rate 0.050, subsample 0.72, colsample 0.84, min_child_weight 4, λ 0.14 |

All settings use `RANDOM_STATE = 42`. The full values are in `results/tables/best_params.json`
and `india_best_params.json`.

---

## 4. Results

### 4.1 Kaggle: cross-validation (5-fold, training split, t/ha), `cv_all_models.csv`

| Model | RMSE (mean ± sd) | MAE | R² (mean ± sd) |
|---|---|---|---|
| polynomial | 0.6349 ± 0.0088 | 0.5074 | 0.6952 ± 0.0111 |
| gradient_boost | 0.6362 ± 0.0073 | 0.5079 | 0.6940 ± 0.0115 |
| lasso | 0.6406 ± 0.0052 | 0.5102 | 0.6898 ± 0.0082 |
| ridge | 0.6408 ± 0.0055 | 0.5102 | 0.6896 ± 0.0079 |
| multiple_lr | 0.6408 ± 0.0055 | 0.5102 | 0.6895 ± 0.0081 |
| random_forest | 0.6456 ± 0.0083 | 0.5158 | 0.6849 ± 0.0087 |
| simple_lr | 0.7352 ± 0.0062 | 0.5850 | 0.5913 ± 0.0146 |
| dummy_mean | 1.1509 ± 0.0148 | 0.9415 | −0.0009 ± 0.0010 |

### 4.2 Kaggle: held-out test (960 rows, evaluated once per pipeline run), `test_results.csv`

| Model | RMSE | MAE | R² |
|---|---|---|---|
| gradient_boost | 0.6461 | 0.5164 | 0.6899 |
| polynomial (best_overall and best_linear, chosen by CV) | 0.6462 | 0.5141 | 0.6898 |
| lasso | 0.6526 | 0.5230 | 0.6836 |
| ridge | 0.6533 | 0.5238 | 0.6830 |
| multiple_lr | 0.6533 | 0.5236 | 0.6829 |
| random_forest | 0.6635 | 0.5290 | 0.6730 |
| simple_lr | 0.7725 | 0.6226 | 0.5567 |
| dummy_mean | 1.1604 | 0.9285 | −0.0003 |

**GroupKFold by region** (`cv_groupkfold.csv`): polynomial 0.6334 and gradient_boost 0.6385
RMSE. These are **no worse than random KFold**, so the models do not memorise region-specific
patterns. That's consistent with region having no effect.

### 4.3 Kaggle: statistical diagnostics (notebook 02)

- **VIF:** every numeric feature is between 1.001 and 1.003, so there is no multicollinearity.
- **OLS** (standardised numeric features): R² 0.6938, adjusted R² 0.6921. Coefficients in t/ha
  per SD:

  | Feature | Coefficient |
  |---|---|
  | fertilizer | +0.879 |
  | pesticide | −0.265 |
  | rainfall | +0.214 |
  | soil moisture | +0.068 |
  | avg temperature | −0.063 |
  | nitrogen | +0.060 |
  | potassium | +0.057 |

  All of these have p < 10⁻⁷. They are small effects that only become detectable once
  fertilizer is controlled for (partial vs marginal association). pH, irrigation and every
  context dummy are not significant after correction.
- **Residuals (out-of-fold):** mean 0.000, σ 0.641, skew −0.03, Jarque-Bera p = 0.60, so they
  are normal. Residuals vs features show ∩-shaped misses of about ±0.1 t/ha for fertilizer,
  rainfall and pH.
- **Breusch-Pagan:** LM 21.8, p = 0.41 on the out-of-fold residuals (in-sample: 21.2,
  p = 0.45). So the residuals are **homoscedastic** and the classical standard errors are
  valid. HC3 changes any standard error by at most 3.1 %, and `ols_summary_hc3.csv` is not
  needed.
- **Lasso:** at the CV alpha of 0.0029, all 11 numeric features are kept and 6 context dummies
  are zeroed. Order of entry: fertilizer, pesticide, rainfall, then the small soil and
  temperature terms.
- **Simple vs multiple:** fertilizer alone captures **85.7 %** of multiple_lr's CV R²
  (0.591 / 0.690).

### 4.4 Kaggle: explainability (notebook 03)

- **Permutation importance** (polynomial, test split). RMSE increase in t/ha:

  | Feature | RMSE increase |
  |---|---|
  | fertilizer | 0.734 |
  | pesticide | 0.116 |
  | rainfall | 0.069 |
  | soil moisture | 0.009 |
  | … soil_ph (rank 8) | 0.007 |
  | context features | ≈ 0 |

- **Mean |SHAP|** (XGBoost, test split):

  | Feature | Mean \|SHAP\| (t/ha) |
  |---|---|
  | fertilizer | 0.745 |
  | pesticide | 0.241 |
  | rainfall | 0.184 |
  | **soil_ph** | **0.067** |
  | soil moisture | 0.058 |

- **Shapes:**
  - fertilizer: near-linear, with diminishing returns (a span of about 3 t/ha)
  - pesticide: falling, roughly linear
  - rainfall: saturating above about 600 mm
  - pH: flat-topped optimum between pH ≈ 6 and 7.5
  - phosphorus: a negligible ±0.03 t/ha threshold
- **Grouped importance:** Farming 64 %, Weather 17 %, Soil 16 %, Context 2 %.
- **Agreement:** all four methods have the same top 3.
  - Spearman ρ: coefficient vs Lasso 0.97; permutation vs SHAP 0.91; across the two families
    0.70–0.84.
  - The disagreements are soil_ph (curvature) and crop_type (coefficient rank 5, because its
    score is the maximum of four noisy dummies; permutation rank 14).

### 4.5 India: time-ordered CV (1997–2015) and the single 2016–2020 test

| Model | CV RMSE (mean ± sd) | CV R² | **Test RMSE** | Test MAE | **Test R²** |
|---|---|---|---|---|---|
| random_forest | 5.885 ± 2.962 | 0.769 | **8.119** | 1.580 | **0.650** |
| gradient_boost | 6.013 ± 2.917 | 0.761 | 8.136 | 1.400 | 0.648 |
| polynomial | 7.916 ± 2.397 | 0.606 | 9.481 | 2.180 | 0.523 |
| multiple_lr | 8.072 ± 2.321 | 0.592 | 9.508 | 2.190 | 0.520 |
| ridge | 8.106 ± 2.298 | 0.588 | 9.517 | 2.192 | 0.519 |
| lasso | 8.133 ± 2.301 | 0.586 | 9.550 | 2.201 | 0.516 |
| polynomial_deg3 | 9.860 ± 3.160 | 0.370 | 9.837 | 2.322 | 0.486 |
| simple_lr | 12.811 ± 1.559 | −0.044 | 14.014 | 4.114 | −0.043 |
| dummy_mean | 12.828 ± 1.560 | −0.047 | 14.039 | 4.080 | −0.047 |

**Reference predictors with no machine learning**, fitted on the training years only
(`india_reference_baselines.csv`):

| Predictor | Test RMSE | Test MAE | Test R² |
|---|---|---|---|
| crop look-up | 10.269 | 2.503 | 0.440 |
| **crop × state look-up** | **7.808** | 1.387 | **0.676** |
| best model (random_forest) | 8.119 | 1.580 | 0.650 |

**Per-crop test results for random_forest**, 10 most common crops (`india_per_crop_rmse.csv`):
- The median absolute error is **15–40 %** of each crop's mean yield.
- Per-crop R² is 0.43 for potato and 0.12 for rice, and **negative for the other 8**.
- The RMSE for maize, sesamum and bajra is inflated by impossible source records (sesamum's
  RMSE is 2.61 on a mean of 0.76, but its median absolute error is only 0.24).

**India SHAP** (random forest; 300-row test sample; shares of mean |SHAP|):

| Feature | Share |
|---|---|
| Crop | 83.4 % |
| State | 5.5 % |
| log_Area | 4.9 % |
| Annual_Rainfall | 4.0 % |
| Fertilizer_per_ha | 0.9 % |
| Season | 0.9 % |
| Pesticide_per_ha | 0.5 % |

### 4.6 India ablation: without the per-hectare "year index" columns (CODE_REVIEW H1)

`Fertilizer_per_ha` and `Pesticide_per_ha` hold one national value per year, so they encode the
year. The identical suite (same 1997–2015 / 2016–2020 split, same seeds, same time-ordered tuning)
was re-run **without both columns**. The results were only compared; no model or selection
changed because of them. Source: `india_ablation.csv`.

| Model | Test RMSE with | Test RMSE without | Δ RMSE (without − with) | R² with | R² without |
|---|---|---|---|---|---|
| random_forest | 8.1195 | 8.1352 | +0.0157 | 0.6499 | 0.6485 |
| gradient_boost | 8.1358 | 8.1545 | +0.0187 | 0.6485 | 0.6469 |
| polynomial | 9.4807 | 9.7929 | +0.3122 | 0.5227 | 0.4907 |
| multiple_lr | 9.5082 | 9.7829 | +0.2747 | 0.5199 | 0.4918 |
| ridge | 9.5169 | 9.7921 | +0.2752 | 0.5190 | 0.4908 |
| lasso | 9.5497 | 9.8247 | +0.2750 | 0.5157 | 0.4874 |
| polynomial_deg3 | 9.8373 | 9.7906 | −0.0467 | 0.4861 | 0.4910 |
| simple_lr | 14.0139 | 14.0356 | +0.0217 | −0.0429 | −0.0462 |
| dummy_mean | 14.0391 | 14.0391 | 0.0000 | −0.0467 | −0.0467 |

Without the two columns, `simple_lr`'s feature switches from Fertilizer_per_ha to log_Area.

**Do the test-year values fall outside the training range?** Yes (`india_feature_ranges.csv`):

| Feature | Train range (1997–2015) | Test range (2016–2020) | Test rows above the training max |
|---|---|---|---|
| Fertilizer_per_ha | 94.67 – 167.52 | 153.25 – 193.61 | 25.9 % (the 2019 and 2020 values) |
| Pesticide_per_ha | 0.09 – 0.33 | 0.35 – 0.38 | **100 %** |

**Verdict:**
- **The columns help the linear models modestly.** RMSE is 0.27–0.31 worse without them
  (≈ 3 %), and R² falls by ≈ 0.03. On the log scale a linear model extrapolates the rising
  index, which acts as a *time trend*. It cannot separate a fertilizer effect from any other
  trend.
- **They do essentially nothing for the tree models** (+0.016 / +0.019 RMSE, ≈ 0.2 %). Trees
  cannot extrapolate, and every test row lies beyond the training range for pesticide, and 26 %
  for fertilizer.
- **They slightly hurt the degree-3 polynomial** (−0.047), where the cubic terms of an
  extrapolated index overfit.
- **Every headline conclusion holds with or without them:** random forest is best (8.12 → 8.14),
  trees beat linear models, and the crop × state look-up (7.81) beats all models.

---

## 5. Figure index

All figures are in `results/figures/`, saved at 150 dpi.

| Figure | Caption |
|---|---|
| `eda_target_distribution.png` | Kaggle yield: raw (skewness −0.10) vs log1p (−0.52). The raw target is used. |
| `eda_hist_soil.png` | Soil features are uniformly distributed, a sign of synthetic data. |
| `eda_hist_weather.png` | Weather features are uniformly distributed. |
| `eda_hist_farming.png` | Farming features are uniformly distributed. |
| `eda_counts_categorical.png` | Crop, region and season levels are almost perfectly balanced. |
| `eda_yield_by_context.png` | Yield is the same for every crop, region and season (ANOVA p ≥ 0.24). |
| `eda_corr_pearson.png` | Pearson correlations: only fertilizer (0.77), pesticide and rainfall relate to yield; features are mutually independent. |
| `eda_corr_spearman.png` | Spearman correlations are practically identical to Pearson (largest gap 0.013). |
| `eda_scatter_top5_lowess.png` | Yield vs the 5 strongest features, with LOWESS trends. |
| `eda_binned_means.png` | Mean yield per decile of each feature (95 % CI). The pH optimum and rainfall saturation are visible. |
| `eda_interaction_fertilizer.png` | Fertilizer slopes are parallel across crops and irrigation levels, so there is no interaction. |
| `eda_dropped_columns.png` | Harvest month and field_id carry no signal, which justifies dropping them. |
| `diag_residuals.png` | multiple_lr out-of-fold residuals: structureless, normal, homoscedastic. |
| `diag_residuals_vs_features.png` | ∩-shaped residual patterns for fertilizer, rainfall and pH: curvature the line misses. |
| `diag_lasso_path.png` | Lasso coefficient paths. Fertilizer enters first; the context dummies stay near 0. |
| `model_comparison.png` | Kaggle test RMSE by model; the dummy baseline is shown in grey. |
| `pred_vs_actual_best.png` | Predicted vs actual, polynomial (best overall; test RMSE 0.646). |
| `pred_vs_actual_linear.png` | Predicted vs actual for the best linear model (also polynomial). |
| `explain_coefficients.png` | Top 15 standardised coefficients of multiple_lr (positive in green, negative in red). |
| `explain_permutation.png` | Permutation importance of the polynomial model on the test split (± SD over 10 repeats). |
| `explain_shap_beeswarm.png` | XGBoost SHAP beeswarm: direction and size of each feature's effect. |
| `explain_shap_bar.png` | Mean \|SHAP\| per original feature (one-hot values summed). |
| `explain_shap_dependence.png` | SHAP dependence: fertilizer linear, pesticide falling, rainfall saturating, pH optimum, phosphorus negligible. |
| `explain_grouped_importance.png` | Importance by group: Farming 64 %, Weather 17 %, Soil 16 %, Context 2 %. |
| `india_inputs_per_ha_by_year.png` | India fertilizer and pesticide per ha: one national value per year. |
| `india_target_distribution.png` | India yield: skewness 21.5 raw vs 1.95 after log1p. |
| `india_model_comparison.png` | India test RMSE (2016–2020) by model. |
| `india_per_crop_error.png` | India per-crop error relative to mean yield: RMSE vs median absolute error. |
| `india_shap_importance.png` | India SHAP: crop type is 83 % of importance; fertilizer and pesticide are under 1 % each. |

---

## 6. Files created

| File | Purpose |
|---|---|
| `requirements.txt` | Top-level dependencies (unpinned, as specified). |
| `requirements-lock.txt` | `pip freeze` of the exact environment that produced the results (identical to the current freeze). |
| `pyproject.toml` | Ruff line length (100), pytest test path, and a filter for a third-party shap deprecation notice. |
| `.gitignore` | Excludes `.venv/`, `data/raw/`, caches, `*.joblib`, `kaggle.json`, `access_token` and `.claude/`. |
| `.claude/launch.json` | Local preview configuration used to test the Streamlit app (git-ignored). |
| `src/__init__.py` | Makes `src` a package. |
| `src/config.py` | Single source of truth: seed, split sizes, paths, every column name, units and feature groups. |
| `src/data.py` | Kaggle loading, dropping columns, deduplication, the 80/20 split to parquet, and the shared KFold. |
| `src/features.py` | `build_preprocessor` (scale, one-hot, optional numeric-only polynomial) and `transformed_frame`. |
| `src/models.py` | The linear family, RF and XGBoost with RandomizedSearchCV, the log-target wrapper and unwrapping helpers. |
| `src/evaluate.py` | `cross_validate_models` (mean ± sd RMSE, MAE, R², fit time, alpha) and `holdout_metrics`. |
| `src/eda.py` | Binned means, the linearity and quadratic tests, the feature-signal table and top-Pearson selection. |
| `src/explain.py` | Coefficients, permutation importance, TreeExplainer SHAP, grouping, agreement, and Kernel SHAP for any model. |
| `src/plots.py` | `save_figure` (150 dpi, tight layout) and axis labels with units. |
| `src/pipeline.py` | `python -m src.pipeline`: the end-to-end Kaggle run (CV, GroupKFold, one test, models, figures, submission). |
| `src/india.py` | `python -m src.india`: India cleaning, time split, YearBlockSplit, look-up baseline, per-crop metrics. |
| `tests/test_data.py` | 6 tests: split sizes, no overlap, target not a feature, config columns exist, determinism, parquet round-trip. |
| `tests/test_models.py` | 5 tests: every pipeline fits and predicts (raw and log target), outputs in t/ha, no preprocessing leakage, simple_lr uses one feature, polynomial expands numeric features only. |
| `tests/test_explain.py` | 3 tests: feature-name mapping, exact Kernel SHAP additivity, readable coefficient baselines. |
| `tests/test_india.py` | 10 tests: Production and Crop_Year never reach a model, time split, per-hectare conversion, no rows ≤ 0, whitespace stripped, time-ordered folds (real and toy data), look-up baseline, ablation models exclude the year-index columns, ablation/range helpers. |
| `notebooks/01_eda.ipynb` | EDA on the training split, including the non-linearity, interaction and dropped-column evidence. |
| `notebooks/02_diagnostics.ipynb` | VIF, OLS inference, out-of-fold residuals, Breusch-Pagan, Lasso path, simple vs multiple. |
| `notebooks/03_explain.ipynb` | Coefficients, permutation importance, SHAP (beeswarm, bar, dependence), grouped importance, agreement. |
| `notebooks/04_india.ipynb` | India audit, leakage checks, cleaning, time split, results, look-up baselines, per-crop errors, SHAP, transfer table. |
| `app.py` | Streamlit demo: sliders, prediction, SHAP waterfall, and 4 tabs. |
| `results/tables/*.csv, *.json` | Every reported table (the CV, test, diagnostics, importance and India tables). |
| `results/figures/*.png` | The 29 figures in §5. |
| `models/*.joblib, *.json` | best_linear, best_overall, best_tree, India best models, model names, app slider metadata. |
| `data/processed/*.parquet` | The locked Kaggle split and the cleaned India time split. |
| `submissions/submission.csv` | Competition-format predictions from best_overall (**not uploaded**). |
| `CODE_REVIEW.md` | Phase 11 findings by severity, and viva questions with answers. |
| `CROP_YIELD_BUILD_REPORT.md` | This report. |

---

## 7. Verification evidence

All output below is from the **final verification round** on the final code (after the Phase 11
fixes and the ablation). It was run on 2026-09-27: ruff, pytest and pipeline run A at
13:21–13:26; run B, the hashes and the India run at 16:33–16:56; the notebooks at 16:57–16:59;
the checklist, the app checks and the safety checks at about 17:00.

### 7.1 Lint and tests (Phase 10 steps 1–2)

```text
$ python -m ruff check src tests app.py
All checks passed!
$ python -m pytest -q
........................                                                 [100%]
24 passed in 3.68s
```

### 7.2 All notebooks executed top to bottom (step 3)

```text
$ python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb notebooks/02_diagnostics.ipynb notebooks/03_explain.ipynb notebooks/04_india.ipynb
[NbConvertApp] Converting notebook notebooks/01_eda.ipynb to notebook
[NbConvertApp] Writing 1687701 bytes to notebooks\01_eda.ipynb
[NbConvertApp] Converting notebook notebooks/02_diagnostics.ipynb to notebook
[NbConvertApp] Writing 727582 bytes to notebooks\02_diagnostics.ipynb
[NbConvertApp] Converting notebook notebooks/03_explain.ipynb to notebook
[NbConvertApp] Writing 568465 bytes to notebooks\03_explain.ipynb
[NbConvertApp] Converting notebook notebooks/04_india.ipynb to notebook
[NbConvertApp] Writing 283236 bytes to notebooks\04_india.ipynb
exit=0 at Sun Sep 27 16:59:14 IST 2026
```

Every executed notebook, scanned for error outputs, stderr outputs and embedded figures:

```text
===== notebooks/01_eda.ipynb
--> errors=0 stderr_outputs=0 images=12
===== notebooks/02_diagnostics.ipynb
--> errors=0 stderr_outputs=0 images=3
===== notebooks/03_explain.ipynb
--> errors=0 stderr_outputs=0 images=6
===== notebooks/04_india.ipynb
--> errors=0 stderr_outputs=0 images=4
```

### 7.3 Reproducibility: delete tables and models, run the pipeline twice, compare hashes

```text
$ rm -rf results/tables models
results/ now contains: figures   models/ exists: no
$ python -m src.pipeline   # run A
exit=0 at Sun Sep 27 13:26:54 IST 2026
$ python -m src.pipeline   # run B
      0 [main] bash 1300 dofork: child -1 - forked process 25640 died unexpectedly, retry 0, exit code 0xC000026B, errno 11
/usr/bin/bash: fork: retry: Resource temporarily unavailable
=== RESUMED Sun Sep 27 16:33:26 IST 2026 (run B failed to start at 13:26 with a bash fork error; run A outputs kept)
$ python -m src.pipeline   # run B
exit=0 at Sun Sep 27 16:43:06 IST 2026
$ sha256sum test_results.csv (run A copy, run B)
b944a98cf03f5968a9a5c5d0593c7bbc38a0a373edfbe687c4b4bf0aa3538ff6 *runA/test_results.csv
b944a98cf03f5968a9a5c5d0593c7bbc38a0a373edfbe687c4b4bf0aa3538ff6 *results/tables/test_results.csv
$ sha256sum best_params.json (A, B)
9d27827f03b7e3d076f51e55aed16f6c76c275ee06c8cdb4a683adf642ff2cea *runA/best_params.json
9d27827f03b7e3d076f51e55aed16f6c76c275ee06c8cdb4a683adf642ff2cea *results/tables/best_params.json
$ sha256sum submission.csv (A, B)
07c94667f91f30fa7becb5a1cc85900ef31f6f0ebd3405d654f46a23bd67c193 *runA/submission.csv
07c94667f91f30fa7becb5a1cc85900ef31f6f0ebd3405d654f46a23bd67c193 *submissions/submission.csv
$ python -m src.india
exit=0 at Sun Sep 27 16:56:28 IST 2026
=== pipeline + india finished Sun Sep 27 16:56:28 IST 2026
```

**Result: identical.** The SHA-256 of `test_results.csv` is the same for run A and run B, as are
the hashes of `best_params.json` and `submission.csv`. Neither pipeline log contains a warning
(`grep -ci warning runA.log runB.log` returned 0 and 0).

### 7.4 Correctness checklist (script: `checklist.py`, reading the regenerated artifacts)

```text
$ python checklist.py
[cv rows] 8 rows: ['dummy_mean', 'gradient_boost', 'lasso', 'multiple_lr', 'polynomial', 'random_forest', 'ridge', 'simple_lr']
          exactly the 8 expected models: True; all metric columns present and non-null: True
[beats dummy] dummy test RMSE = 1.1604; models NOT beating it: none
[units] target std (test split) = 1.1608 t/ha, target mean = 6.282 t/ha; best model test RMSE = 0.6461 t/ha < std: True; all non-dummy test RMSE < std: True
[leakage flag] max R2 per table: {'kaggle_cv': np.float64(0.6952), 'kaggle_test': np.float64(0.6899), 'india_cv': np.float64(0.7688), 'india_test': np.float64(0.6499)}; any > 0.98: False
[india years] test years = [np.int64(2016), np.int64(2017), np.int64(2018), np.int64(2019), np.int64(2020)]; min = 2016 (>= 2016: True); train max = 2015
[production] 'Production' in feature matrix columns: False; in fitted model inputs (93 one-hot/numeric inputs): False; in parquet columns: False
[figures] 25 figure names saved by the notebooks; missing files: none; total PNGs in results/figures: 29
```

| Checklist item | Result |
|---|---|
| Delete, run twice, identical `test_results.csv` | ✅ identical SHA-256 (§7.3) |
| `cv_all_models.csv` has the 8 models, each with RMSE, MAE and R² mean and std | ✅ |
| Every model beats `dummy_mean` on test RMSE | ✅ none fail (dummy 1.1604) |
| Test RMSE in t/ha, below the target std, same scale as the target | ✅ best 0.6461 < std 1.1608 (target mean 6.28 t/ha) |
| No R² > 0.98 | ✅ maximum is 0.7688 (India CV) |
| Every figure referenced in a notebook exists | ✅ 25 referenced, 0 missing (29 PNGs in total, including 4 from the pipelines) |
| India test set only contains Crop_Year ≥ 2016 | ✅ years 2016–2020; training max is 2015 |
| "Production" is nowhere in the India feature matrix | ✅ not in X, not in the 93 fitted model inputs, not in the parquet |

### 7.5 Demo checklist (script: `app_check.py`, which starts a real server and uses Streamlit `AppTest`)

```text
$ python app_check.py
2026-09-27 17:01:03.099 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-09-27 17:01:06.454 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-09-27 17:01:08.450 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-09-27 17:01:10.426 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
2026-09-27 17:01:12.314 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
[startup] server ready in 1.1s; first full app run (load model, predict, SHAP waterfall, all tabs) in 3.4s; total 4.5s
[errors] exceptions: 0; error boxes: 0
[tabs] ['Predict', 'Model comparison', 'What drives yield', 'Real-data validation']
[default] Predicted yield (polynomial model): 6.52 t/ha
[waterfall] pyplot figures rendered: 0
[explanation] **fertilizer_amount** pushed this prediction up the most (+0.10 t/ha); **pesticide_usage** pushed it down the most (-0.02 t/ha).
[slider] fertilizer amount (kg/ha) = 299.9330210739181 -> 7.76 t/ha   |   fertilizer amount (kg/ha) = 50.04871558432929 -> 4.70 t/ha
[slider] irrigation frequency (times/week) = 6.0 -> 6.58 t/ha   |   irrigation frequency (times/week) = 1.0 -> 6.54 t/ha
[terminal] server output lines: 8; warning/error lines: none
```

The "missing ScriptRunContext" lines come from `AppTest`'s bare-mode test harness, not from
the app. The real `streamlit run` server's own output had **no warning or error lines**
(`[terminal]` line). A separate `python -m streamlit run app.py` in the browser preview also
logged only its start-up banner.

| Demo item | Result |
|---|---|
| Starts with no errors or warnings | ✅ 0 exceptions, 0 error boxes, and no warning lines from the server |
| Start-up under 10 s | ✅ 4.5 s (server ready in 1.1 s; first full run in 3.4 s) |
| Most important slider (fertilizer) changes the prediction visibly | ✅ 4.70 → 7.76 t/ha across its range (default 6.52) |
| Least important slider (irrigation) changes it only slightly | ✅ 6.54 → 6.58 t/ha across its range |
| SHAP waterfall renders for the default input | ✅ in the browser, the Predict tab showed one visible 1460×1238 px waterfall image plus the explanation line |
| All four tabs render | ✅ Predict, Model comparison, What drives yield and Real-data validation; no exceptions; images present in each figure tab |

### 7.6 Regression and safety checks

```text
$ (PowerShell) files modified today in sibling folders of C:\Users\Saurav\Desktop\WORK
08.33.03  Saurav'sWorkSpace\Welcome.md
08.33.04  Saurav'sWorkSpace\.obsidian\app.json
08.33.04  Saurav'sWorkSpace\.obsidian\appearance.json
08.33.03  Saurav'sWorkSpace\.obsidian\core-plugins.json
08.33.05  Saurav'sWorkSpace\.obsidian\graph.json
08.33.04  Saurav'sWorkSpace\.obsidian\workspace.json
11.38.13  Stacked Ensemble Model for Accurate Crop Yield Prediction Using Machine Learning Techniques\Crop Yield Data\crop_yield.csv
$ grep -rnI --exclude-dir={.venv,.git,data} "kaggle.json" .
./.gitignore:6:kaggle.json
$ grep -rnI ... "\"key\""
$ grep -rnI ... "username"
$ find . -name kaggle.json -o -name access_token
(none found above = no credential files)
$ git check-ignore -v data/raw/india/crop_yield.csv .venv/pyvenv.cfg models/best_overall.joblib
.gitignore:2:data/raw/	data/raw/india/crop_yield.csv
.gitignore:1:.venv/	.venv/pyvenv.cfg
.gitignore:5:*.joblib	models/best_overall.joblib
$ git rev-list --all --count
0
$ git status --short
?? .gitignore
?? CODE_REVIEW.md
?? CROP_YIELD_BUILD_REPORT.md
?? app.py
?? data/
?? models/
?? notebooks/
?? pyproject.toml
?? requirements-lock.txt
?? requirements.txt
?? results/
?? src/
?? submissions/
?? tests/
```

- **Files outside the project:** the only files modified today in sibling folders are an
  Obsidian vault created at 08:33 (`Saurav'sWorkSpace\…`) and
  `Stacked Ensemble Model…\Crop Yield Data\crop_yield.csv` at 11:38:13. That file has the same
  size (1,620,945 bytes) and write time as the India CSV you copied into this project at
  11:38:40. **Neither was created or modified by this build.**
- **Credentials:** there are no credential files. The only `kaggle.json` match in code or config
  is the `.gitignore` rule; the two `.md` reports mention the filename in prose. There are no
  `"key"` or `username` matches.
- **Git:** `.gitignore` covers `data/raw/`, `.venv/` and `*.joblib`, verified with
  `git check-ignore`. The repository has **0 commits**.

---

## 8. Code review summary

See `CODE_REVIEW.md` for details.

| Severity | Count | Fixed | Documented / follow-up |
|---|---|---|---|
| CRITICAL | 0 | — | — |
| HIGH | 1 | 1: the misleading "Crop_Year is never a feature" claim (the year enters via the per-hectare index). Resolved by (a) correcting the text in notebook 04, config and tests, and (b) **measuring** it with the ablation in §4.6. | — |
| MEDIUM | 4 | 3 (hardcoded column names ×2, wrong Lasso reasoning) | 1 follow-up: the India Ridge/Lasso alphas are chosen with folds that are not time-ordered |
| LOW | 12 | 9 | 3 documented (global RNG seeding in Kernel SHAP, India dummy = geometric mean, repeated test evaluation across reruns) |

**No reported number changed because of a fix.** The fixes are refactors, renames, text
corrections and one extra printed statistic (the in-sample Breusch-Pagan test). In the final,
post-fix run, pipeline runs A and B produced byte-identical `test_results.csv`, `best_params.json`
and `submission.csv` files (§7). Every value equals those from the pre-fix runs (e.g. test RMSE
0.6461 / 0.6462 / 0.6526 …; India 8.1195 / 8.1358 …). The only new numbers are the ablation's
(§4.6).

---

## 9. Deviations from the specification

1. **Framing:** the Kaggle data is treated as a synthetic benchmark and India as the real-world
   test, with Phase 8 made mandatory. Agreed with you after Phase 1.
2. **Kaggle credentials:** the new-format token `~/.kaggle/access_token` was used instead of
   `kaggle.json`, following your auth note. It was checked for existence only, never opened,
   and is git-ignored.
3. **Nested CV for the tree models:** the RandomizedSearchCV runs inside every outer fold, so
   the tree models' CV is not optimistically biased. The spec only said "light tuning" plus
   the same KFold.
4. **Parallelism:** RF and XGBoost use `n_jobs=1` *inside* a search that uses `n_jobs=-1`, so
   all cores are used at the search level. Nested `n_jobs=-1` produced 2,224 joblib warnings
   and no speed-up. Because this changes XGBoost's floating-point summation order, XGBoost
   moved from **CV 0.6375 → 0.6362** and **test 0.6459 → 0.6461**. The tuned hyperparameters
   are identical, and both settings are exactly reproducible run-to-run. The change was made
   for the warnings, not because of the test result, and model selection is unchanged.
5. **Model saving:** the saved `.joblib` files hold the refitted best *estimator*, not the
   search wrapper. The wrapper pickled the India `YearBlockSplit` as `__main__.YearBlockSplit`,
   which could not be loaded again.
6. **GroupKFold:** GroupKFold also runs on the best *tree* model, because best linear = best
   overall = polynomial, which would otherwise have left a single row.
7. **Extra saved model:** `best_tree.joblib` is saved in addition to the two required models,
   because SHAP needs the best *tree* model and the best overall model is the polynomial.
8. **Streamlit SHAP:** the app uses **Kernel SHAP** in the original feature space. The
   permutation explainer's numba compilation took about 20 s on the first call, which broke
   the start-up budget. Additivity is tested to be exact.
9. **Permutation importance** runs serially (`n_jobs=None`). It takes seconds, and parallel
   workers printed a harmless joblib clean-up error when the Jupyter kernel exited on Windows.
10. **India-specific choices:**
    - Coconut excluded (its units are nuts/ha).
    - `Crop_Year` is used for splitting only.
    - SHAP uses a **300-row** test sample (single-threaded TreeExplainer on 513 deep trees took
      more than 10 minutes for 2,000 rows).
    - Crop and crop × state look-up baselines were added.
    - A per-crop median absolute error was added.
11. **Extra EDA** as you requested: mutual information, binned means, a quadratic test with
    Bonferroni correction (the most sensitive curvature test), interaction F-tests, and
    evidence for the dropped columns.
12. **Python 3.14.4** was the only interpreter installed. All packages installed without
    pinning.
13. **Ablation added after the code review** (at your request): India without
    Fertilizer_per_ha and Pesticide_per_ha, plus a train/test range check (§4.6), notebook 04
    §8b, the app's Real-data tab, and 2 tests.
14. **Verification runs:**
    - Two earlier verification rounds were stopped on purpose. They had been started before the
      review fixes and the ablation, and every number in this report must come from the final
      code.
    - In the final round, run B failed to *start* (a bash `fork: Resource temporarily
      unavailable` error on Windows). It was re-launched with no code change in between;
      `find src tests app.py -newer runA/test_results.csv` returned nothing.
    - Run A's outputs were kept for the hash comparison.

---

## 10. Limitations and follow-ups

- **The Kaggle data is likely synthetic.** Its importances describe the generator: yield is
  essentially a fertilizer function plus small curves and Gaussian noise (σ ≈ 0.64 t/ha). No
  agronomic conclusion should be drawn from it alone.
- **The India data has no soil variables at all**, only **state-level annual rainfall**, and
  **fertilizer and pesticide that are national yearly averages** disguised as per-row totals.
  It cannot test the field-level mechanisms the Kaggle model relies on. District- or
  farm-level data with measured inputs would be needed.
- **There is no district-level weather.** A single rainfall value per state-year hides
  within-state variation, and there is no temperature at all.
- **Data quality:** 82 India records are over 10× their crop's median, and several are
  impossible (e.g. maize at 990 t/ha). They were kept to avoid test-set tuning. A documented
  outlier rule applied *before* splitting would be a better follow-up.
- **Mixed units:** after excluding Coconut, some crops may still use non-tonne units (cotton,
  jute and mesta are often reported in bales). Per-crop metrics reduce, but don't remove, this
  concern.
- **Within-crop skill is poor.** Per-crop R² is negative for 8 of the 10 top crops, and a
  crop × state look-up table beats the models. The real-data models learn yield levels, not
  year-to-year variation.
- **Follow-up (CODE_REVIEW M4):** choose the India Ridge/Lasso alphas with year-ordered inner
  folds, which needs scikit-learn metadata routing.
- **The test sets were evaluated in several reruns** (for reproducibility and the thread-count
  fix). No choice was made from test results, but a single pre-registered run would be
  stricter.
- **India SHAP uses a 300-row sample.** Rankings at the top are stable, but the small shares
  (under 1 %) carry sampling noise.
- **The Streamlit app explains the polynomial model with Kernel SHAP** (50 background rows),
  which is sampled. The explanations are exactly additive but approximate for individual
  features.
