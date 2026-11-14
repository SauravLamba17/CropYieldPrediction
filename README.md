# 🌾 Crop Yield Prediction Using Machine Learning

**Predicting agricultural crop yield (tons/hectare) from soil, weather and farming parameters, and testing whether the findings hold on real Indian government data.**

---

## Table of Contents

1. [What this project is about](#1-what-this-project-is-about)
2. [Key findings at a glance](#2-key-findings-at-a-glance)
3. [Research questions](#3-research-questions)
4. [Datasets](#4-datasets)
5. [Methodology](#5-methodology)
6. [Results in detail](#6-results-in-detail)
7. [The interactive demo app](#7-the-interactive-demo-app)
8. [Project structure](#8-project-structure)
9. [How to reproduce everything](#9-how-to-reproduce-everything)
10. [Verification and reproducibility](#10-verification-and-reproducibility)
11. [Limitations](#11-limitations)
12. [Further documentation](#12-further-documentation)

---

## 1. What this project is about

Farmers, agronomists and policy-makers want to know how much a field will produce before harvest and which factors matter most for that output. This project treats that as a **supervised regression problem**: given measurements of soil, weather and farming practices, predict crop yield in tons per hectare.

The project goes beyond fitting one model. It:

- **Compares eight models of increasing flexibility**, from a baseline that always predicts the average, through simple, multiple, polynomial, Ridge and Lasso regression, up to Random Forest and XGBoost. This shows how much accuracy each step in complexity actually buys.
- **Runs full statistical diagnostics** (multicollinearity, residual normality, heteroscedasticity, coefficient significance) instead of reporting a single accuracy score.
- **Explains the models** with four independent importance methods (standardized coefficients, Lasso selection, permutation importance and SHAP) and checks whether they agree.
- **Tests the conclusions on real data.** The primary Kaggle dataset turned out to be synthetic, so every finding is re-tested on 24 years of real Indian state-level crop data, with a time-based split that simulates forecasting future years.
- **Ships an interactive Streamlit app** where you can move sliders for each input and see the predicted yield and a per-feature explanation update live.

```
 Soil     (pH, moisture, N, P, K)      ─┐
 Weather  (rainfall, temp, sunlight)   ─┤
 Farming  (fertilizer, pesticide,       ├─► Preprocessing ─► 8 regression models ─► Yield (t/ha)
           irrigation)                  │   (scaling, one-hot,    compared by CV,      + explanation
 Context  (crop, region, season)       ─┘    optional log target)  tested once          (SHAP)
```

---

## 2. Key findings at a glance

| Question | Answer |
|---|---|
| **How accurately can yield be predicted?** | On the Kaggle benchmark: **test RMSE 0.646 t/ha, R² 0.69**, against 1.160 for a predict-the-mean baseline. On real Indian data: **R² 0.65** with Random Forest. |
| **What drives yield?** | **Fertilizer**, then **pesticide** (negative effect), then **rainfall**, with the same top three under all four importance methods. Farming practices account for 64 % of the model's reliance. |
| **Is the relationship linear?** | **Mostly.** There is real but mild curvature: soil pH has an **optimum around 6–7.5**, rainfall **saturates**, and fertilizer shows **diminishing returns**. Polynomial regression ties XGBoost and beats plain linear regression by only 0.007 t/ha. |
| **Do the findings hold on real data?** | **Largely not.** On real Indian data, **crop type carries 83 % of the importance**, tree models clearly win, and a **simple crop×state historical-average lookup beats every ML model**. |

**The most important lesson:** a model that looks good on a clean benchmark can tell a completely different story on real data. And a strong, simple baseline, like looking up the historical average for that crop in that state, must always be checked before claiming that machine learning adds value.

---

## 3. Research questions

| # | Question | How it is answered |
|---|---|---|
| **RQ1** | How accurately can crop yield be predicted from environmental and agricultural variables? | RMSE, MAE and R² under 5-fold cross-validation, plus a single final evaluation on a held-out test set |
| **RQ2** | Which variables have the greatest influence on yield? | Standardized regression coefficients, Lasso feature selection, permutation importance and SHAP, with rank agreement measured by Spearman ρ |
| **RQ3** | Is the relationship linear? | Simple LR vs Multiple LR vs Polynomial vs Ridge/Lasso vs Random Forest vs XGBoost; curvature tests in EDA; SHAP dependence plots |
| **RQ4** | Do the findings transfer to real-world data? | The same model suite re-run on real Indian state data (1997–2020) with a time-based split |

---

## 4. Datasets

### A. Kaggle Crop Yield Prediction Challenge (primary benchmark)

- **Source:** https://www.kaggle.com/competitions/crop-yield-prediction-challenge
- **Size:** 4,800 labelled rows (training file) plus 1,200 unlabelled rows (competition test file)
- **Target:** `yield_tpha`, crop yield in tons per hectare (mean 6.27, std 1.15, range 2.34–9.54)

| Group | Features |
|---|---|
| Soil | `soil_ph`, `soil_moisture`, `nitrogen_content`, `phosphorus_content`, `potassium_content` |
| Weather | `avg_temperature`, `total_rainfall`, `sunlight_hours` |
| Farming | `fertilizer_amount`, `pesticide_usage`, `irrigation_frequency` |
| Context | `crop_type` (5 crops), `region` (5), `season` (3) |

**Dropped columns:** `id` (a row identifier), plus `harvest_date` and `field_id`. Both are undocumented and were tested: they are statistically indistinguishable from random (harvest month ANOVA p = 0.56; field ID permutation p = 0.26). Keeping `field_id` would only let a model memorise field identities.

> ⚠️ **This dataset is almost certainly synthetic.** The evidence:
> - Every numeric feature follows a uniform distribution.
> - Features that must correlate in reality don't (rainfall vs soil moisture: r = 0.006).
> - Crop type has no effect on yield, although in reality corn out-yields soybean several times over.
> - There are no missing values, no duplicates, and suspiciously round value ranges.
>
> The project therefore treats this dataset as a **controlled benchmark**. The importance findings describe *how the data was generated*, not real agronomy. The real-world question is answered by dataset B.

### B. Crop Yield in Indian States, 1997–2020 (real-world validation)

- **Source:** https://data.mendeley.com/datasets/ncw2vbcgnk (CC BY 4.0).
- **Size:** 19,689 rows covering 55 crops across Indian states and union territories
- **Columns:** Crop, Crop_Year, Season, State, Area, Production, Annual_Rainfall, Fertilizer, Pesticide, Yield

**Cleaning and leakage fixes:**

| Step | Why |
|---|---|
| Dropped `Production` | Yield = Production ÷ Area, so keeping it would let the model learn division instead of agronomy |
| Converted Fertilizer and Pesticide to per-hectare values | The raw columns are totals that grow with farm size |
| Removed 112 rows with zero yield | Invalid target |
| Removed Coconut (165 rows) | Its yield is recorded in nuts per hectare, a different unit from every other crop |
| Used `log1p(Area)` as a feature | Area is extremely skewed |
| Log-transformed the target | Yield skewness drops from 21.5 to 1.95 |

The result is **19,412 clean rows**. The data also contains 82 physically implausible yield records (for example, maize at 990 t/ha). They were kept, because removing them would mean second-guessing the source, and they are reported openly. A per-crop median absolute error is included so that these records don't distort the per-crop comparison.

> ⚠️ **An important discovery:** the India Fertilizer and Pesticide columns are **not field measurements**. They are Area multiplied by a single national rate per year, so year explains 100 % of their per-hectare variation. They therefore cannot test the fertilizer finding from dataset A. An ablation study (section 6.5) measures exactly what they contribute.

---

## 5. Methodology

### 5.1 Train/test splits

| Dataset | Split | Rationale |
|---|---|---|
| Kaggle | Random 80/20 (3,840 train / 960 test), seed 42 | The rows are independent, so a random split is fine |
| India | **Time-based:** train on 1997–2015 (15,186 rows), test on 2016–2020 (4,226 rows) | A random split would let the model "see the future"; a time split simulates real forecasting |

The **test set is touched exactly once**, after every modelling decision is final. All model selection and tuning happens with cross-validation on the training data only.

### 5.2 Preprocessing

Everything is inside scikit-learn `Pipeline` and `ColumnTransformer` objects, so scaling and encoding are fitted only on the training data within each fold. This prevents data leakage by construction.

- Numeric features: `StandardScaler`
- Categorical features: `OneHotEncoder(handle_unknown="ignore")`. `drop="first"` is used for plain OLS to avoid the dummy-variable trap.
- Target: raw for Kaggle (skewness −0.10, so no transform is needed), `log1p` for India. All metrics are reported on the original scale.

### 5.3 Models

| Model | Description |
|---|---|
| `dummy_mean` | Always predicts the training mean (on India, the mean of log-yield, back-transformed). Every model must beat it. |
| `simple_lr` | Linear regression on the single most correlated feature (fertilizer), chosen using training data only |
| `multiple_lr` | Linear regression on all features |
| `polynomial` | Degree-2 polynomial expansion of the numeric features plus RidgeCV. Degree 3 was skipped on Kaggle because it would create 363+ terms. |
| `ridge` | RidgeCV, with α searched over 10⁻³ to 10³ |
| `lasso` | LassoCV with 5-fold internal CV |
| `random_forest` | Tuned with nested `RandomizedSearchCV` (20 iterations) |
| `gradient_boost` | XGBoost, tuned the same way |

On India, every penalty (polynomial, Ridge, Lasso) is instead tuned by a grid search on **time-ordered year blocks**, the same scheme as the evaluation, so no later year informs the choice of α.

**Nested cross-validation** is used for the tree models. Their hyperparameter search runs *inside* each outer fold, just as RidgeCV and LassoCV choose α inside each fold. This keeps the comparison fair and avoids optimistic scores.

### 5.4 Evaluation

- 5-fold `KFold` cross-validation (shuffled, seed 42) on Kaggle; time-ordered year-block CV on India
- Metrics: **RMSE** (the Kaggle competition metric), MAE and R²
- An extra generalization check: **GroupKFold by region** (leave one region out)
- An extra reference baseline for India: **crop×state historical-average lookup** (no machine learning)

### 5.5 Statistical diagnostics

- **Variance Inflation Factor** for multicollinearity
- **statsmodels OLS** with coefficients, p-values and 95 % confidence intervals
- **Out-of-fold residual analysis:** residuals vs fitted values, Q-Q plot, Jarque-Bera normality test, **Breusch-Pagan** heteroscedasticity test
- **Lasso coefficient path**
- **Curvature tests in EDA:** mutual information (against a shuffled-target noise floor), the correlation ratio η² vs r², and quadratic-term tests with a **Bonferroni correction** across the 11 numeric features tested

### 5.6 Explainability

- Standardized coefficients (from `multiple_lr`)
- Lasso selection (which features survive the penalty)
- Permutation importance (how much test RMSE rises when a feature is shuffled)
- **SHAP**: summary beeswarm, mean |SHAP|, dependence plots, and grouped importance by Soil / Weather / Farming / Context
- A rank-agreement table across all four methods

---

## 6. Results in detail

### 6.1 Model comparison (Kaggle benchmark)

Ranked by CV RMSE, the criterion the deployed model was chosen by. **Selected by cross-validation; test set used once for final evaluation.**

| Model | CV RMSE (t/ha) | Train RMSE | Test RMSE (t/ha) | Test R² |
|---|---|---|---|---|
| **polynomial** (deployed) | **0.635 ± 0.009** | 0.618 | **0.646** | **0.690** |
| gradient_boost (XGBoost) | 0.636 ± 0.007 | 0.573 | 0.646 | 0.690 |
| lasso | 0.641 ± 0.005 | 0.637 | 0.653 | 0.684 |
| ridge | 0.641 ± 0.006 | 0.637 | 0.653 | 0.683 |
| multiple_lr | 0.641 ± 0.006 | 0.637 | 0.653 | 0.683 |
| random_forest | 0.646 ± 0.008 | 0.487 | 0.663 | 0.673 |
| simple_lr (fertilizer only) | 0.735 ± 0.006 | 0.735 | 0.773 | 0.557 |
| dummy_mean | 1.151 ± 0.015 | 1.151 | 1.160 | 0.000 |

XGBoost's test RMSE is 0.00006 t/ha lower than the polynomial's, a practical tie. It was not picked because the choice is made on CV before the test set is opened. The train/test gap is small for every model except Random Forest (0.49 vs 0.66), which fits the training fields closely.

![Model comparison](results/figures/model_comparison.png)

**Interpretation:**
- Every model beats the baseline by a wide margin.
- Polynomial regression and XGBoost are in a **statistical tie**. Flexible models gain only about 0.007 t/ha over linear ones, because the curved effects in the data are small and there are no interactions.
- XGBoost's tuning chose depth-2 trees, which is consistent with the absence of interactions.
- **Fertilizer alone explains 86 % of what all features explain together** (R² 0.59 vs 0.69).
- The residual standard deviation of about 0.64 t/ha appears to be the data generator's built-in noise. Residuals are normal and constant in spread, so this error is essentially irreducible, and no model can do much better.
- **GroupKFold by region** gives RMSE 0.633 (polynomial) and 0.639 (XGBoost), no worse than random folds, so nothing region-specific is memorised.

![Predicted vs actual](results/figures/pred_vs_actual_best.png)

### 6.2 Statistical diagnostics

| Diagnostic | Result | Meaning |
|---|---|---|
| VIF | 1.001–1.003 for every feature | No multicollinearity, so each coefficient is estimated independently. This is why Ridge ≈ OLS. |
| Breusch-Pagan | p = 0.41 | Residual variance is constant, so ordinary standard errors are valid. HC3 robust errors differ by at most 3.1 %. |
| Jarque-Bera | p = 0.60 | Residuals are consistent with a normal distribution |
| OLS R² / adjusted R² | 0.694 / 0.692 | |
| Lasso (α ≈ 0.003) | Keeps all 11 numeric features and zeroes 6 context dummies | Crop, region and season add nothing |

**Marginal vs partial effects.** Soil moisture, nitrogen, potassium and temperature look useless in raw correlations but are **highly significant once fertilizer is controlled for** (p < 10⁻⁷). Their effect is real but small, and the dominant fertilizer effect hides it in simple correlations.

### 6.3 What drives yield (Kaggle)

| Rank | Feature | Effect |
|---|---|---|
| 1 | Fertilizer amount | Strong positive, mild diminishing returns. Permutation importance: +0.73 t/ha RMSE when shuffled |
| 2 | Pesticide usage | Negative |
| 3 | Total rainfall | Positive, **saturating** around 600 mm |
| 4* | Soil pH | **Optimum at pH 6–7.5** (see below) |

**Grouped importance (mean |SHAP|):** Farming 64 % · Weather 17 % · Soil 16 % · Context 2 %

**The soil pH paradox.** This is the clearest single example for RQ3. Soil pH has a Pearson correlation of just **0.019**, is **not significant** in OLS, and ranks **13th** by linear coefficient. Yet SHAP ranks it **4th**. The reason is that yield peaks at pH 6–7.5 and falls on both sides (an inverted U). A straight line through a peak has zero slope, so **linear models report that pH doesn't matter when it clearly does.** Only the non-linear models and SHAP dependence plots reveal the effect.

**Method agreement:** Spearman ρ is 0.97 between the two coefficient-based methods and 0.91 between permutation and SHAP. The two families disagree mainly on soil pH (explained above) and crop type, whose coefficient score is inflated by taking the maximum of several noisy dummy coefficients.

### 6.4 Real-world validation (India, 2016–2020 test years)

| Model | Test RMSE | Test R² |
|---|---|---|
| **Crop×state lookup (no ML)** | **7.81** | **0.68** |
| random_forest | 8.12 | 0.65 |
| gradient_boost (XGBoost) | 8.14 | 0.65 |
| linear models (OLS, Ridge, Lasso) | 9.51–9.52 | 0.52 |
| polynomial (degree 2) | 9.72 | 0.50 |
| polynomial (degree 3) | 10.02 | 0.47 |
| simple_lr | 14.01 | −0.04 |
| dummy_mean | 14.04 | −0.05 |

*Linear-model penalties are tuned on time-ordered folds (Phase 1). Before that fix, with folds that let later years inform α, the polynomial scored 9.48 and degree 3 9.84; Ridge is unchanged and Lasso improved from 9.55 to 9.52. RMSE is in the dataset's own yield units (mostly t/ha, but bales per hectare for cotton, jute, mesta and sannhemp). The spread is large because yields range across 55 very different crops. Full values for every model are in `results/tables/india_test.csv`.*

**What the real data says:**
- **Crop type dominates:** SHAP shares are Crop 83 %, State 5.5 %, log(Area) 4.9 %, Rainfall 4 %, Fertilizer 0.9 %, Pesticide 0.5 %.
- **Tree models clearly beat linear models,** because they can encode crop×state combinations.
- **A simple historical-average lookup beats every model.** Most of the predictable signal is "what does this crop usually yield in this state," which a lookup table captures directly.
- **Within a single crop, the models largely fail.** Per-crop R² is negative for 8 of the 10 most common crops. The pooled R² of 0.65 mostly reflects telling crops apart, not predicting good and bad seasons for a given crop.

### 6.5 Ablation: do India's fertilizer and pesticide columns matter?

Because these columns are really a national per-year index (section 4B), the full India suite was re-run **without** them, with the same split, seeds and tuning.

| Models | Change in test RMSE without the columns | Verdict |
|---|---|---|
| Linear (OLS, Ridge, Lasso) | +0.27 (about 3 % worse) | Help modestly, acting as an extrapolated time trend |
| Polynomial degree 2 | +0.07 | Help a little (its time-ordered penalty is stronger) |
| Random Forest, XGBoost | +0.016 / +0.019 (about 0.2 %) | Essentially no effect |
| Polynomial degree 3 | −0.24 | Hurt (overfitting) |

**Why trees ignore them:** 100 % of test rows have pesticide values above the training maximum, and 26 % have fertilizer values above it. Tree models cannot extrapolate beyond the range they were trained on. **Every headline India conclusion holds with or without these columns.**

### 6.6 Answering the research questions

- **RQ1:** Yield is predictable to about ±0.65 t/ha (R² ≈ 0.69) on the benchmark. That is close to the data's irreducible noise level.
- **RQ2:** On the benchmark: fertilizer, then pesticide, then rainfall, with consistent agreement across four methods. On real data: crop type dominates everything.
- **RQ3:** The relationship is mostly linear with mild curvature (a pH optimum, rainfall saturation, fertilizer diminishing returns) and no interactions. Linear models are nearly optimal on the benchmark but completely miss the pH effect.
- **RQ4:** The conclusions largely **do not transfer**. Real yields are driven by crop and location, flexible models are needed, and a simple lookup is the benchmark to beat.

---

## 7. The interactive demo app

A Streamlit app (`app.py` plus one script per page in `app_pages/`) makes the model explorable without any code, including for readers with no technical background. The design is "Field & Harvest": parchment and crop-green in light mode, a deep field-green in dark mode, with Manrope headings, Inter text and IBM Plex Mono numbers. The fonts are bundled in `static/fonts/` (OFL), and the app makes **no request to any other server**, so it looks the same on an offline projector laptop.

- **Pages** (top navigation): Overview · Predict · What-if · Model comparison · Data explorer · India.
- **Simple / Technical** switch at the top right, kept across pages. Simple uses plain words and everyday names ("Typical error" instead of RMSE); Technical shows the exact metrics, tables and SHAP waterfall.
- **Light and dark mode** from the ⋮ menu (Light / Dark / System). The first visit opens in light mode, which is safer on projectors; the menu choice is remembered.
- **Responsive** from 768 to 1920 px. KPI cards reflow 4 → 2 → 1 per row and their numbers never wrap; the navigation folds extra pages into a "more" menu on narrow screens.
- Every page opens with an **"In simple words"** card generated from the saved numbers (`ui/plain.py`), and every name and unit comes from one mapping (`ui/labels.py`), so no column name such as `soil_ph` ever appears.

| Page | What it shows |
|---|---|
| **Overview** | The question, the three headline numbers read from the results tables (best CV RMSE, test RMSE, test R²) plus the 90 % range, a "how it works" pipeline strip (data → preprocess → 8 models → cross-validate → test once → explain), the data sources with links, and a glossary. |
| **Predict** | Inputs grouped and colour-coded (soil · weather · farming · context), scenario presets, the prediction with its **90 % range** (gauge ring in Simple mode), "Why this prediction" (SHAP bars, or an interactive **SHAP waterfall** in Technical mode), and **Download CSV / PDF** report buttons. |
| **What-if** | The prediction for the current field as one factor sweeps its range, with the 90 % band shaded; what drives yield overall (factor groups, feature ranking, response curves, and an interactive SHAP beeswarm of the 960 test fields in Technical mode). |
| **Model comparison** | Models ranked by CV RMSE with ± 1 sd error bars, predicted-vs-real and residual plots for the deployed or the best tree model, KPIs, and the full CV / train / test table. |
| **Data explorer** | The key EDA from notebook 01 as interactive charts: yield and input distributions, a Pearson / Spearman correlation heatmap, yield by crop, region and season, and yield by harvest month (needs the raw Kaggle CSV; a setup card explains how to get it). |
| **India** | The real-world check: lessons, look-up vs best model vs guessing, per-crop verdicts (Simple) and all five India tables (Technical). |

![Predict page, Simple mode, light](screenshots/phase2/simple_light_predict_1440.png)

![Predict page, Technical mode, dark](screenshots/phase2/technical_dark_predict_1440.png)

![Model comparison, Technical mode, light](screenshots/phase2/technical_light_model_comparison_1440.png)

### The 90 % prediction range

Every prediction is shown with a range such as **6.52 t/ha (90 % range 5.48–7.56)**. It is a split-conformal interval: each of the 3,840 training fields is predicted by the deployed model trained on the other four CV folds, and the half-width is the ⌈(n + 1) · 0.9⌉-th smallest absolute error, **± 1.037 t/ha**. The pipeline computes it once (`results/tables/conformal.json`); the app only reads it. On the 960 test fields it covered **88.5 %** of the real yields, 1.5 points under the 90 % target and within normal sampling variation for 960 fields (about ± 1 point). The range has the same width for every field.

**Things to try:**
- Click the **scenario presets**: Ideal field predicts 9.15 t/ha, Drought year 5.99, Low-input farm 5.51, and Typical field returns to 6.52.
- Move **fertilizer** from minimum to maximum. The prediction swings from about 4.70 to 7.76 t/ha.
- Open **"What if I change…"** and choose Soil pH. The curve peaks around pH 6.6.
- Move **irrigation frequency** across its whole range. The prediction barely changes (6.54–6.58 t/ha).
- In **Technical** mode, set **soil pH** to 4.5, then 6.7, then 8.5. Its SHAP bar goes from −0.19 to +0.08 and back to −0.16.
- On **What-if** (Simple mode), choose Rainfall under "How yield responds". The model's predicted yield rises, then levels off above about 805 mm.
- Press **Reset to defaults** to return to 6.52 t/ha.

The app loads saved models and never retrains. Measured in Phase 2 (headless Chrome at 1366×768, cold server): the first page (Overview) appears in about 9 s, the first Predict page in about 9.5 s (SHAP set-up), later page loads take 0.6 s (Overview) to 1.7 s (Predict), and a scenario preset updates the prediction in about 0.4–0.5 s (fully settled in about 1 s). The Phase 1 app, measured the same way on the same day, took 20.2 s, 1.8–2.8 s and 1.7–1.9 s. It uses Kernel SHAP instead of the permutation explainer, which avoids a roughly 20-second compile delay on first use. The design and verification records are in [`UI_REDESIGN_REPORT.md`](UI_REDESIGN_REPORT.md) (first redesign) and [`UI_V2_REPORT.md`](UI_V2_REPORT.md) (Simple / Technical modes).

---

## 8. Project structure

```
CropYieldPrediction/
├── data/
│   ├── raw/kaggle/            # Competition CSVs (downloaded; git-ignored)
│   ├── raw/india/             # Indian states CSV (manual download; git-ignored)
│   └── processed/             # Train/test parquet splits
├── notebooks/
│   ├── 01_eda.ipynb           # Distributions, correlations, curvature tests, synthetic-data evidence
│   ├── 02_diagnostics.ipynb   # VIF, OLS inference, residuals, Breusch-Pagan, Lasso path
│   ├── 03_explain.ipynb       # Coefficients, permutation importance, SHAP, agreement table
│   └── 04_india.ipynb         # Real-data validation, per-crop analysis, lookup baseline, ablation
├── src/
│   ├── config.py              # Single source of truth: seeds, paths, column names, units
│   ├── data.py                # Loading, cleaning, train/test split, CV splitter
│   ├── eda.py                 # Correlation, mutual information, curvature statistics
│   ├── plots.py               # Figure saving and styling
│   ├── features.py            # Preprocessing (ColumnTransformer, polynomial expansion)
│   ├── models.py              # All model pipelines and tuning setups
│   ├── evaluate.py            # Cross-validation and hold-out metrics
│   ├── explain.py             # Permutation importance and SHAP helpers
│   ├── india.py               # India cleaning, time-split CV, lookup baseline, ablation
│   └── pipeline.py            # End-to-end Kaggle pipeline (python -m src.pipeline)
├── tests/                     # pytest: data integrity, leakage guards, models, SHAP, India, conformal interval, display names, theme contrast, report, every app page (AppTest)
├── results/
│   ├── figures/               # Every plot (eda_*, diag_*, explain_*, india_*, model comparison)
│   └── tables/                # Every results table (CSV/JSON)
├── models/                    # Saved pipelines. best_overall.joblib (app) and best_tree.joblib (notebook 03) are committed; the India model is git-ignored. Regenerated by the pipelines
├── submissions/               # Local Kaggle submission file (never uploaded)
├── app_pages/                 # One script per page: overview, predict, whatif, models, explorer, india
├── ui/
│   ├── theme.py               # Design tokens (light + dark), WCAG contrast checks, CSS
│   ├── labels.py              # Display names and units for every column, model and group
│   ├── components.py          # Cards, KPI grid, setup cards, results tables
│   ├── charts.py              # Altair charts coloured by theme series (follow light/dark live)
│   ├── data.py                # Cached loaders, input state, presets, curves
│   ├── report.py              # Prediction report as CSV and one-page PDF
│   └── plain.py               # Plain-language sentences and tooltips, generated from the real numbers
├── static/fonts/              # Inter, Manrope, IBM Plex Mono (woff2, OFL-1.1) served by the app
├── .streamlit/
│   └── config.toml            # Light + dark themes from ui/theme.py, bundled fonts, no file watcher
├── docs/
│   ├── VIVA_NOTES.md          # Likely examiner questions with short answers
│   └── screenshots/           # Earlier app versions (v2_, archive/); current ones are in screenshots/phase2/
├── app.py                     # Streamlit entry point: theme, header, mode switch, navigation
├── run.py                     # One-command entry point: set up what's missing, then launch (dev/build/check/app)
├── start.bat                  # Double-click launcher for Windows; runs python run.py
├── requirements.txt           # Runtime packages, pinned, cross-platform (Windows, Linux, macOS, Colab, Streamlit Cloud)
├── requirements-dev.txt       # + Jupyter, ruff, pytest, Kaggle client (same pins)
├── requirements-lock-windows.txt # Full Windows lock that produced the reported results
├── CHANGELOG.md               # Mid-term → end-sem changes, then per phase
├── PROGRESS.md                # Audit, roadmap and per-phase measured results
├── CROP_YIELD_BUILD_REPORT.md # Full technical report with verification evidence
├── UI_REDESIGN_REPORT.md      # Dashboard redesign: before/after screenshots, scroll measurements, checks
├── UI_V2_REPORT.md            # Simple/Technical modes: sentence-to-number checks, fit measurements
└── CODE_REVIEW.md             # Self-review findings and viva Q&A
```

**Key result tables** in `results/tables/`:

| File | Contents |
|---|---|
| `cv_all_models.csv` / `test_results.csv` | Kaggle CV and final test metrics (test_results is in CV order and includes train RMSE) |
| `cv_groupkfold.csv` | Leave-one-region-out check |
| `best_params.json` | Tuned tree hyperparameters |
| `feature_signal.csv` | Pearson, Spearman and mutual information per feature |
| `vif.csv` / `ols_summary.csv` | Multicollinearity and OLS inference |
| `importance_agreement.csv` | Feature ranks under all four importance methods |
| `india_cv.csv` / `india_test.csv` | India metrics |
| `india_reference_baselines.csv` | Crop×state lookup comparison |
| `india_per_crop_rmse.csv` | Per-crop error breakdown |
| `india_ablation.csv` / `india_feature_ranges.csv` | Ablation study and range check |
| `india_shap_shares.csv` | India best model's mean \|SHAP\| and share per feature (crop type 83 %) |

---

## 9. How to reproduce everything

### Quick start

**Double-click `start.bat`.** That's the whole setup: the Python equivalent of `npm run dev`. It needs Python 3.12+ installed and a Kaggle API token (Step 2 below). It works from anywhere, with no virtual environment to activate.

On the first run it creates `.venv`, installs the locked package versions, downloads the Kaggle data, trains the models, runs the India validation (if the India CSV is present), and opens the app at http://localhost:8501. Later runs skip everything that already exists and go straight to the app. If anything is missing or fails, it stops, names the step, and prints the exact command to re-run just that step.

| Command | What it does |
|---|---|
| `start.bat` or `python run.py` | **dev:** set up whatever is missing (environment, data, models, India results), then launch the app |
| `python run.py build` | Force a full rebuild: Kaggle pipeline, India validation (if the CSV is present), then execute all 4 notebooks |
| `python run.py check` | Lint (`ruff`) and run the tests (`pytest`) |
| `python run.py app` | Launch the app only |
| add `--no-browser` | Start the app server without opening a browser tab (`dev` and `app`) |

`start.bat` passes arguments through, so `start.bat check` equals `python run.py check`. Stop the app with Ctrl+C.

### Manual steps (what start.bat does for you)

Tested on Windows 11 with Python 3.14.4. The pinned packages need **Python 3.12 or newer** (3.11 fails on `contourpy==1.4.0`). All models run on CPU; no GPU is needed.

### Step 1 — Clone and set up the environment

Three requirement files, all with the same pinned versions:

| File | Use it for |
|---|---|
| `requirements.txt` | Running the app, pipelines and notebooks on any platform (Windows, Linux, macOS, Colab, Streamlit Cloud) |
| `requirements-dev.txt` | The above plus Jupyter, ruff, pytest and the Kaggle client (development) |
| `requirements-lock-windows.txt` | The exact full Windows environment that produced the reported numbers |

**Windows:**

```bat
git clone https://github.com/SauravLamba17/CropYieldPrediction.git
cd CropYieldPrediction
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements-lock-windows.txt
```

**Linux / macOS:**

```bash
git clone https://github.com/SauravLamba17/CropYieldPrediction.git
cd CropYieldPrediction
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

`python run.py` does this for you and picks the right file for the platform. On Linux, XGBoost also pulls in NVIDIA's NCCL library (`nvidia-nccl-cu13`, not pinned); it is unused on CPU.

The trained Kaggle models are committed, so after this step `python -m streamlit run app.py` (or `python run.py app`) opens the dashboard straight away; Steps 2–5 are only needed to rebuild the results.

### Run the notebooks in Google Colab

Open a notebook from GitHub in Colab (File → Open notebook → GitHub, paste the repository URL) and run the first code cell, **Setup (Colab)**. It clones the repository into `/content/CropYieldPrediction`, installs `requirements.txt`, and switches to that folder; everywhere else the cell only locates the project root. If Colab asks to restart the runtime after the install, restart and run the cell again. Notebooks 01–03 need nothing else: the train/test split and the models are in the repository. Notebook 04 also needs the India CSV: upload it to `data/raw/india/crop_yield.csv` in the cloned folder.

### Step 2 — Kaggle access

1. Create a Kaggle account and **accept the rules** on the [competition page](https://www.kaggle.com/competitions/crop-yield-prediction-challenge). Downloads fail with a 403 error otherwise.
2. Generate an API token under Kaggle → Settings → API. It starts with `KGAT_`.
3. Save it in a file named `access_token` (no extension) at `C:\Users\<your-username>\.kaggle\access_token`.

Never place the token inside the project folder.

### Step 3 — Download the data

```bat
python -m kaggle competitions download -c crop-yield-prediction-challenge -p data/raw/kaggle
```

Unzip the archive into `data/raw/kaggle/` (`run.py` does this automatically). The zip can be deleted afterwards.

For the India data, download the CSV from https://data.mendeley.com/datasets/ncw2vbcgnk ("Download All"), extract it, and place the CSV at `data/raw/india/crop_yield.csv` (the code expects exactly this file name, set in `src/config.py`).

### Step 4 — Run the pipelines

```bat
python -m src.pipeline
python -m src.india
```

The first command runs the Kaggle pipeline end to end: split, cross-validation, nested tuning, the one-time test evaluation, figures and saved models. The second runs the India validation, including the ablation. Nested cross-validation is CPU-heavy. Actual timings from the final verification run (i7 13th-gen laptop, CPU only):

| Step | Time |
|---|---|
| `python -m src.pipeline` | 12 min 37 s |
| `python -m src.india` (including the ablation and the time-ordered penalty search) | 22 min 40 s |
| 4 notebooks (`run.py build`, step 5) | 3 min 57 s |
| **Full `python run.py build`** | **39 min 15 s** |

Measured in Phase 1 on a fresh clone (Windows 11, Python 3.14.4, 16 logical CPUs). Installing `requirements-dev.txt` into a fresh venv took 5 min 59 s. The India tree searches take the longest.

### Step 5 — Execute the notebooks

```bat
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb notebooks/02_diagnostics.ipynb notebooks/03_explain.ipynb notebooks/04_india.ipynb
```

Or open them in Jupyter and run them top to bottom; they work with the working directory at the repo root or at `notebooks/`. Measured in Phase 1: from `notebooks/` 01 took 53 s, 02 18 s, 03 21 s and 04 143 s (234 s in total); from the repo root 62 s, 21 s, 28 s and 159 s (270 s). Notebook 04 is the slowest (mostly the Random Forest SHAP computation).

### Step 6 — Launch the app

```bat
python -m streamlit run app.py
```

It opens at http://localhost:8501. The app does not reload itself when code changes (the file watcher is off for speed); restart it after editing.

### Step 7 — Run the checks

```bat
python -m ruff check src tests ui app_pages app.py run.py
python -m pytest -q
```

---

## 10. Verification and reproducibility

- **Deterministic:** every random process uses `RANDOM_STATE = 42`. Two independent full pipeline runs produced **byte-identical** `test_results.csv`, `best_params.json` and `submission.csv` (verified by file hash).
- **132 automated tests** (none of which write to tracked files) cover split integrity, the preprocessor being fitted only on training data, `Production` never reaching an India model, the test years staying strictly after the training years, per-hectare conversions, SHAP additivity, the ablation logic, unseen categories, the leaderboard's CV ordering, the notebooks' setup cell (from the repo root and from `notebooks/`), and the app's plain-language sentences (correct rounded numbers, curve shapes, no t/ha on India), the conformal interval, display names for every column, theme contrast and config sync, the report, and every app page in both modes (Streamlit AppTest).
- **Leakage guard:** every result was checked against a red-flag threshold of R² > 0.98 in the verification checklist. The code has no automatic stop. The best observed R² was 0.77.
- **Notebooks** execute top to bottom with zero errors, and every claim in their text was checked against the printed output.
- **Self-review:** a line-by-line review of the whole codebase found 0 critical, 1 high, 4 medium and 12 low issues. All critical and high findings were resolved. See `CODE_REVIEW.md`.

**Documented deviation:** Random Forest and XGBoost run single-threaded inside a parallel hyperparameter search. The original nested parallelism produced about 2,200 warnings with no speed benefit. Changing XGBoost's thread count alters its floating-point summation order, which shifted its test RMSE by 0.0002 (0.6459 → 0.6461). Each setting is exactly reproducible on its own, and model selection was unaffected.

---

## 11. Limitations

1. **The Kaggle dataset is synthetic.** Its importance rankings describe the data generator, not real farms. The India validation exists to address this.
2. **The India data lacks soil measurements** (pH, N, P, K and moisture are absent), and its fertilizer and pesticide columns are national yearly estimates, not farm measurements. The fertilizer finding therefore cannot be tested on real data.
3. **The India data has quality issues:** 82 physically implausible yield records were kept and reported rather than silently removed.
4. **State-level aggregation.** India yields are averaged over whole states, which hides the field-level variation a farmer would care about.
5. **India SHAP uses a 300-row sample** of the test years, because the Random Forest explainer is single-threaded and too slow on the full set.
6. **India units are mixed.** Most crops are in tonnes per hectare, but cotton, jute, mesta and sannhemp are counted in bales per hectare, so the pooled India RMSE has no single unit. The app labels it "yield units (varies by crop)".
7. **Unseen categories.** A crop, state or season never seen in training is one-hot encoded as all zeros (`handle_unknown="ignore"`), so the model falls back to its baseline for that column instead of failing. On India this affects 10 test rows ("Other Summer Pulses", which only appears in 2016–2020). The crop × state look-up falls back to the overall training mean.
8. **Platform.** The pinned versions need Python 3.12+. Dependencies were verified to resolve for Linux (Python 3.12 and 3.14) and installed and run on Windows; the app has not yet been deployed to Streamlit Cloud (Phase 5).
9. **India penalty tuning.** After making India's penalty tuning time-ordered (Phase 1), degree-2 polynomial no longer beats plain linear regression on the test years (9.72 vs 9.51).
10. **The 90 % range is not adaptive.** It has the same width for every field and covered 88.5 % of the test fields (target 90 %).
11. **Associations, not causes.** This is observational data, so the results show which variables predict yield, not what would happen if a farmer changed them.

---

## 12. Further documentation

- **[`CROP_YIELD_BUILD_REPORT.md`](CROP_YIELD_BUILD_REPORT.md):** the full technical report, with every number, the figure index, verification evidence as real command output, and all deviations.
- **[`CODE_REVIEW.md`](CODE_REVIEW.md):** every review finding with its severity and fix, plus the anticipated examiner questions with answers grounded in the results.

---

*Built as an academic AIML project. Datasets are used under their respective licences: Kaggle competition rules, and CC BY 4.0 for the Indian states dataset.*
