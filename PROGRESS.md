# PROGRESS — Phase 0 audit and showcase plan

CSL236 AIML course project, Team 2: crop yield prediction (t/ha).
Audit date: 2026-11-14. Commit audited: `511cdfe`.

Every number in this file was produced in this session by a fresh `git clone` of the repo, with a fresh virtual environment built from `requirements-lock.txt` (Python 3.14.4, Windows 11, 16 logical CPUs). Nothing was copied from the old outputs. The original working folder was only read: no code, notebook, data or result file in it was changed.

---

## 1. Project summary

The app predicts crop yield in tons per hectare from 14 inputs:

- **Soil:** pH, moisture, N, P, K
- **Weather:** temperature, rainfall, sunlight
- **Farming:** fertilizer, pesticide, irrigation
- **Context:** crop, region, season

The data is the Kaggle *Crop Yield Prediction Challenge* (4,800 labelled rows). The README already calls this dataset synthetic, with the evidence. To check whether the findings hold on real data, every model is re-run on **real Indian state-level data, 1997–2020**: Mendeley `ncw2vbcgnk`, 19,412 clean rows, split by time.

The suite has 8 models: mean baseline, simple LR, multiple LR, polynomial+Ridge, Ridge, Lasso, Random Forest and XGBoost. The tree models are tuned with nested CV. Explainability uses coefficients, permutation importance and SHAP. A 5-tab Streamlit dashboard comes with Simple and Technical modes.

**The repo is well past the mid-term state given in the Phase 0 brief:**

- Every scaler and encoder already sits inside an sklearn `Pipeline` / `ColumnTransformer`, so the CV-leakage weakness is already fixed (see §4.3).
- There is **no Lasso 25→20 feature-selection step** anywhere in the code.
- The git history has a single commit, so the mid-term notebook is not in this repo.

### Flow

```
data/raw/kaggle/crop_yield_train.csv (4,800 rows)          data/raw/india/crop_yield.csv (19,689 rows)
        │ src/data.py: load_kaggle_train()                          │ src/india.py: clean_india()
        │   drop id/harvest_date/field_id, dedupe                   │   drop Production (target leak), per-ha inputs,
        │ make_split(): 80/20 shuffle, seed 42                      │   log1p(Area), remove Coconut and yield<=0
        ▼                                                           ▼ time_split(): train ≤2015 / test ≥2016
data/processed/kaggle_{train,test}.parquet (3,840 / 960)    data/processed/india_{train,test}.parquet (15,186 / 4,226)
        │                                                           │
        ▼ src/features.py: build_preprocessor()  (StandardScaler + OneHot [+ PolynomialFeatures]) — inside each Pipeline
        ▼ src/models.py: get_linear_models() + get_nonlinear_models() (RandomizedSearchCV → nested CV)
        ▼ src/evaluate.py: cross_validate_models()  5-fold KFold (Kaggle) / YearBlockSplit (India)
        ▼ src/pipeline.py: main()  → best model chosen by CV → ONE test evaluation (run_final_test)
        │                    → results/tables/*.csv, results/figures/*.png, models/*.joblib, submission.csv
        ▼ notebooks/01_eda → 02_diagnostics (VIF, OLS, residuals) → 03_explain (coef, permutation, SHAP) → 04_india
        ▼ app.py (Streamlit): loads models/best_overall.joblib + tables → Predict · Model comparison ·
                              What drives yield · Real-world validation · How it works   (Kernel SHAP per prediction)
```

| Step | Where |
|---|---|
| Load and clean | `src/data.py:load_kaggle_train` (drops `DROP_COLUMNS`, dedupes); `src/india.py:clean_india` |
| Split | `src/data.py:make_split` (80/20, `RANDOM_STATE=42`); `src/india.py:time_split` |
| Preprocessing | `src/features.py:build_preprocessor` (unfitted `ColumnTransformer`, fitted only inside each Pipeline) |
| Models | `src/models.py:get_linear_models`, `get_nonlinear_models` (20-iteration `RandomizedSearchCV`) |
| CV and test | `src/evaluate.py:cross_validate_models`, `holdout_metrics`; `src/pipeline.py:run_cv`, `run_group_cv`, `run_final_test` |
| Model selection | `src/pipeline.py:best_names` (lowest CV RMSE, chosen before the test set) |
| EDA | `notebooks/01_eda.ipynb` + `src/eda.py` (feature_signal, linearity / quadratic tests) |
| Diagnostics | `notebooks/02_diagnostics.ipynb` (VIF, statsmodels OLS, Breusch-Pagan, Jarque-Bera, Lasso path) |
| Explainability | `notebooks/03_explain.ipynb` + `src/explain.py` (coefficients, permutation, TreeExplainer, `RawFeatureExplainer` Kernel SHAP) |
| India validation | `src/india.py:main` (CV, test, ablation, crop×state lookup) + `notebooks/04_india.ipynb` |
| UI | `app.py` + `ui/theme.py` (CSS), `ui/charts.py` (Altair), `ui/plain.py` (generated plain-language text) |
| One-command run | `run.py` (`dev` / `build` / `check` / `app`), `start.bat` |

---

## 2. Current state (verified this session)

| What | Result | Runtime |
|---|---|---|
| `py -3.14 -m venv` + `pip install -r requirements-lock.txt` | ❌ **first attempt failed**: `OSError` on a jupyterlab file whose path exceeded Windows MAX_PATH (260 characters). Passed after remapping the scratch folder to a short drive letter (`subst`) | 5 min 17 s (failed) / 6 min 52 s (passed) |
| `pip check` | No broken requirements | – |
| App on a fresh clone **before** training | Loads, but shows only `Missing files: ['models\best_overall.joblib']` (the `.joblib` files are git-ignored) | – |
| `python run.py build` (full rebuild) | ✅ exit 0 | **45 min 52 s** total |
| ↳ `python -m src.pipeline` (Kaggle) | ✅ | 12 min 49 s |
| ↳ `python -m src.india` (India + ablation) | ✅ | 26 min 01 s |
| ↳ 4 notebooks via `nbconvert --execute` | ✅ 0 errors, 0 stderr outputs | 7 min 01 s |
| &nbsp;&nbsp;`01_eda` / `02_diagnostics` / `03_explain` / `04_india` | 16 / 8 / 9 / 17 code cells, all ran | 63.7 s / 20.3 s / 27.5 s / 277.7 s |
| &nbsp;&nbsp;slowest cell | `04_india` code cell 15 (Random Forest SHAP on India) | 258.8 s |
| `python run.py check` | ✅ ruff: all checks passed; **pytest: 31 passed** | 2.5 s + 17.9 s |
| Reproducibility vs committed outputs | `test_results.csv`, `best_params.json`, `submission.csv`, `model_names.json`, `app_metadata.json`, `india_test.csv` **byte-identical**. In the CV tables only the `fit_time_s` column changed; every metric is identical. `explain_shap_beeswarm.png` differs pixel-wise (see L4) | – |
| Audit metrics script (train + test refit of every model) | ✅ | 8 min 40 s |
| Streamlit app (built clone) | ✅ all 5 tabs × 2 modes render, no console errors | first render ≈ 5.1 s; preset click → re-render 2.2 s |

---

## 3. File inventory

Status key: **used**; **output** (generated by the code, read by the app or reports); **docs**; **unused / dead**; **duplicate**.

| File | Purpose | Status |
|---|---|---|
| `README.md` | Main documentation (34 KB): results, setup, app tour | used, but its timing claims are stale (I8) |
| `CROP_YIELD_BUILD_REPORT.md` | Full technical report with verification evidence | docs |
| `CODE_REVIEW.md` | Self-review findings + likely viva Q&A | docs, very useful for the viva |
| `UI_REDESIGN_REPORT.md` | Record of the v1 "Cyber Sand" redesign | docs, superseded by v2 |
| `UI_V2_REPORT.md` | Record of the v2 Simple/Technical modes | docs |
| `app.py` | Streamlit dashboard (1,018 lines) | used |
| `run.py` | One-command launcher (stdlib only) | used |
| `start.bat` | Windows double-click launcher → `run.py` | used |
| `requirements.txt` | Unpinned direct dependencies | used (not by the launcher) |
| `requirements-lock.txt` | Exact pins (Python 3.14, Windows-only `pywinpty`) | used by `run.py`; not portable (I2) |
| `pyproject.toml` | ruff + pytest config | used |
| `.streamlit/config.toml` | Theme, forced light mode, toolbar | used |
| `.gitignore` | Ignores `.venv`, `data/raw`, `*.joblib`, caches | used |
| `src/__init__.py` | Package marker | used |
| `src/config.py` | Paths, seeds, column names, units | used |
| `src/data.py` | Load, dedupe, split, KFold | used |
| `src/features.py` | `ColumnTransformer` builder | used |
| `src/models.py` | All model pipelines + search spaces | used |
| `src/evaluate.py` | CV + hold-out metrics | used |
| `src/eda.py` | Correlation, MI, linearity tests | used |
| `src/explain.py` | Coefficients, permutation, SHAP helpers | used |
| `src/india.py` | India cleaning, YearBlockSplit, lookup, ablation | used |
| `src/pipeline.py` | End-to-end Kaggle run | used |
| `src/plots.py` | `save_figure`, axis labels | used |
| `ui/theme.py` | Palette + CSS (no responsive rules) | used |
| `ui/charts.py` | Altair charts | used |
| `ui/plain.py` | Plain-language sentences/tooltips | used |
| `tests/test_data.py` | Split integrity (writes the parquet files, I10) | used |
| `tests/test_models.py` | Pipelines fit, no preprocessor leakage | used |
| `tests/test_explain.py` | SHAP additivity, readable names | used |
| `tests/test_india.py` | Leakage, time split, per-ha, ablation | used |
| `tests/test_plain.py` | App sentences contain the right numbers | used |
| `notebooks/01_eda.ipynb` | EDA, curvature tests, synthetic-data evidence | used |
| `notebooks/02_diagnostics.ipynb` | VIF, OLS, residuals, Breusch-Pagan, Lasso path | used |
| `notebooks/03_explain.ipynb` | Coefficients, permutation, SHAP, agreement table | used |
| `notebooks/04_india.ipynb` | Real-data validation, per-crop, lookup, SHAP | used |
| `data/raw/kaggle/crop_yield_train.csv` | Labelled Kaggle data (git-ignored) | used |
| `data/raw/kaggle/crop_yield_test.csv` | Unlabelled competition test file | used (submission only) |
| `data/raw/kaggle/sample_submission.csv` | Submission format check | used |
| `data/raw/kaggle/crop-yield-prediction-challenge.zip` | Original download archive | duplicate of the 3 CSVs |
| `data/raw/india/crop_yield.csv` | India data (git-ignored) | used |
| `data/processed/kaggle_{train,test}.parquet` | Saved 80/20 split | output (app reads it) |
| `data/processed/india_{train,test}.parquet` | Saved time split | output |
| `models/best_overall.joblib` | Polynomial+Ridge pipeline (8 KB), the app's model | used |
| `models/best_linear.joblib` | Same model, byte-identical | duplicate, nothing loads it |
| `models/best_tree.joblib` | XGBoost pipeline (770 KB) | used by notebook 03 |
| `models/india_best_overall.joblib` | India Random Forest (79 MB) | used by notebook 04 |
| `models/india_best_tree.joblib` | Same model, byte-identical (79 MB) | duplicate, nothing loads it |
| `models/model_names.json`, `india_model_names.json`, `app_metadata.json` | Best-model names, slider ranges | used by the app |
| `results/tables/*.csv/json` (17 files) | All metrics | output. The app reads 9 of them; `cv_linear.csv` is a subset of `cv_all_models.csv` |
| `results/figures/*.png` (28 files) | All static figures | output. The app shows only `explain_shap_beeswarm.png`; the rest are for README and notebooks |
| `submissions/submission.csv` | Local Kaggle submission (never uploaded) | output, not used by the demo |
| `docs/screenshots/*.png` (28 files) | v1 before/after + v2 screenshots | docs. `before_*`, `after_*`, `v1_*` are historical; only `v2_*` appear in the README |
| `__pycache__/run.cpython-314.pyc` (root) | Stale bytecode, git-ignored | dead |
| `.venv/` (yours, untouched) | Existing environment | used locally, git-ignored |
| `.claude/launch.json` | Preview config for Claude tooling, git-ignored | tooling |
| `screenshots/` **(new)** | 40 audit screenshots from this phase | docs |
| `PROGRESS.md` **(new)** | This file | docs |

---

## 4. Results from this run

### 4.1 Kaggle benchmark (train 3,840 / test 960; 5-fold shuffled KFold on train, seed 42)

| Model | Train RMSE | Test RMSE | Test R² | Test MAE | 5-fold CV RMSE (mean ± std) |
|---|---|---|---|---|---|
| gradient_boost (XGBoost, nested-tuned) | 0.5730 | **0.6461** | **0.6899** | 0.5164 | 0.6362 ± 0.0073 |
| **polynomial** (deg 2 + RidgeCV), *selected by CV and deployed* | 0.6183 | 0.6462 | 0.6898 | **0.5141** | **0.6349 ± 0.0088** |
| lasso (LassoCV) | 0.6371 | 0.6526 | 0.6836 | 0.5230 | 0.6406 ± 0.0052 |
| ridge (RidgeCV) | 0.6368 | 0.6533 | 0.6830 | 0.5238 | 0.6408 ± 0.0055 |
| multiple_lr (OLS) | 0.6368 | 0.6533 | 0.6829 | 0.5236 | 0.6408 ± 0.0055 |
| random_forest (nested-tuned) | 0.4870 | 0.6635 | 0.6730 | 0.5290 | 0.6456 ± 0.0083 |
| simple_lr (fertilizer only) | 0.7347 | 0.7725 | 0.5567 | 0.6226 | 0.7352 ± 0.0062 |
| dummy_mean (baseline) | 1.1508 | 1.1604 | −0.0003 | 0.9285 | 1.1509 ± 0.0148 |

- **Leave-one-region-out (GroupKFold):** polynomial 0.6334 ± 0.0066, XGBoost 0.6385 ± 0.0047. These are no worse than random folds.
- **Tuned parameters:**
  - XGBoost: max_depth 2, 847 trees, learning rate 0.0356
  - Random Forest: max_depth 16, min_samples_leaf 8, 587 trees
  - Ridge α: polynomial 44.98, ridge 25.60
  - Lasso α: 0.0029 when refit on the full training split (CV-fold median 0.0038)
- **Random Forest overfits:** a 0.18 t/ha train–test gap. Every model sits near a residual σ of about 0.64 t/ha. That looks like the noise built into the data generator (the diagnostics show normal, constant-variance residuals).

### 4.2 India real-world validation (train 1997–2015, 15,186 rows / test 2016–2020, 4,226 rows; 5 time-ordered year-block folds; log1p target, metrics in original units)

| Model | Train RMSE | Test RMSE | Test R² | Test MAE | Time-ordered CV RMSE (mean ± std) |
|---|---|---|---|---|---|
| random_forest | 8.2058 | **8.1195** | **0.6499** | 1.5800 | 5.8855 ± 2.9620 |
| gradient_boost (XGBoost) | 8.5627 | 8.1358 | 0.6485 | **1.4002** | 6.0125 ± 2.9173 |
| polynomial | 11.3477 | 9.4807 | 0.5227 | 2.1801 | 7.9156 ± 2.3973 |
| multiple_lr | 11.3455 | 9.5082 | 0.5199 | 2.1897 | 8.0719 ± 2.3208 |
| ridge | 11.3508 | 9.5169 | 0.5190 | 2.1923 | 8.1060 ± 2.2983 |
| lasso | 11.3716 | 9.5497 | 0.5157 | 2.2010 | 8.1325 ± 2.3010 |
| polynomial_deg3 | 11.3376 | 9.8373 | 0.4861 | 2.3224 | 9.8602 ± 3.1599 |
| simple_lr | 15.0951 | 14.0139 | −0.0429 | 4.1137 | 12.8106 ± 1.5590 |
| dummy_mean | 15.0959 | 14.0391 | −0.0467 | 4.0800 | 12.8282 ± 1.5604 |
| *crop×state lookup (no ML)* | – | **7.8075** | **0.6763** | 1.3872 | – |

- **The no-ML lookup beats every model on the test years.** This is the headline real-data finding.
- **SHAP shares:** Crop 83.4 %, State 5.5 %, log(Area) 4.9 %, Rainfall 4.0 %.
- **Train RMSE is higher than test RMSE.** This is not a bug. RMSE is dominated by about 82 physically implausible yields kept in the source data, the training years contain more of them, and the log-target models do not chase them.

### 4.3 Leakage check (the mid-term weakness)

Ridge CV RMSE on the same 5 folds:

| Setup | CV RMSE |
|---|---|
| Scaler and one-hot fitted on all 3,840 training rows **before** CV (leaky) | 0.64077 ± 0.00552 |
| The same steps **inside a Pipeline** (current code) | 0.64077 ± 0.00552 |
| Difference | < 1×10⁻⁶ |

- **Conclusion:** the code already fixes this weakness, and on this data it was numerically negligible.
- **Viva answer:** with 3,840 rows and uniform features, the per-fold mean and std are almost identical to the full-train ones. The Pipeline is still the correct design, because the leak grows with smaller data, target-dependent steps (feature selection) and outliers.

---

## 5. Issues found

Severity: **critical** = can break the live demo · **high** = blocks a showcase goal or invites a hard viva question · **medium** = correctness or robustness worth fixing · **low** = polish.

| # | Sev. | File / location | Problem (observed) | Proposed fix |
|---|---|---|---|---|
| I1 | **Critical** | `ui/theme.py:72-75` and `:174` (`[role="tablist"] { position:absolute; top:-58px; right:… }`); no `@media` rule anywhere in `ui/theme.py` | At **1024 px** the tab pills overlap the app title. At **768 px**, "Predict" and "Model comparison" are pushed off-screen and **cannot be clicked**. Labels break mid-word ("Phosphor/us"), dropdowns truncate ("Bar", "Cer"), scenario pills and the Reset button clip. A 1024×768 projector would show the broken header. Evidence: `screenshots/simple_predict_1024_full.png`, `simple_predict_768.png`, `technical_model_comparison_768.png` | Move navigation to its own row (or `st.navigation` / sidebar) below about 1200 px. Add `@media` breakpoints at 1200 and 900 px: stack the 4 input groups 2×2 then 1 column, and drop the glossary grid from 5 columns to 2 |
| I2 | **High** | `requirements-lock.txt:108` | `pywinpty==3.0.5` is Windows-only, with no marker. A pip dry-run for Linux (Python 3.12 and 3.14) fails with `No matching distribution found for pywinpty==3.0.5`, so **Colab and Streamlit Cloud cannot install the lock**. The lock also pins Python-3.14-era versions and pulls the whole Jupyter stack into the app environment | Split into a pinned cross-platform `requirements.txt` (app runtime only) and `requirements-dev.txt` (jupyter, ruff, pytest); add `runtime.txt` / Python version for Cloud |
| I3 | **High** | `.gitignore:5` (`*.joblib`); `app.py:115-126` | A fresh clone has no models, so the app shows only the missing-models banner (`screenshots/state_missing_models_1440.png`). Training takes 12 min 49 s, so a cloud deploy cannot train on boot | Commit the 8 KB `best_overall.joblib` (and the 770 KB `best_tree.joblib` if needed) via a `.gitignore` exception. Never ship the 79 MB India models; the app does not load them |
| I4 | **High** | `app.py:532-555, 588-607` | Model comparison: the leaderboard ranks **XGBoost #1** (test 0.64613 vs 0.64619), while the KPI says "Curved formula, our best model" and the app predicts with polynomial (chosen by CV, which is correct). An examiner will ask "why not deploy #1?" | Rank the leaderboard by CV RMSE, or label it "selected by CV, test shown for reference". State the 0.00006 t/ha tie in one line |
| I5 | **High** | Brief vs repo | The brief's mid-term pipeline (Lasso selection 25→20, scaling before CV) is **not in this repository**, and the git history has one commit. Mid-term slides and current numbers will not match | Decide which is canonical (Q1). Add a "mid-term → end-sem changes" slide using the §4.3 numbers |
| I6 | Medium | `app.py:384-399` | The prediction is a single number, with **no uncertainty range**. The residual σ of about 0.64 t/ha is about 10 % of the mean | Add a prediction interval (split-conformal or quantile on CV residuals) as a band on the gauge |
| I7 | Medium | `notebooks/0*.ipynb` first cell: `sys.path.insert(0, str(Path.cwd().parent))` | The notebooks only work when the working directory is `notebooks/`. They break in **Colab** (`/content`) or when run from the repo root | Resolve the repo root by searching upward for `src/`, plus a Colab setup cell (clone + pip install) |
| I8 | Medium | `README.md:439-450` | README timings are stale. It claims the pipeline takes 5 min 9 s – 9 min 40 s and the notebooks 1 min 52 s; this run measured **12 min 49 s** and **7 min 01 s**. India took 26 min 01 s vs 13 min 22 s claimed. The "< 1 s" interaction claim (`README.md:305`) did not hold: a preset click took 2.2 s | Re-measure and quote ranges with the hardware. In the live demo, never retrain |
| I9 | Medium | `src/models.py:80-90` (open review item M4) | On India, the Ridge/Lasso α is chosen with inner folds that are not time-ordered | Already documented. Optionally pass a time-ordered `cv` to `RidgeCV`/`LassoCV` |
| I10 | Medium | `tests/test_data.py:8,13,33-34,40` | Tests call `make_split()`, which **overwrites the tracked `data/processed/kaggle_*.parquet`**. Tests should not mutate the repo | Monkeypatch `config.PROCESSED_DIR` to `tmp_path` in those tests |
| I11 | Medium | `app.py:846-866` + `ui/plain.py` India sentences | Simple mode labels India errors "t/ha", but the dataset's units are only mostly t/ha (some crops use other units). The README itself says "yield units" | Use "yield units" or restrict the Simple-mode India chart to t/ha crops |
| I12 | Low | `.streamlit/config.toml:4-5` (`base = "light"`) | No dark mode. A `prefers-color-scheme: dark` request still renders light (`screenshots/dark_request_simple_predict_1440.png`) | Add light and dark theme sections and theme-aware chart colours (Phase 2) |
| I13 | Low | `app.py:784-787` | The "Detailed view" shows a static matplotlib PNG with ~7 px text and raw snake_case names, off-theme | Redraw the beeswarm in Altair with friendly names |
| I14 | Low | `notebooks/03_explain.ipynb` code cell 3 | SHAP beeswarm jitter is unseeded, so the PNG differs between runs (bounding box (366, 90, 984, 994)) | Call `np.random.seed(config.RANDOM_STATE)` before plotting |
| I15 | Low | `app.py:123-126` | The error state is a bare red banner showing a Windows path (`models\best_overall.joblib`), with no title or next steps | A branded empty state with the exact command and the expected runtime |
| I16 | Low | `models/` | Two byte-identical duplicates: `best_linear` = `best_overall` and `india_best_tree` = `india_best_overall` (158 MB of India models on disk) | Save only distinct models |
| I17 | Low | `src/india.py` / India test set | 10 of 4,226 test rows have a crop unseen in training ("Other Summer Pulses"). They are one-hot encoded as all zeros with only a `UserWarning` | Report the count in the India notebook |
| I18 | Low | Repo root | 5 large markdown reports at the root, 22 historical screenshots in `docs/screenshots`, and stale `__pycache__/run.cpython-314.pyc` | Move the reports to `docs/` and keep only the current screenshots |
| I19 | Low | `results/figures/*.png` | Static figures have titles, units and legends (good), but use default matplotlib styling and snake_case labels, so they do not match the app | Shared matplotlib style + friendly-name mapping |
| I20 | Info | `notebooks/04_india.ipynb` code cell 15 | The India Random Forest SHAP cell takes 258.8 s, which is a live-demo risk if anyone re-runs it | Cache the SHAP values to a file; show the notebook pre-executed |

**No issues were found** in these checks:

- target leakage: India `Production` is dropped and tested
- wrong train/test split: random for exchangeable Kaggle rows, time-based for the India panel
- metrics on the wrong set: the test set is used once, after CV-based selection
- unseeded model randomness: every estimator, split and search uses `RANDOM_STATE=42`
- silently dropped rows: every drop is logged and documented
- hard-coded absolute paths: everything resolves from `src/config.py:ROOT`

---

## 6. UI assessment

The app is a single-screen Streamlit dashboard with 5 tabs and Simple/Technical modes. Screenshots are in `screenshots/` (40 PNGs). Naming: `<mode>_<tab>_<width>.png`; `_full` means a full-length capture when the content is taller than the viewport. Viewports used: 1440×900, 1024×768, 768×1024. Dark mode is not supported (I12).

| Aspect | Assessment |
|---|---|
| **Layout at 1440** | Clean and balanced. Every tab fits in 900 px with no horizontal scroll. Roughly the bottom 25–35 % of the screen is empty on most tabs (e.g. `simple_what_drives_yield_1440.png`) |
| **Layout at 1024 / 768** | **Broken header** (I1). At 1024 the content overflows the viewport by 13–69 px on Predict, Model comparison and How it works. At 768, columns get very narrow: slider labels wrap to 3 lines and the Technical KPI cards wrap their values ("0.646 / t/ha") |
| **Typography** | Inter, 14 px minimum on charts, clear hierarchy (26/18/16 px). Good |
| **Colour** | "Cyber Sand": oat `#F7F4EB`, navy `#1E1B4B`, indigo `#6366F1`, purple `#A855F7`. It is consistent, but reads as a tech/SaaS palette, not agriculture. Purple vs indigo carries the positive/negative SHAP meaning, which is a weak hue contrast |
| **Spacing / cards** | Consistent 16 px radius cards with soft shadows. Good |
| **Plot quality (app)** | Altair charts have axis titles with units, value labels and tooltips. The gauge has no visible band labels (hover only). The predicted-vs-actual scatter has a y = x reference line. The detailed SHAP view is a low-resolution PNG (I13) |
| **Plot quality (notebooks)** | Titles, units and legends are present; default styling and snake_case labels (I19) |
| **Loading / error states** | First render ≈ 5.1 s on a warm server. There is no skeleton or spinner text during the cold start (8.5–23.5 s observed on a cold server). The error state is a bare banner (I15). When the India tables are missing, a clear `st.info` with the run command is shown |
| **Interactions** | Presets work (Drought year → 5.99 t/ha, as the README says). The What-if and Detailed-view toggles work. Re-render takes about 2.2 s (Kernel SHAP on every rerun) |
| **Mobile** | Not usable below about 900 px (I1) |
| **Content** | Strong. The "In simple words" cards are generated from the real numbers, and the Simple/Technical split suits a mixed audience |

---

## 7. Showcase gap analysis

Effort: S = under 3 h · M = 3–8 h · L = over 8 h.

| Feature | Exists? | What is there / what is missing | Effort |
|---|---|---|---|
| Interactive prediction form (all inputs, realistic ranges, defaults) | ✅ mostly | 11 sliders bounded by the training min/max, defaults at the medians, 3 dropdowns, 4 presets. **Missing:** a confidence/uncertainty range on the output | S (conformal band) |
| Model comparison dashboard | ✅ mostly | KPIs, leaderboard, RMSE bars, predicted-vs-actual scatter, full table. **Missing:** residual plot in the app, CV-vs-test view, train RMSE; leaderboard ordering (I4) | S |
| Explainability | ✅ partly | Per-prediction Kernel SHAP, global mean \|SHAP\|, grouped importance, beeswarm PNG. Coefficients and permutation importance exist **only in notebook 03** and are not in the app | S–M |
| What-if simulator / partial dependence | ✅ | Single-factor what-if for the current field plus PDP-style response curves over 300 fields. **Missing:** a 2-factor interaction (e.g. fertilizer × rainfall heatmap) and a side-by-side scenario comparison | S–M |
| Stronger models (XGBoost/LightGBM/GB) tuned and compared fairly | ✅ mostly | XGBoost and RF with nested 20-iteration RandomizedSearch. LightGBM is only a fallback import. On Kaggle the noise floor (σ ≈ 0.64) limits gains; on India there is room (the lookup still wins) | M (LightGBM + Optuna, India crop×state target-encoding features) |
| Pipeline + ColumnTransformer, single saved artifact | ✅ | Done. The `best_overall.joblib` pipeline takes the raw 14 columns | – |
| EDA page | ❌ in app (✅ in notebook 01) | 13 EDA figures exist as PNGs. The app has no EDA tab | M |
| Downloadable PDF/CSV report of a prediction | ❌ | Missing | S (CSV) / M (PDF) |
| Deployment (Streamlit Cloud, public link) | ❌ | Blocked by I2 (Windows-only lock) and I3 (models not in git) | S–M once those are fixed |
| README with screenshots, setup, dataset links | ✅ | Thorough README with screenshots, both dataset links and setup. Needs a live-app link, refreshed timings (I8) and fresh screenshots after Phase 2 | S |
| Reproducibility (pinned requirements, seeds, one command) | ✅ on Windows | Seeds everywhere; fresh-clone results byte-identical; `run.py` / `start.bat` one-command run. **Not portable** (I2), and the install fails under long Windows paths | S |
| Demo script / rehearsal material | ❌ | `CODE_REVIEW.md` has viva Q&A, but there is no timed demo script | S |

---

## 8. Recommended UI visual direction: "Field & Harvest"

A clean agricultural theme that reads as farming at a glance, stays calm on a projector, and keeps a colour-blind-safe positive/negative pair for SHAP.

| Role | Light | Dark |
|---|---|---|
| Background (canvas) | `#F6F4EC` parchment | `#121611` |
| Surface (cards) | `#FFFFFF` | `#1B211A` |
| Text primary | `#1F2A1F` | `#E8EDE4` |
| Text secondary | `#5B6656` | `#A7B1A1` |
| Primary (crop green, buttons, active tab) | `#2E7D32` | `#66BB6A` |
| Accent (harvest gold, highlights, best value) | `#E0A526` | `#F2C14E` |
| Soil (soil group, neutral bars) | `#8D6E63` | `#BCAAA4` |
| Sky (weather group) | `#3B82C4` | `#7FB3E6` |
| SHAP positive / negative (colour-blind safe) | `#1B7F5B` teal-green / `#D9822B` orange | `#4CC79A` / `#F0A04B` |
| Border / grid | `#1F2A1F1A` | `#E8EDE41F` |

- **Fonts:**
  - Headings: **Manrope** 700/600
  - Body: **Inter** 400/500
  - KPI numbers: **IBM Plex Mono** 600, with tabular figures so the numbers don't jitter as sliders move
  - All three from Google Fonts. Scale: 28/20/16/14 px, never below 13 px on charts
- **Layout:**
  - Top bar: logo, title, light/dark toggle
  - Navigation as `st.navigation` pages (Predict · Compare models · Explain · What-if · EDA · Real-world check · About), so the tab bar never collides with the header
  - Predict page uses a 12-column grid: inputs in a left rail (4 collapsible groups: Soil 🟤, Weather 🔵, Farming 🟢, Context), a result hero with a prediction band, and the SHAP waterfall on the right
  - Breakpoints: ≥1200 three columns; 900–1199 two columns (inputs on top); <900 single column with the inputs in an expander
- **Feature-group colour coding:** use the same 4 colours everywhere (sliders' group headers, donut, SHAP bars grouped by colour) so the audience learns them once.
- **Light/dark:** use Streamlit's light and dark theme sections. Altair charts read the active theme instead of hard-coding colours (today `ui/charts.py` hard-codes navy). Test both themes at 1440, 1024 and 768.
- **Projector-safe:** check contrast with WCAG AA, avoid thin 1 px lines on key charts, and keep the hero number ≥ 48 px.

---

## 9. Roadmap

Estimates are in focused hours for one person.

### Phase 1: correctness, portability, reproducibility (≈ 6–8 h)

The Pipeline fix itself is already done. This phase hardens the code around it.

- [ ] Split dependencies: cross-platform pinned `requirements.txt` (runtime) + `requirements-dev.txt` (jupyter, ruff, pytest); drop `pywinpty` from the runtime set (I2)
- [ ] Commit `models/best_overall.joblib` (8 KB) and `best_tree.joblib` via a `.gitignore` exception; delete the duplicate models (I3, I16)
- [ ] Repo-root bootstrap in every notebook + a Colab setup cell (I7)
- [ ] Tests write the split to `tmp_path` instead of `data/processed/` (I10)
- [ ] Seed the SHAP beeswarm (I14); report the unseen-category count (I17)
- [ ] Optional: time-ordered inner CV for India Ridge/Lasso (I9)
- [ ] Record the leakage experiment (§4.3) in notebook 02 as a "mid-term vs now" cell
- [ ] Re-run `python run.py build` + `check`; confirm `test_results.csv` is unchanged
- [ ] Refresh the README timings (I8)

### Phase 2: Streamlit app redesign (≈ 12–16 h)

- [ ] Move to `st.navigation` multipage with a top bar; fix the header collision (I1)
- [ ] Apply the "Field & Harvest" palette and fonts; add light + dark themes (I12)
- [ ] Responsive breakpoints at 1200 / 900 px; verify 1440 / 1024 / 768 in both themes with screenshots
- [ ] EDA page: distributions, correlation heatmap, yield by crop/region/season (Altair, interactive)
- [ ] Prediction band (conformal interval from CV residuals) on the hero/gauge (I6)
- [ ] Leaderboard ranked by CV, with test RMSE shown as a reference column (I4)
- [ ] Branded loading and error states (I15); India units wording (I11)
- [ ] Download a prediction report as CSV (inputs, prediction, interval, top SHAP contributions)

### Phase 3: explainability + what-if (≈ 8–10 h)

- [ ] SHAP waterfall per prediction in friendly names; native Altair beeswarm (I13)
- [ ] Coefficient and permutation-importance views in the app (from notebook 03), plus the 4-method agreement table
- [ ] 2-factor what-if heatmap (e.g. fertilizer × rainfall) and a side-by-side scenario A vs B comparison
- [ ] "Soil pH paradox" explainer card: linear coefficient ≈ 0 vs SHAP rank 4, with the dependence curve
- [ ] PDF export of the prediction report (one page)

### Phase 4: model upgrades, compared fairly (≈ 8–12 h)

- [ ] Add LightGBM and sklearn `HistGradientBoosting` to the same nested-CV harness
- [ ] Optuna (or a wider RandomizedSearch) under the same outer folds; report CV mean ± std and test once
- [ ] India: crop×state historical-mean feature (computed inside folds) to try to beat the lookup
- [ ] Statistical tie check (paired fold differences) so "tie" claims are backed by numbers
- [ ] Update tables and figures; keep model selection by CV only

### Phase 5: deploy, README, demo script (≈ 5–7 h)

- [ ] Deploy to Streamlit Community Cloud; smoke-test every page on the public URL
- [ ] README: live link badge, fresh screenshots (light + dark), 3-command setup, both dataset links
- [ ] 5-minute demo script with timings + a backup plan (local `start.bat`, pre-recorded GIF, static screenshots)
- [ ] Rehearse the viva Q&A from `CODE_REVIEW.md` §2, plus the new §4.3 leakage numbers
- [ ] Final full rebuild + check on a clean clone

**Total ≈ 39–53 h.** Phase 1 and the I1 fix are the minimum for a safe live demo.

---

## 10. Open questions

1. **Mid-term vs repo:** the brief describes a Lasso 25→20 selection with leaky scaling, but this repo has neither (one commit, Pipelines throughout). Should the current repo be the canonical end-sem version? Or is there a mid-term notebook to bring in for a before/after story?
2. **Palette:** switch to the agricultural "Field & Harvest" direction (recommended), or keep the existing "Cyber Sand" identity and only add dark mode and responsiveness?
3. **Deployment:** OK to commit the two small Kaggle model files (8 KB + 770 KB) so Streamlit Cloud can start without training? The 79 MB India models would stay out.
4. **Demo hardware:** what resolution is the presentation projector or screen? This decides how urgent I1 is.
5. **Scope of Phase 4:** on Kaggle the noise floor (σ ≈ 0.64) caps gains, so more tuning will barely move RMSE. Should the effort go to India instead (beating the lookup), or should LightGBM/Optuna still be added for the rubric?
