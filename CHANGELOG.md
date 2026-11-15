# Changelog

## Mid-term → End-sem

What changed between the mid-term submission and this repository (the canonical end-sem version).

- **Leakage fix: preprocessing inside Pipelines.** At mid-term the features were scaled before cross-validation, so every fold's scaler had seen its validation rows. Now every scaler, one-hot encoder and polynomial expansion sits inside an sklearn `Pipeline` / `ColumnTransformer` (`src/features.py`) and is refitted on each training fold only.
  - Measured effect on this data (Ridge, 5-fold CV RMSE on the 3,840 training rows): **0.640771 with scaling before CV vs 0.640771 inside the Pipeline, a difference of 5.5 × 10⁻⁸ t/ha** (under 10⁻⁶). The fix matters for correctness, but the leak was negligible here because the features are near-uniform and the 80 % training folds already pin down each column's mean and spread.
- **8 models instead of a handful:** mean baseline, simple linear (fertilizer only), multiple linear, polynomial + Ridge, Ridge, Lasso, Random Forest and XGBoost, all compared on the same 5 folds.
- **Nested cross-validation** for the tree models: the hyperparameter search (`RandomizedSearchCV`, 20 configurations) runs inside every outer fold, just as RidgeCV/LassoCV pick α inside each fold. The model is chosen by CV RMSE and the test set is used once.
- **Explainability:** standardised coefficients, permutation importance and SHAP (TreeExplainer for XGBoost, Kernel SHAP for single predictions of the deployed model), with an agreement table across the four methods (notebook 03).
- **Statistical diagnostics:** VIF, OLS summary, Breusch-Pagan, Jarque-Bera and residual plots (notebook 02).
- **Real-world validation on the India dataset** (Mendeley `ncw2vbcgnk`, 1997–2020, CC BY 4.0): time-based split (train ≤ 2015, test 2016–2020), time-ordered CV, a log1p target, a crop × state look-up baseline (no ML) and an ablation of the year-index columns (`src/india.py`, notebook 04).
- **Streamlit dashboard** (`app.py`): 5 tabs (Predict, Model comparison, What drives yield, Real-world validation, How it works) with Simple and Technical modes, scenario presets, what-if curves and per-prediction SHAP explanations.
- **One-command workflow:** `python run.py` / `start.bat` (set up, train, launch), `run.py build` (full rebuild), `run.py check` (ruff + pytest).

## Phase 1 — portability, correctness, demo-safe header

See `PROGRESS.md` (Phase 1 section) for the measured results behind each item.

- **Requirements split:** `requirements.txt` (cross-platform runtime set, the exact lock versions, no Windows-only packages), `requirements-dev.txt` (adds Jupyter, ruff, pytest, the Kaggle client) and `requirements-lock-windows.txt` (the previous full lock, renamed). `run.py` installs the Windows lock on Windows and `requirements-dev.txt` elsewhere. The pinned set needs Python 3.12+.
- **Kaggle models committed:** `models/best_overall.joblib` (loaded by the app) and `models/best_tree.joblib` (loaded by notebook 03), so a fresh clone opens the dashboard without training. The India model stays git-ignored.
- **Duplicates removed:** `best_linear.joblib` and `india_best_tree.joblib` (byte-identical copies nothing loaded) are no longer written; the Kaggle `.zip` and a stray root `__pycache__` were deleted locally. Historical `before_` / `after_` / `v1_` screenshots moved to `docs/screenshots/archive/`.
- **One "best model" story:** the leaderboard, tables and figures rank models by CV RMSE (the selection criterion); test RMSE is shown alongside, and a train RMSE column was added. Note in the app: "Selected by cross-validation; test set used once for final evaluation."
- **Notebooks run anywhere:** a "Setup (Colab)" cell finds the project root from the repo root, `notebooks/` or Colab, and on Colab clones the repo and installs `requirements.txt`.
- **Deterministic SHAP beeswarm:** `shap.summary_plot` gets a seeded `rng`, so two runs give a byte-identical PNG (seeding NumPy's global RNG also worked, but shap then printed a FutureWarning containing a temp path).
- **Tests no longer write tracked files:** the split tests write to pytest's `tmp_path`, and skip (like the India tests) when the git-ignored Kaggle CSV is absent. New tests cover unseen categories, leaderboard ordering, notebook root resolution and the India unit wording (31 → 47 tests).
- **India:** errors are labelled "yield units (varies by crop)" with a tooltip instead of t/ha; unseen test-year crops are handled (all-zero one-hot) and tested; the Ridge/Lasso penalty is now tuned on time-ordered year blocks.
- **Responsive header:** the tab pills move to their own row below 1360 px and wrap below 900 px, so the title and tabs never overlap and all five tabs stay clickable; input groups go 2 × 2 below 1200 px.
- **Friendly setup states:** missing models or a missing India build show an in-app card with the exact command instead of a red error.

## Phase 2 — UI redesign ("Field & Harvest")

See `PROGRESS.md` (Phase 2 section) for the measured results.

- **Design tokens** (`ui/theme.py`): one source for the CSS, the chart palette and matplotlib (PDF report), light and dark. Every text/background pair passes WCAG AA (checked in tests). Two shades were adjusted to pass: the SHAP "negative" orange is `#C2711C` in light mode (the agreed `#D9822B` was 2.9:1 on white), and gold, soil and sky get darker text variants for text. In dark mode primary buttons use dark text (white on `#66BB6A` is only 2.4:1).
- **Fonts bundled offline**: Inter, Manrope and IBM Plex Mono (woff2, OFL-1.1, with licences) in `static/fonts/`, loaded through `[[theme.fontFaces]]`. No CDN; verified with all non-local hosts blocked.
- **Light / dark mode** via `[theme.light]` and `[theme.dark]` and the ⋮ menu. CSS uses `light-dark()` and charts colour by theme series, so a live switch restyles everything without a rerun. First visit defaults to light.
- **Page navigation** (`st.navigation`, `app_pages/`): Overview (new landing page with headline numbers read from the tables and a pipeline strip), Predict, What-if, Model comparison, Data explorer, India. The Simple / Technical switch is global.
- **Layout system**: spacing scale, card components, KPI grid that reflows 4 → 2 → 1 with numbers that never wrap; one display-name mapping (`ui/labels.py`) for every column, model and group.
- **Charts**: one themed Altair style; CV RMSE bars with ± 1 sd error bars; predicted-vs-actual and residual plots for the deployed or best tree model; an interactive SHAP waterfall per prediction and a SHAP beeswarm, replacing the static PNG.
- **90 % prediction interval** (split-conformal on out-of-fold CV residuals, computed once in `src/pipeline.py`, saved to `results/tables/conformal.json`): ± 1.037 t/ha, 88.5 % coverage on the 960 test fields. No existing metric changed.
- **Data explorer** with the key EDA from notebook 01 (works from the committed training split; the harvest-month chart shows a setup card without the raw CSV).
- **Prediction report** download as CSV and a one-page PDF (matplotlib, already installed).
- **Performance**: the Streamlit file watcher is off (`fileWatcherType = "none"`); it cost about 5 s per new browser session. Cold start 20.2 s → 8.7 s; preset click 1.7–1.9 s → 0.4–0.5 s (same harness, same day).

## Phase 3 — explainability and what-if depth

See `PROGRESS.md` (Phase 3 section) for the measured results.

- **Screenshots:** a curated set in `docs/screenshots/` (`p2_*`, `p3_*`); the per-phase bulk captures in `screenshots/` are git-ignored and were removed from the unpushed history (approved; verified commit by commit; repo pack 13.2 MiB).
- **Pipeline explainability tables** (`src/explain_tables.py`, run inside `src/pipeline.py` or alone with `python -m src.explain_tables`): permutation importance on the 960 test fields (30 repeats, fixed seed) for the deployed model and XGBoost, per original feature; mean |SHAP| per original feature for both; standardised coefficients of multiple linear, Ridge and Lasso with 95 % intervals from 500 bootstrap refits; per-crop and per-region coverage of the 90 % interval. No model, split, CV scheme or existing metric changed.
- **Explain page** ("Why the model decides"): importance with error bars, a "do the methods agree?" rank chart with Spearman ρ and a generated summary, coefficients with bootstrap intervals, PDP with 50 ICE lines, method notes.
- **Soil pH paradox card** with live Pearson / Spearman, the model's response curve and its optimum region read from the curve; linked from the Data explorer.
- **What-if:** a 25 × 25 two-factor heatmap restricted to the observed data range, and Scenario A vs B with a SHAP gap chart, Swap / Reset and a CSV export (`ui/report.comparison_frame`).
- **Loading states:** spinners with short messages around the first SHAP explanation, the heatmap, PDP/ICE, A vs B and the SHAP beeswarm.
- **Conformal transparency:** per-group coverage table on the Predict page (Technical mode).
- The Explain page's navigation label is "Explain" (heading "Why the model decides"): the long label made the top navigation measurably slower to lay out.

## Phase 4 — India model upgrade

See `PROGRESS.md` (Phase 4 section) for the measured results.

- **Look-up audit:** the crop × state look-up uses only training years (and, in CV, each fold's training years), pools seasons, works on log1p and falls back to the overall training mean for the 147 test rows with an unseen pair (10 of them an unseen crop). No leakage, so no fix was needed; notebook 04 section 12a recomputes it.
- **Time-safe India features** (`src/india_features.py`): target encoding of crop × state, crop and state from strictly earlier years only (smoothed, smoothing tuned by CV); lags (last year's yield, 3-year rolling mean, years on record); the year as a trend. Forecasting assumption: predicting year *t*, actual yields up to *t* − 1 are known. A strict variant (only 1997–2015 known) is reported for every model.
- **Models** (`src/india_models.py`, `src/models.py`): LightGBM and HistGradientBoosting added; random forest and XGBoost re-run with the new features; a hybrid (time-safe look-up + residual model). Every model gets the same seeded `RandomizedSearchCV` budget (20 candidates, inner 3 year-block folds) inside the same 5 outer year blocks. Failed fits now raise instead of scoring NaN (`error_score="raise"`).
- **Ablation, selection, error analysis and explainability** (`src/india_analysis.py`, run by `python -m src.india`): base / + target encoding / + lags / + trend / all / hybrid plus the look-up and last-year references; selection by CV RMSE only, then one test; per-crop and per-state errors against the look-up, rows with no history and the unseen crop; permutation importance (30 repeats) and mean |SHAP| for the selected model. New tables: `india_leaderboard.csv`, `india_feature_ablation.csv`, `india_error_by_group.csv`, `india_error_segments.csv`, `india_explain_permutation.csv`, `india_explain_shap.csv`, `india_selection.json`.
- **Result:** random forest with all features selected (CV 5.12 vs 5.97 for the look-up; test 6.85 vs 7.81). Repeating last year's yield already scores 6.95, and the strict variant scores 9.65 (worse than the look-up).
- **Kaggle comparison rows** (`kaggle_extra_models.csv`): HistGradientBoosting CV 0.639, LightGBM 0.647, against the deployed polynomial's 0.635. The deployed model, its features, splits, CV, conformal interval, metrics and model files are unchanged.
- **India page:** leaderboard ranked by CV with test as its own column, ablation chart with the look-up reference line, a generated "how the model beats the look-up" card, the 10 worst crops vs the look-up, permutation and SHAP for the selected model, and the new tables. The Simple-mode lessons follow the selected model. Without the Phase 4 tables the page keeps the old view and shows a setup card.
- `run.py` re-runs the India step when `india_selection.json` is missing.
- Notebook 04: section 12 (audit, leaderboard, ablation, errors) and an explainability section that reads the saved tables.

## Phase 5 — deploy-ready, CI and demo

See `PROGRESS.md` (Phase 5 section) for the measured results behind each item.

- **Requirements split (approved):** `requirements.txt` is now the app runtime only, which is what Streamlit Community Cloud installs. It pins 12 direct packages at the same versions, with `xgboost-cpu` on Linux (no NVIDIA libraries). `requirements-pipeline.txt` adds statsmodels and seaborn for the pipelines and notebooks, and the Colab cells now install it. `requirements-dev.txt` builds on it. Linux / Python 3.12 install: **782 MB instead of 1,097 MB**.
- **`python run.py build --fast`:** runs the Kaggle pipeline and the notebooks, and reuses the committed India tables (33 min 35 s vs 129 min 44 s for the full build). The full build is unchanged.
- **Paired statistical tie check (rerun approved):** `python -m src.tie_check` re-runs only the compared models' CV, with the same code, seeds and folds. It saves per-fold scores (`cv_folds.csv`) after checking they reproduce the committed means and stds (max difference 4.4 × 10⁻¹⁶), then writes `tie_check.csv`. Results:
  - Kaggle: polynomial vs XGBoost p = 0.71, vs HistGB p = 0.25, both ties.
  - India random forest (all features): vs last year's yield p = 0.68 (a tie), vs the look-up p = 0.040, vs HistGB (all features) p = 0.36.
  - In the app: a "Statistically distinguishable?" card on Model comparison and India (Technical). The India verdict now says the gain over last year's yield is within fold-to-fold noise.
- **Cloud readiness:**
  - A health-check card under the header lists any missing results table and the command that rebuilds it. Pages never raise: each of the 30 tables was removed in turn, and so were all of them at once.
  - The input-keyed what-if cache is bounded (64 entries).
  - Measured with `requirements.txt` only: peak memory 468 MB, health check 2.2 s, first page 14.6 s, 0 external requests.
- **CI:** `.github/workflows/ci.yml` runs on Ubuntu with Python 3.12 and needs no secrets. Two jobs:
  - lint + the full test suite on the dev set;
  - a smoke job on `requirements.txt` only that renders every page in both modes.
- **Docs:**
  - `docs/DEPLOY.md`: the Cloud deploy guide;
  - `docs/DEMO_SCRIPT.md`: a timed 8-minute demo for three presenters, with a checklist and plan B;
  - `docs/VIVA_NOTES.md`: rewritten as a final Q&A sheet by topic;
  - README: a top summary with headline results, a screenshot strip, a live-demo placeholder, the CI badge, runtime vs full install, fast vs full build, and team and course details.
- **Small fixes:**
  - India page: a setup card is also shown when `india_model_names.json` is missing.
  - Build-time hints are updated (26 / 105 min).
