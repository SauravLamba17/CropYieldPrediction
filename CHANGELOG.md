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
