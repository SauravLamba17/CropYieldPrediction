# PROGRESS — Phase 0 audit and showcase plan, Phase 1–4 results

CSL236 AIML course project, Team 2: crop yield prediction (t/ha).
Audit date: 2026-11-14. Commit audited: `511cdfe`.

Every number in this file was produced in this session by a fresh `git clone` of the repo, with a fresh virtual environment built from `requirements-lock.txt` (Python 3.14.4, Windows 11, 16 logical CPUs). Nothing was copied from the old outputs. The original working folder was only read: no code, notebook, data or result file in it was changed.


> **Screenshots:** since Phase 3 the per-phase captures in `screenshots/` (Phase 0 audit, `phase1/`, `phase2/`, `phase3/`) are git-ignored and local only; they were also removed from the unpushed history (with approval). A curated set is committed in `docs/screenshots/` (`p2_*`, `p3_*`). Paths below that start with `screenshots/` refer to the local folder.

---

## Phase 4 results (2026-11-15): India model upgrade

Measured in the Phase 4 session: the real repo, then a fresh `git clone` with a fresh venv from `requirements-dev.txt` in a short `subst` path, and my own headless Chrome. The real repo's `.venv` was not touched. Tuning: seeded `RandomizedSearchCV`, 20 candidates, inner 3 year-block folds, inside the same 5 outer year blocks for every model. *Forecasting assumption:* predicting year *t*, actual yields up to *t* − 1 are known; a strict variant (only 1997–2015 known) is reported next to every test score.

| Check | Result |
|---|---|
| `pip install -r requirements-dev.txt` (fresh clone) | OK, 14 min 13 s (two PyPI read timeouts; succeeded with `--timeout 300 --retries 10`) |
| `python run.py check`, fresh clone, no raw data | ruff clean; 143 passed, 17 skipped (61 s); `git status` clean |
| Full `python run.py build`, fresh clone | exit 0, **129 min 44 s**: Kaggle 25 min 49 s, India 102 min 59 s, notebooks 55 s. **Over the 60-min budget** (see issues) |
| Kaggle unchanged | every existing metric max \|diff\| **0.0** (only `fit_time_s` differs); `best_overall.joblib` / `best_tree.joblib` byte-identical (sha256) |
| India reproducible (two runs: real repo, fresh clone) | every India table identical apart from `fit_time_s` and the timing block in `india_selection.json` |
| Original India suite | `india_cv.csv` max \|diff\| 0.0 against Phase 3 |
| Notebooks | all 4 executed, **0 errors** |
| Offline test (only localhost resolvable) | India page, 2 modes × 2 themes at 1440 and 768: 1,857 requests, 0 external |
| Screenshots + geometry | India page, 2 modes × 2 themes × 6 widths: 24 shots, 5,409 requests, 0 external, **0 problems** |

**Lookup audit:** built from training years only, seasons pooled. 147 test crop × state pairs never appear in training (10 rows of them are the one unseen crop); all fall back to the overall training mean of the log target (1.724). No leak, no fix needed.

**Leaderboard (selected by CV only, test scored once):**

| Model | CV RMSE | Test | Strict test | Test R² |
|---|---|---|---|---|
| **Random forest, all features (selected)** | **5.119 ± 2.882** | **6.848** | 9.648 | **0.751** |
| Random forest + lags | 5.144 ± 2.921 | 6.939 | 9.130 | 0.744 |
| XGBoost, all features | 5.225 ± 2.613 | 7.045 | 8.779 | 0.737 |
| HistGradientBoosting, all features | 5.254 ± 2.780 | 6.941 | 8.397 | 0.744 |
| LightGBM, all features | 5.292 ± 2.919 | 7.052 | 9.124 | 0.736 |
| Last year's yield (no ML) | 5.405 ± 3.619 | 6.950 | 7.721 | 0.744 |
| Random forest (original suite) | 5.886 ± 2.962 | 8.120 | 8.120 | 0.650 |
| Crop × state lookup (no ML) | 5.967 ± 2.887 | 7.808 | 7.808 | 0.676 |
| Hybrid: lookup + random forest residual | 6.062 ± 2.773 | 7.526 | 8.622 | 0.699 |

**Ablation (random forest, CV RMSE):** base 5.886 → + target encoding 6.053, **+ lags 5.144**, + trend 5.806, all 5.119; hybrid 6.062; lookup 5.967. Lags are the group that matters (−0.74).

**Verdict:** ML beats the lookup under the one-year-ahead assumption (CV 5.12 vs 5.97, test 6.85 vs 7.81, better for 48 of 54 crops and 26 of 30 states). But repeating last year's yield already scores 6.95. In the strict variant the model (9.65) is worse than the lookup. On the 51 test rows with no earlier crop × state record it is slightly worse too (RMSE 1.070 vs 0.971 t/ha); for unseen pairs it is better (0.696 vs 0.951; 147 rows).

**Explainability (selected model):** permutation (30 repeats, test years): 3-year rolling mean 7.07, last year's yield 2.79, crop 0.378, state 0.038, log area 0.021. Mean |SHAP| (1,000 test rows): rolling mean 70.6 %, crop 9.9 %, last year 8.7 %, state 7.0 %, season 1.1 %. Spearman ρ 0.78. Year and pesticide have zero permutation importance (every test year is outside the training range).

**Kaggle comparison rows (not deployed):** HistGradientBoosting CV 0.6392 ± 0.0090 (test 0.6593), LightGBM 0.6474 ± 0.0084 (test 0.6654), deployed polynomial 0.6349. Neither beats it by 1 CV sd; nothing switched.

**Model file:** `models/india_best_overall.joblib` is 10.06 MB, git-ignored and not loaded by the app. Not committed (over 5 MB: needs approval).

**Issues found and fixed:**
- AppTest without the Phase 4 tables raised `KeyError 'random_forest + all'` → the page now falls back to the best test row; selection read from `india_selection.json`.
- Simple-mode lesson claimed "crop matters most", which is false for the lag-based model → lessons now come from the SHAP shares (history vs crop).
- The verdict did not compare the strict score with the lookup, and Simple mode lacked the last-year caveat → both added.
- Ablation chart: the lookup label overlapped the hybrid's value label → label pinned top-right with more headroom.
- Simple label "Chosen model (Forest of decision tre…" truncated → "Chosen model"; permutation axis title clipped at 768 px → "Rise in test RMSE".
- Screenshot geometry check reported "header covers content" on every shot → false positive (container top padding was ignored); check fixed, then 0 problems.
- **Not fixed: compute budget.** The first India run on battery took 3.4 h; on mains power the full build is 2 h 10 min, against a ~60 min target. The 20-candidate budget was kept so the results stay as reported; reducing it means rerunning and re-reporting.

---

## Phase 3 results (2026-11-15): explainability and what-if depth

Measured in the Phase 3 session (scratch working copies, a fresh `git clone` with a fresh venv from `requirements-dev.txt`, my own headless Chrome). The real repo's `.venv` was not touched.

**Repo hygiene (item 0, approved):** the 20 unpushed commits were rewritten to drop `screenshots/` (277 files). A git bundle of the old HEAD was kept in the scratch folder, and the rewrite was verified before cleanup:
- 20 commits, identical messages, authors and dates;
- each commit differs from its original only by `screenshots/` deletions;
- `run.py`-equivalent checks passed (132 tests).

Object store: **53.83 MiB loose (≈ 51 MB packed, the bundle size) → 13.22 MiB packed** after `git gc --prune=now`. `/screenshots/` is now git-ignored; a curated set lives in `docs/screenshots/`.

| Check | Result |
|---|---|
| `pip install -r requirements-dev.txt` (fresh clone) | OK, 9 min 51 s |
| `python run.py check`, fresh clone, no raw data | ruff clean; 128 passed, 17 skipped; `git status` clean |
| pytest with both CSVs (working tree) | **145 passed** |
| Kaggle pipeline rebuild (fresh clone, 27 min 52 s incl. the new explainability step) | max \|diff\| **0.0** on every existing metric; `best_overall.joblib` / `best_tree.joblib` byte-identical; `conformal.json` identical |
| Explainability tables reproducible | identical across 3 runs (dev run, fresh-clone pipeline, standalone rerun), byte for byte |
| Offline test (only localhost resolvable, OS dark, empty profile) | 243 requests over 7 pages × 2 modes, 0 external, fonts loaded, opens in light |
| Geometry (stricter check: also buttons outside their column/card) | Explain + What-if: 6 widths × 2 themes × 2 modes, **0 problems**; other pages rechecked, 2 older clipped toggles found and fixed (Predict at 1024 px, Data explorer at 768 px), then 0 |

**Importance, deployed polynomial model (test set):**
- **Top 5 by permutation (Δ RMSE):** fertilizer 0.733, pesticide 0.122, rainfall 0.069, soil moisture 0.009, average temperature 0.008.
- **Top 5 by mean |SHAP|:** fertilizer 0.742, pesticide 0.229, rainfall 0.196, soil pH 0.060, soil moisture 0.057.
- **Top 5 by |coefficient| (multiple linear):** fertilizer 0.879, pesticide 0.265, rainfall 0.214, soil moisture 0.068, crop type 0.066.
- **Spearman ρ between methods:** permutation–SHAP 0.895, permutation–coefficient 0.732, SHAP–coefficient 0.736 (XGBoost: 0.908 / 0.754 / 0.701).
- **App sentence:** "All three methods rank **fertilizer** first … average Spearman ρ of **0.79** … They disagree most on **soil pH** (ranks 4–13)."

**Soil pH paradox:** Pearson r = +0.019, Spearman ρ = +0.018 (3,840 training fields). The deployed model's average prediction runs 6.14 → 6.41 → 6.17 t/ha over pH 4.5 → 6.55 → 8.5: a genuine interior peak at pH 6.55. The region within 10 % of the peak is pH 5.9–7.2. The effect is modest: 0.27 t/ha across the whole range. Quadratic-term p = 4.4 × 10⁻⁴. Lasso keeps pH at +0.0100 t/ha per sd, but its 95 % bootstrap interval (−0.0032 to +0.0321) crosses zero.

**Per-group coverage of the 90 % interval (test set):**
- **By crop:** Barley 91.1 %, Corn 86.4 %, Rice 85.9 %, Soybean 89.2 %, Wheat 89.5 %.
- **By region:** Central 87.6 %, East 91.3 %, North 92.0 %, South 86.0 %, West 85.9 %.
- No group is under 85 %.

**Timings (headless Chrome 1366×768; the machine was noticeably busier late in the session, so two runs are given):**

| | Quieter run | Busier run |
|---|---|---|
| Explain page, cold start (first page of a cold server) | 3.4–4.7 s | 13.8 s |
| Explain page, warm load | 0.65–0.85 s | 1.49–1.59 s |
| PDP/ICE switch (first compute / cached) | 0.55 s | 1.40–1.64 s |
| Heatmap switch | 0.52–0.54 s | 1.0–1.28 s |
| A vs B (reopen What-if after changing A / Swap) | – | 1.46–1.51 s / 1.35–1.47 s |

**Regression check vs Phase 2:** the same Phase 2 commit measured back-to-back with the same harness:
- **Overview warm median:** Phase 2 0.785 / 0.762 s, Phase 3 0.794 / 0.742 s.
- **Preset click:** Phase 2 0.411 / 0.543 s, Phase 3 0.426 / 0.411 s.
- No regression. The absolute Phase 2 figures (0.56–0.63 s, 0.42–0.44 s) were not reproduced by *either* version on the day; a long nav label ("Why the model decides") had cost about 13 % on Overview loads, so the nav item is "Explain".

**Issues found during Phase 3 testing, all fixed:**

1. **The bare `screenshots/` ignore rule** also matched `docs/screenshots/`; it is now anchored as `/screenshots/`.
2. **The history rewrite's final checkout** deleted the bulk screenshots from disk; they were restored from the scratch copy made beforehand.
3. **shap logged every Kernel SHAP row at INFO** (thousands of lines in the pipeline log); its logger is set to WARNING during that step.
4. **The rank chart's connecting lines** didn't line up with the offset dots; the lines were removed.
5. **The coefficient chart's rows** were too tight; the chart is taller.
6. **The pH curve's 1-decimal ticks** repeated "6.3"; flat curves now get 2 decimals.
7. **The headline lower-cased "soil pH"** to "soil ph"; a `labels.lower()` helper now keeps "pH".
8. **The Explain page's model toggle** was clipped at 768 px; it has its own row. The geometry check now also flags buttons outside their column or card, which found two older clipped toggles (Predict at 1024 px, Data explorer at 768 px); both fixed.
9. **An expanded B next to a collapsed A** left a large gap; both start collapsed.
10. **A long nav label** made Overview loads slower (measured); the label is "Explain".
11. **A comparison test** looked up a row name used twice (an input and its SHAP gap); it now selects by section.
12. **The setup card's "about 6 minutes"** became the measured 5–17 minutes.
13. **My timing scripts** first reported 0.1 s for interactions that hadn't happened (react-aria comboboxes ignore synthetic clicks). They now type into the combobox, and every timing waits for proof of the change.

**Deferred:**
- India explainability tables.
- A per-group (Mondrian) conformal width.
- A slider-driven A vs B timing: the B slider ignores keyboard events in automation, so A vs B was timed via preset + Swap.
- The A vs B inputs stay inside expanders to keep the page short.

---

## Phase 2 results (2026-11-14): UI redesign

Measured in the Phase 2 session: a scratch copy of the working tree, a fresh `git clone` with a fresh venv from `requirements-dev.txt`, and my own headless Chrome (Python 3.14.4, Streamlit 1.64.0, Windows 11). The real repo's `.venv` was not touched.

| Check | Result |
|---|---|
| `pip install -r requirements-dev.txt` (fresh clone) | OK, 9 min 48 s (a screenshot run was using the CPU at the same time; Phase 1 measured 5 min 59 s) |
| `python run.py check`, fresh clone, no raw data | ruff clean; 115 passed, 17 skipped (need the git-ignored CSVs); `git status` clean |
| pytest with the Kaggle CSV present (fresh clone) / with both CSVs (working tree) | 121 passed, 11 skipped / **132 passed** |
| App from the fresh clone without a build | every page renders in both modes (Streamlit AppTest) and in Chrome |
| Kaggle pipeline rebuild (fresh clone, `python -m src.pipeline`, 7 min 29 s) | max \|diff\| **0.0** on every test, CV, GroupKFold and linear-CV metric; same row order; `conformal.json` identical. Only `fit_time_s` differs |
| Offline test (Chrome resolving only localhost; OS in dark mode; empty profile) | 233 requests over all 6 pages, **0 external**, 0 failed; Inter, Manrope and IBM Plex Mono loaded from the app; app opened in light mode |
| Geometry check: 6 pages × 2 modes × 2 themes × 1920/1440/1366/1280/1024/768 | **0 problems**: no sibling overlaps, no horizontal scroll, no clipped or wrapped KPI numbers, no clipped controls/tables, title below the header, every page reachable from the navigation (visible link or the "… more" menu at 768 px) |

**90 % prediction interval (split-conformal, out-of-fold 5-fold CV residuals of the deployed polynomial model):** half-width **± 1.0367 t/ha** (n = 3,840), empirical coverage on the 960 test fields **88.54 %** (850 / 960). Example: Typical field **6.52 t/ha (90 % range 5.48–7.56)**; Drought year 5.99 t/ha (4.95–7.03).

**Performance, same harness (headless Chrome 1366×768, cold server, 3 runs):**

| | Phase 1 app | Phase 2 app |
|---|---|---|
| Cold start (first page rendered) | 20.2 s (Predict) | 8.7 s (Overview); 9.5 s for the first Predict page |
| Warm page load | 1.8–2.8 s | 0.56–0.63 s (Overview), 1.5–1.7 s (Predict) |
| Preset click: number changes / fully settled | 1.73–1.85 s / 2.2–2.5 s | 0.42–0.44 s / 0.90–0.98 s (first click 1.29 / 1.97 s) |

The Phase 1 report measured 16.2 s / 1.5 s / 1.8 s on a quieter machine; both columns above were measured on the same day for a fair comparison.

**Contrast (WCAG):** every pair in `ui/theme.CONTRAST_PAIRS` passes in both modes (text ≥ 4.5:1, chart marks ≥ 3:1); `tests/test_theme.py` enforces it. Main pairs, light / dark: text on background 13.53 / 15.38; muted text on background 5.48 / 8.49; green on background 4.66 / 7.73; primary-button text on green 5.13 / 7.73; white on the hero green 5.13–7.87 / 5.13–13.36.

**Issues found during Phase 2 testing, all fixed:**

1. With both light and dark themes defined, Streamlit follows the OS theme, so the app opened dark on dark-mode laptops. A one-time script now sets Light on the first visit; the ⋮ menu still switches (and is remembered).
2. Streamlit does not rerun the script when the viewer switches theme, so Python-side colours went stale. Fixed by making all colours browser-side: CSS `light-dark()` variables and charts coloured by theme series from `config.toml`.
3. Value labels and lines with a fixed colour stayed black in dark mode; `alt.ColorDatum` cannot carry the theme scale, so fixed colours now use a constant series field added in `charts.style()`.
4. Charts drew the page background as a box inside cards (a white box on the green gauge); backgrounds are now transparent.
5. The new header (navigation) covered the app title; the top padding was increased (checked by the geometry test).
6. The y-axis dropped every other factor name on short charts (tick de-collision); de-collision now applies to the numeric axis only.
7. The model picker on Model comparison was clipped; it has its own row, and the geometry check now flags clipped controls and tables.
8. The SHAP waterfall started at 0, so the pushes were slivers; the axis now zooms to the running total.
9. Warm page loads took about 5–6 s: Streamlit's file watcher re-scanned every imported module on each new session. `fileWatcherType = "none"` (0.6 s Overview loads).
10. Widgets with a `default` whose value is also kept in Session State logged a warning; defaults now live in `ui/data.init_state()`.
11. My screenshot driver initially matched nav links by full text (which includes the icon's ligature name); fixed in the driver, not the app.

**Deferred (not Phase 2 scope):** coefficient and permutation-importance views in the app, a 2-factor what-if heatmap and scenario A vs B (Phase 3); a custom loading state (Streamlit's spinner remains); deployment (Phase 5). India Pesticide-per-hectare has no unit in the source data, so its label carries none.

Screenshots: `screenshots/phase2/<mode>_<theme>_<page>_<width>.png` (144), plus `_full` captures at 1440 px (24) and `layout_geometry.json`.

---

## Phase 1 results (2026-11-14): portability, correctness, demo-safe header

Every number below comes from runs in the Phase 1 session: a scratch copy of the working tree, then a fresh `git clone` of the committed result with a fresh venv built from `requirements-dev.txt` (Python 3.14.4, Windows 11, 16 logical CPUs, mapped to a short `subst` drive to stay under the 260-character path limit). The real repo's `.venv` was not touched.

| Check | Result |
|---|---|
| `pip install -r requirements-dev.txt` (fresh venv) | OK, 5 min 59 s |
| Linux resolution of `requirements.txt` / `requirements-dev.txt` (`uv pip compile --python-platform x86_64-manylinux_2_28`, Python 3.12 and 3.14) | 0 errors; 62 / 152 packages, no Windows-only package. Python 3.11 fails (`contourpy==1.4.0` needs 3.12+). Plain `pip --dry-run --platform` cannot evaluate `win32` markers for another OS, so it reports `pywinpty` unless those lines are removed (then 0 errors) |
| `python run.py check`, fresh clone, no raw data | ruff clean; 30 passed, 17 skipped (need the git-ignored CSVs); `git status` clean |
| `python run.py check`, fresh clone, raw data copied in | ruff clean; **47 passed**; `git status` clean |
| App from the fresh clone **without a build** | Starts; Kaggle tabs and India tab work (models and tables are committed). First load after a cold start 16.2 s, later loads 1.4–1.5 s, a preset click settles in 1.6–1.8 s (6.52 → 5.99 t/ha) |
| `python run.py build` (fresh clone) | exit 0, **39 min 15 s**: pipeline 12 min 37 s, India 22 min 40 s, notebooks 3 min 57 s |
| Metrics after that build vs the committed tables | max \|diff\| 0.0 for every Kaggle and India test/CV/ablation metric, same row order. Only `fit_time_s` and notebook execution timestamps differ |
| Notebooks, kernel cwd = repo root | 0 errors; 62 / 21 / 28 / 159 s (270 s) |
| Notebooks, kernel cwd = `notebooks/` | 0 errors; 53 / 18 / 21 / 143 s (234 s) |
| SHAP beeswarm, two consecutive executions of notebook 03 | byte-identical PNG (sha256 `ae9d8393…`) |
| Leakage demo (Ridge, 5-fold CV RMSE): scaler fitted before CV vs inside the Pipeline | 0.640771 vs 0.640771, difference 5.5 × 10⁻⁸ |
| Header geometry, 5 tabs × 2 modes × 1920/1440/1366/1024/768 (+1280) | 0 problems: no tab overlaps the title or toggle, all tabs inside the viewport and clickable at their centre, no horizontal page overflow |

**India, old vs new (time-ordered penalty tuning, item 11).** Only the four penalised linear models change:

| Model | CV RMSE old → new | Test RMSE old → new | Test R² old → new | median CV-fold α old → new |
|---|---|---|---|---|
| ridge | 8.106 → 9.123 | 9.5169 → 9.5169 | 0.519 → 0.519 | 0.373 → 0.869 |
| lasso | 8.133 → 8.455 | 9.5497 → 9.5173 | 0.516 → 0.519 | 0.00018 → 0.00027 |
| polynomial | 7.916 → 8.977 | 9.4807 → 9.7175 | 0.523 → 0.499 | 0.373 → 4.71 |
| polynomial_deg3 | 9.860 → 10.807 | 9.8373 → 10.0209 | 0.486 → 0.467 | 0.494 → 59.6 |
| random_forest (selected), XGBoost, OLS, simple_lr, dummy | unchanged | unchanged (RF 8.1195) | unchanged | – |

CV RMSE rises because later years no longer inform α; the degree-2 polynomial no longer beats plain OLS on the test years. No headline conclusion changes. The α of each final fit is in `results/tables/india_best_params.json`.

**Unseen categories:** 1 India test crop never appears in training ("Other Summer Pulses", 10 of 4,226 test rows); no unseen state or season. All are encoded as all-zero one-hot columns; covered by tests.

**Issues found during Phase 1 testing, all fixed:**

1. India penalty search scored in original units: `expm1` overflowed for weakly penalised candidates on later years, giving NaN scores that silently dropped those α values. Fixed by tuning inside the log1p wrapper (as RidgeCV/LassoCV did), so only the folds changed; 0 NaN scores afterwards.
2. A cold-started Lasso grid took 730 s single-threaded; the search now uses all cores (India step 22 min 40 s, vs 26 min 01 s in Phase 0).
3. On a fresh clone `run.py check` failed 6 split tests (the Kaggle CSV is git-ignored); they now skip like the India tests.
4. Seeding NumPy's global RNG made shap print a FutureWarning containing a temp path, so notebook 03's output changed every run; `summary_plot` now gets a seeded `rng`.
5. A 1200 px breakpoint was not enough: the inline pill row needs about 1344 px, so it still overlapped the title at 1280 px. The pills take their own row below 1360 px.
6. `ruff` flagged `exec` in the first notebook test (S102); the setup cell now runs in a fresh interpreter via `subprocess`.
7. `start.bat`: an unescaped `)` inside an `if (...)` block would have ended the block early; escaped.
8. The Technical results table crashed (`KeyError: train_rmse`) on tables from older builds; it now shows only the columns present.

**Deferred (not Phase 1 scope):** visual redesign, dark mode, `st.navigation` pages (Phase 2); EDA page, prediction band, CSV report (Phase 2); Technical KPI value wrapping and crowded axis ticks at 768–1024 px (Phase 2 layout); Streamlit Cloud deployment and a smaller Linux XGBoost install (`nvidia-nccl-cu13` is pulled in on Linux; Phase 5); the "mid-term vs now" leakage cell in notebook 02 (the measured result is in `CHANGELOG.md`); a joblib `delayed` UserWarning printed by the parallel India penalty search (harmless).

Screenshots: `screenshots/phase1/<mode>_<tab>_<width>.png` (`_full` = whole page), plus `state_india_not_built_1366.png`, `state_missing_models_1366.png` and `header_geometry.json` (the measured header boxes).

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

- [x] Split dependencies: cross-platform pinned `requirements.txt` (runtime) + `requirements-dev.txt` (jupyter, ruff, pytest); drop `pywinpty` from the runtime set (I2)
- [x] Commit `models/best_overall.joblib` (8 KB) and `best_tree.joblib` via a `.gitignore` exception; delete the duplicate models (I3, I16)
- [x] Repo-root bootstrap in every notebook + a Colab setup cell (I7)
- [x] Tests write the split to `tmp_path` instead of `data/processed/` (I10)
- [x] Seed the SHAP beeswarm (I14); report the unseen-category count (I17)
- [x] Time-ordered inner CV for India Ridge/Lasso (I9): old vs new in the Phase 1 section
- [ ] Record the leakage experiment (§4.3) in notebook 02 as a "mid-term vs now" cell (deferred; re-measured in Phase 1 and recorded in `CHANGELOG.md`)
- [x] Re-run `python run.py build` + `check`; confirm `test_results.csv` is unchanged (max \|diff\| 0.0)
- [x] Refresh the README timings (I8)

### Phase 2: Streamlit app redesign (≈ 12–16 h)

- [x] Move to `st.navigation` multipage with a top bar (Phase 2: 6 pages, top navigation)
- [x] Apply the "Field & Harvest" palette and fonts; add light + dark themes (I12) (Phase 2; fonts bundled offline)
- [x] Responsive breakpoints at 1200 / 900 px; verified at 1920 / 1440 / 1366 / 1280 / 1024 / 768 in both themes (Phase 2, 0 geometry problems)
- [x] EDA page: distributions, correlation heatmap, yield by crop/region/season (Altair, interactive) (Phase 2: Data explorer)
- [x] Prediction band (conformal interval from CV residuals) on the hero/gauge (I6) (Phase 2: ± 1.037 t/ha, 88.5 % test coverage)
- [x] Leaderboard ranked by CV, with test RMSE shown as a reference column (I4) (done in Phase 1)
- [x] Branded error / setup states (I15) and India units wording (I11) (Phase 1 + 2; a custom loading state was not added: Streamlit's own spinner remains)
- [x] Download a prediction report as CSV (inputs, prediction, interval, top SHAP contributions) (Phase 2)

### Phase 3: explainability + what-if (≈ 8–10 h)

- [x] SHAP waterfall per prediction in friendly names; native Altair beeswarm (I13) (done early, in Phase 2)
- [x] Coefficient and permutation-importance views in the app, plus a method-agreement view (Phase 3: permutation, mean |SHAP|, |coefficient|, Spearman ρ)
- [x] 2-factor what-if heatmap and a side-by-side scenario A vs B comparison (Phase 3)
- [x] "Soil pH paradox" explainer card (Phase 3: r = +0.019, SHAP rank 4, interior peak at pH 6.55)
- [x] PDF export of the prediction report (one page) (done early, in Phase 2, with matplotlib)

### Phase 4: model upgrades, compared fairly (≈ 8–12 h)

- [x] Add LightGBM and sklearn `HistGradientBoosting` to the same nested-CV harness (Phase 4: Kaggle comparison rows + India)
- [x] Optuna (or a wider RandomizedSearch) under the same outer folds; report CV mean ± std and test once (Phase 4: seeded RandomizedSearchCV, 20 candidates)
- [x] India: crop×state historical-mean feature (computed inside folds) to try to beat the lookup (Phase 4: earlier-years target encoding + lags; beats it only via lags)
- [ ] Statistical tie check (paired fold differences) so "tie" claims are backed by numbers (deferred: not part of the Phase 4 brief)
- [x] Update tables and figures; keep model selection by CV only (Phase 4)

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
