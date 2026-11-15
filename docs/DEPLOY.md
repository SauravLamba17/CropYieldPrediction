# Deploying to Streamlit Community Cloud

The app runs on Streamlit Community Cloud straight from the public GitHub repository. Nothing is trained on the server: it reads the committed model files (`models/`), results tables (`results/tables/`) and the processed splits (`data/processed/`). No secrets, no API keys, no network access at runtime (the fonts are bundled in `static/fonts/`).

Facts about Community Cloud below are from the Streamlit docs (docs.streamlit.io → Deploy → Streamlit Community Cloud) as read on 2026-11-15; check them again if a screen looks different.

## What the server installs

Community Cloud looks for a dependency file in the entry point's folder, then in the repository root, and uses the first it finds in this order: `uv.lock`, `Pipfile`, `environment.yml`, `requirements.txt`, `pyproject.toml`. This repository has only `requirements.txt` (the root `pyproject.toml` holds ruff/pytest settings and no dependencies, and comes later in the order), so the server installs **`requirements.txt` = the app runtime only**: 12 direct packages pinned to the exact versions that produced the results, with the CPU-only `xgboost-cpu` on Linux (no NVIDIA libraries).

| | Linux, Python 3.12 (measured with `uv`) |
|---|---|
| `requirements.txt` (what Cloud installs) | 55 packages, **782 MB** installed |
| Before the split (old `requirements.txt`) | 1,097 MB, including 205 MB of NVIDIA NCCL and the 84 MB GPU build of xgboost |
| `requirements-dev.txt` (pipelines, notebooks, tests) | 1,320 MB |

Installing `requirements.txt` into a fresh venv took **5 min 04 s** on the development laptop (Windows 11, Python 3.14.4; 788 MB on disk there); the time on Cloud will differ, so allow several minutes for the first deploy.

## Deploy, click by click

1. Push `main` to GitHub (`SauravLamba17/CropYieldPrediction`) and check that the **CI** workflow is green on the Actions tab (see the end of this page).
2. Open **share.streamlit.io** and sign in with the GitHub account that owns the repository (the docs say deploying needs admin rights on the repo).
3. Click **Create app** (upper right). When asked "Do you already have an app?", click **Yup, I have an app.**
4. Fill in:
   - **Repository:** `SauravLamba17/CropYieldPrediction`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL** (optional): pick a subdomain, e.g. `crop-yield-team2`.
5. Click **Advanced settings**:
   - **Python version:** choose **3.12** (the Cloud default; CI also tests 3.12). The Python version cannot be changed later without deleting and redeploying the app.
   - **Secrets:** leave empty. The app needs none.
   - Click **Save**.
6. Click **Deploy**. The log panel shows the `pip install`; the first install takes several minutes, then the app opens.
7. Copy the app URL into the README: replace `LIVE_DEMO_URL` (two places at the top) and commit.

## After the deploy: what to check

Open the URL in a private window (no cached theme) and check, in this order:

- [ ] No "Health check" card under the header (it lists any results table that is missing).
- [ ] **Overview** shows 0.63 t/ha (CV), 0.65 t/ha (test), 69 % and ± 1.04 t/ha.
- [ ] **Predict**: the Typical field predicts **6.52 t/ha (90 % range 5.48–7.56)**; pick **Drought year** and the number drops; **Download PDF** works.
- [ ] **What-if**: the heatmap draws; in Scenario A vs B, move a B slider and the gap changes.
- [ ] **Model comparison**: Technical mode shows the tie-check card.
- [ ] **Explain**: the method-agreement card (ρ 0.89 / 0.73 / 0.74) and the soil pH paradox card.
- [ ] **Data explorer**: charts draw. The harvest-month card shows "This chart needs the raw Kaggle file" on Cloud, because the raw CSV is not in git; that is expected.
- [ ] **India**: Simple shows 6.85 vs 7.81 and the "last year's yield 6.95" note; Technical shows the leaderboard, the ablation chart, the 10 worst crops and the tie-check card.
- [ ] Switch **Simple ↔ Technical** on two pages, and **Light ↔ Dark** from the ⋮ menu (top right) → Settings.
- [ ] Narrow the browser window to about 768 px: nothing overlaps.

Measured locally with only `requirements.txt` installed (the same set Cloud installs): the server answered its health check **2.2 s** after launch and the first page was fully drawn after **14.6 s** (cold start, measured with a headless browser that was itself starting); peak memory was **468 MB** (resident set of the server process; Windows peak working set 498 MB) after visiting all 7 pages in both modes and both themes, with 0 requests to any outside host. The docs give Community Cloud apps between 690 MB and 2.7 GB of memory, so the app fits, but avoid opening it in many tabs at once on the free tier.

## Reboot, clear the cache, read the logs

- **Logs:** open the app, click **Manage app** (lower right). **Download log** is in that panel's ⋮ menu.
- **Reboot:** in your workspace (share.streamlit.io), click ⋮ next to the app → **Reboot** → confirm **Reboot**. Or from **Manage app** → ⋮ → **Reboot app**. Visitors see "Your app is in the oven" while it restarts.
- **Clear the cache:** the app's caches live in memory, so a **Reboot** clears them all. (The app sets `toolbarMode = "viewer"`, which hides Streamlit's developer menu items such as "Clear cache" from visitors.) Reboot after pushing new results tables.
- **New commit:** Cloud redeploys automatically when `main` changes. If a dependency changed, a reboot forces a clean install.

## Sleep on the free tier, and waking the app before the presentation

The docs say apps **without traffic for 12 hours go to sleep**. A sleeping app shows "Zzzz. This app has gone to sleep due to inactivity." with a **Yes, get this app back up!** button, which anyone who can view the app may click. The docs do not state how long waking takes, so:

1. **10 minutes before** the presentation, open the URL and click **Yes, get this app back up!** if it is asleep.
2. When it loads, visit **every page once** in the mode you will present (this builds the SHAP explainer and fills the caches, so the live clicks are instant).
3. Keep that tab open. Do not reboot or clear the cache after warming up.
4. Keep the local fallback ready (`start.bat`, see `docs/DEMO_SCRIPT.md`, "Plan B").

## CI on GitHub (what to check after pushing)

`.github/workflows/ci.yml` runs on every push and pull request to `main`, on `ubuntu-latest` with Python 3.12 and no secrets:

- **Lint and tests (requirements-dev.txt):** ruff, then the whole pytest suite. Tests that need the git-ignored raw CSVs skip, exactly as on a fresh clone; the skip reasons are listed at the end of the log.
- **App smoke test (requirements.txt only):** installs only the runtime set (plus pytest to drive Streamlit's AppTest), checks that statsmodels/seaborn/jupyter and any NVIDIA package are absent, then renders every page in both modes and runs the health-card test. This proves the Cloud set is enough.

On the repository's **Actions** tab, the run named after your commit should show two green jobs. The README's CI badge turns green after the first passing run on `main`.
