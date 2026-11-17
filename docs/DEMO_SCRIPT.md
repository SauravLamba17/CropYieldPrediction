# Live demo script: 8 minutes + 2 minutes buffer

CSL236 AIML course project, Team 2: **Saurav Lamba** (24csu188), **Sara Sharma** (24csu184), **Anvi** (24csu024).

**Live app:** https://cropyieldprediction-team.streamlit.app/ (Streamlit Community Cloud). **Fallback:** the local app (`start.bat`) and `docs/screenshots/`.

Every number below is what the app shows from the committed tables (`results/tables/`); if a number on screen differs, say the one on screen. The app opens in **Simple** mode with 5 pages (Home, Predict, What-if, What matters, Real-world check) and in **light** theme. Switch to **Show details** only where the script says so: it adds the exact scores, every chart and 2 more pages (Model comparison, Data explorer).

## Pre-demo checklist (T − 15 min)

- [ ] **Wake the cloud app (T − 10 min):**
  1. Open https://cropyieldprediction-team.streamlit.app/ in the presenting browser.
  2. If it shows "Zzzz. This app has gone to sleep due to inactivity", click **Yes, get this app back up!** and wait for Home to appear. Free-tier apps sleep after 12 hours without traffic; the wake-up time was not measured, which is why this is done 10 minutes early.
  3. Check Home shows **0.65 t/ha**, **69 %** and **89 %** and no "Health check" card.
- [ ] **Warm the caches (T − 8 min):** click through the top navigation once in Simple: Predict, What-if, What matters, Real-world check. Then switch to **Show details**, open Model comparison, What matters and Real-world check, and switch back to **Simple** and **Home**. The first Predict visit builds the explainer; after that the clicks are fast.
- [ ] **Expected speed on the warm live app** (measured 2026-11-15 at 1366 px, headless Chrome, before the Phase 9 layout; not re-measured since):
  - full page load: median **4.6 s** over 34 page loads, range 3.5–10.8 s;
  - page switch through the top navigation: **0.8–1.1 s**, Predict **2.3 s**;
  - mode switch: median **0.5 s**, at most 1.3 s.

  If a page takes much longer than 10 s, switch to plan B.
- [ ] **Fallback running:** on the presenting laptop, double-click `start.bat` (or `python run.py app`) so the local app is already open on `http://localhost:8501` in a second tab.
- [ ] **Projector:** display at **1366 × 768 or 1280 × 720**, browser zoom 100 %, the browser in full screen (F11). Check the header and the top cards fit without scrolling sideways.
- [ ] **Light mode:** ⋮ (top right) → **Light**. A browser opening the app for the first time starts in light on every page, even on a dark-mode laptop.
- [ ] Predict reset: click **Reset to defaults** so the Typical field shows **about 6.5 tonnes per hectare** (6.52 t/ha on the gauge).
- [ ] Phones on silent; one person keeps the timer.

## Segments

| # | Time | Speaker | Page | Length |
|---|---|---|---|---|
| 1 | 0:00–1:00 | Saurav | Home | 1:00 |
| 2 | 1:00–2:15 | Sara | Predict | 1:15 |
| 3 | 2:15–3:15 | Sara | What-if | 1:00 |
| 4 | 3:15–4:15 | Anvi | What matters | 1:00 |
| 5 | 4:15–5:30 | Anvi | Show details → Model comparison | 1:15 |
| 6 | 5:30–7:00 | Saurav | Real-world check | 1:30 |
| 7 | 7:00–8:00 | Saurav (all three on Q&A) | Real-world check → Home | 1:00 |
| | 8:00–10:00 | — | buffer for questions or a slow click | 2:00 |

### 1. What the app does · Saurav · Home · 1:00

**Clicks:** none (start on Home, Simple mode), then **Try a prediction →** at the end.

**Say:** "This app predicts how much a field will harvest from its soil, weather and farming, shows what drives each prediction, and checks the method on real Indian crop records. On 960 fields it never saw, it is typically off by 0.65 tonnes per hectare, it explains 69 % of the differences between fields, and its 90 % range was right 89 % of the time."

**Point at:** the three cards: **Typical error 0.65 t/ha**, **How much it explains 69 %**, **How often the range is right 89 %**.

### 2. A prediction with its range · Sara · Predict · 1:15

**Clicks:** (Predict opens from the button; Typical field is shown) → in the scenarios, click **Drought year** → then **Typical field** again → open **More field conditions** for a second, close it → point at **Download PDF**.

**Say:** "Four inputs matter most, so they are in front: fertilizer, pesticide, rainfall and the crop. The other ten are under More field conditions. For a typical field the app says: this field should produce about 6.5 tonnes per hectare, usually between 5.5 and 7.6. That is a good yield, higher than 56 % of the fields the model learned from, and fertilizer helps this field most. In a drought year the prediction drops. Every prediction can be downloaded as a one-page report."

**Point at:** the sentence at the top, the gauge **6.52 t/ha** with **A good yield: higher than 56 %…**, **Fertilizer helps this field most (+0.10 t/ha)**, and the drop after Drought year.

### 3. What-if and A vs B · Sara · What-if · 1:00

**Clicks:** top nav **What-if** → point at the one-factor chart (Soil pH is selected) → scroll to **Scenario A vs B** → open **Scenario B inputs**, move **Fertilizer** to the right → point at **B − A** and the headline above the bars.

**Say:** "Here we sweep one factor for the same field: soil pH works best around pH 6.6. In A versus B we copy the field, change one thing, and the app says how much more or less B should yield, with the bars showing which factors explain the gap."

**Point at:** **Soil pH works best around pH 6.6**, and the **B − A** gap (say the number on screen).

### 4. What matters · Anvi · What matters · 1:00

**Clicks:** top nav **What matters** → scroll to **The soil pH surprise**.

**Say:** "Fertilizer matters most, by far: scramble it and the typical error grows by 0.73 tonnes per hectare, six times more than the next factor. Three different ways of measuring all put it first. The surprise is soil pH: a straight-line check finds almost no link with yield, +0.02 on a scale from −1 to 1, but the model's prediction rises and then falls, best around pH 5.9 to 7.2. A hump has no straight-line trend, so that check misses it."

**Point at:** **Fertilizer matters most, by far** (bar **0.733**), **Soil pH works best around pH 6.6**, the shaded best range **5.9–7.2**.

### 5. For the examiners: Show details · Anvi · Model comparison · 1:15

**Clicks:** switch the header to **Show details** (the nav now lists 7 pages) → top nav **Model comparison** → point at the KPI row → scroll to **Statistically distinguishable?** → back to **What matters** for **Do the methods agree?**

**Say:** "Show details gives the exact scores and two more pages. All models are compared on the same five cross-validation folds, and the lowest CV error is chosen before the test set is opened. The curved formula, a degree-2 polynomial, wins with 0.635, and the test set is used once: 0.65 tonnes per hectare, 44 % better than guessing the average. The top five are within 0.01 of each other, so we tested whether the differences are real. They are not: on the five folds the polynomial cannot be told apart from XGBoost, p = 0.71, or from HistGradientBoosting, p = 0.25. It's a statistical tie, so our fixed rule decides, and it gives us the smallest and most readable model. The three importance methods agree with an average Spearman correlation of 0.79."

**Point at:** **0.65 t/ha**, **44 %**, the tie-check table, **ρ 0.89 / 0.73 / 0.74**.

### 6. Real data: India · Saurav · Real-world check · 1:30

**Clicks:** switch back to **Simple** → top nav **Real-world check** → point at the three lessons and **How the model beats the look-up** → switch to **Show details** → point at the KPI row, the feature-group chart and **Statistically distinguishable?**

**Say:** "On real Indian records the benchmark story does not transfer. The hard reference is a simple look-up of each crop's usual yield in each state: it is off by 7.81. Our model, a random forest with time-safe history features, is off by 6.85, so it wins. But be honest about why: just repeating last year's yield, with no model at all, scores 6.95. The feature-group chart shows the gain comes almost entirely from last years' yields. And if we only knew data up to 2015, the model would score 9.65, worse than the look-up. Our tie check agrees: across the five validation periods the model is distinguishable from the look-up, p = 0.04, but not from simply repeating last year's yield, p = 0.68."

**Point at:** **6.85 vs 7.81**, **last year's yield 6.95**, the **+ Lags** bar (− 0.74 CV), and the strict **9.65**.

### 7. Limitations and future work · Saurav · Real-world check → Home · 1:00

**Clicks:** switch back to **Simple**, top nav **Home**.

**Say:** "Three limitations. The Kaggle data behaves like synthetic data, so its findings are a controlled benchmark, not agronomy. On India, the inputs are coarse: one rainfall number per state and national fertilizer figures, so most of what any model can learn is the crop, the state and last year. And the India result assumes last year's yield is known when predicting. With more time we would add district-level weather and soil data, report a strict multi-year forecast as the headline, and widen the 90 % range per crop where it under-covers. Thank you, we're happy to take questions."

**Point at:** the Home "The data" card (the two dataset links).

## Plan B: if the internet fails

1. **Local app (preferred):** switch to the `http://localhost:8501` tab started with `start.bat` during the checklist. It is the same code and the same committed tables, fully offline (fonts are bundled). Follow the same script.
2. **If the laptop's app will not start:** open these screenshots from `docs/screenshots/` full screen, in order, and talk over them with the same script:
   - Home: `p9_simple_light_home_1366.png`
   - Predict: `p9_simple_light_predict_1366.png`
   - What-if: `p9_simple_light_whatif_1366.png`
   - What matters: `p9_simple_light_what_matters_1366.png`
   - Model comparison: `p2_technical_light_model_comparison_1440_full.png`
   - Real-world check: `p4_simple_dark_india_1440.png`, then `p4_technical_light_india_1440.png`
   - The p2–p4 captures predate the tie-check card and the Phase 9 names ("Technical" is now "Show details"); say the tie-check result from this script.
3. **Numbers to say without any screen:** test RMSE 0.65 t/ha, R² 0.69, 90 % range ± 1.04 t/ha held 88.5 %; India 6.85 vs look-up 7.81, last year's yield 6.95, strict 9.65.
