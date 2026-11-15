# Live demo script: 8 minutes + 2 minutes buffer

CSL236 AIML course project, Team 2: **Saurav Lamba** (24csu188), **Sara Sharma** (24csu184), **Anvi** (24csu024).

**Live app:** https://cropyieldprediction-team.streamlit.app/ (Streamlit Community Cloud). **Fallback:** the local app (`start.bat`) and `docs/screenshots/`.

Every number below is what the app shows from the committed tables (`results/tables/`); if a number on screen differs, say the one on screen. Start in **Simple** mode and **light** theme; switch to Technical only where the script says so.

## Pre-demo checklist (T − 15 min)

- [ ] **Wake the cloud app (T − 10 min):**
  1. Open https://cropyieldprediction-team.streamlit.app/ in the presenting browser.
  2. If it shows "Zzzz. This app has gone to sleep due to inactivity", click **Yes, get this app back up!** and wait for the Overview to appear. Free-tier apps sleep after 12 hours without traffic; the wake-up time was not measured, which is why this is done 10 minutes early.
  3. Check the Overview shows **0.63 / 0.65 t/ha, 69 %, ± 1.04 t/ha** and no "Health check" card.
- [ ] **Warm the caches (T − 8 min):** click through the top navigation once in Simple: Predict, What-if, Model comparison, Explain, Data explorer, India. Then switch to **Technical** on Model comparison, Explain and India, and back to **Simple** and **Overview**. The first Predict visit builds the SHAP explainer; after that the clicks are fast.
- [ ] **Expected speed on the warm live app** (measured 2026-11-15 at 1366 px, headless Chrome):
  - full page load: median **4.6 s** over 34 page loads, range 3.5–10.8 s;
  - page switch through the top navigation: **0.8–1.1 s**, Predict **2.3 s**;
  - Simple → Technical switch: median **0.5 s**, at most 1.3 s.

  If a page takes much longer than 10 s, switch to plan B.
- [ ] **Fallback running:** on the presenting laptop, double-click `start.bat` (or `python run.py app`) so the local app is already open on `http://localhost:8501` in a second tab.
- [ ] **Projector:** display at **1366 × 768 or 1280 × 720**, browser zoom 100 %, the browser in full screen (F11). Check the header and the top cards fit without scrolling sideways.
- [ ] **Light mode:** ⋮ (top right) → **Light**. A browser opening the app for the first time starts in light on every page, even on a dark-mode laptop.
- [ ] Predict reset: click **Reset to defaults** so the Typical field shows **6.52 t/ha**.
- [ ] Phones on silent; one person keeps the timer.

## Segments

| # | Time | Speaker | Page | Length |
|---|---|---|---|---|
| 1 | 0:00–1:00 | Saurav | Overview | 1:00 |
| 2 | 1:00–2:15 | Sara | Predict | 1:15 |
| 3 | 2:15–3:15 | Sara | What-if | 1:00 |
| 4 | 3:15–4:30 | Anvi | Model comparison | 1:15 |
| 5 | 4:30–5:30 | Anvi | Explain | 1:00 |
| 6 | 5:30–7:00 | Saurav | India | 1:30 |
| 7 | 7:00–8:00 | Saurav (all three on Q&A) | India / Overview | 1:00 |
| | 8:00–10:00 | — | buffer for questions or a slow click | 2:00 |

### 1. Problem and data · Saurav · Overview · 1:00

**Clicks:** none (start on Overview, Simple mode).

**Say:** "We want to predict how many tonnes per hectare a field will yield from its soil, weather and farming. We trained 8 models on 3,840 fields, picked the best by cross-validation, and tested it once on 960 fields it never saw. Then we re-checked everything on real Indian government crop records from 1997 to 2020."

**Point at:** the four KPI cards: typical error **0.65 t/ha** on unseen fields, **69 %** of the differences explained, and the 90 % range **± 1.04 t/ha**, which held **88.5 %** of the time.

### 2. A prediction with its range · Sara · Predict · 1:15

**Clicks:** top nav **Predict** → (Typical field is shown) → in **Scenario**, click **Drought year** → then click **Typical field** again → point at **Download PDF**.

**Say:** "For a typical field the model predicts 6.52 tonnes per hectare. It doesn't just give one number: the real yield usually lands between 5.48 and 7.56, and on the 960 test fields that range held 89 % of the time. In a drought year the prediction drops, and the 'What moved it' bars show which factor pushed it. Every prediction can be downloaded as a one-page report."

**Point at:** **6.52 t/ha**, the range **5.48–7.56 t/ha**, and the drop after Drought year.

### 3. What-if and A vs B · Sara · What-if · 1:00

**Clicks:** top nav **What-if** → scroll to **Two factors at once** (Fertilizer across, Rainfall up) → scroll to **Scenario A vs B** → open **Scenario B inputs**, move **Fertilizer** to the right → point at **B − A**.

**Say:** "Here we change two factors at once. The best cell in this grid, about 300 kg of fertilizer with 1,154 mm of rain, gives 7.94 tonnes, against 6.52 now. The grid stops at the edges of the training data, because outside them the model would only be guessing. In A versus B we copy the field, change one thing, and see the difference with both ranges."

**Point at:** the best cell **7.94 t/ha** vs **6.52 t/ha**, and the **B − A** gap.

### 4. Model comparison and the tie check · Anvi · Model comparison · 1:15

**Clicks:** top nav **Model comparison** (Simple) → point at the leaderboard → switch the header to **Technical** → scroll to **Statistically distinguishable?** → switch back to **Simple**.

**Say:** "All models are compared on the same five cross-validation folds, and the lowest CV error is chosen before the test set is opened. The curved formula, a degree-2 polynomial, wins with 0.635, and the test set is used once: 0.65 tonnes per hectare, 44 % better than guessing the average. The top five are within 0.01 of each other, so we tested whether the differences are real. They are not real: on the five folds the polynomial cannot be told apart from XGBoost, p = 0.71, or from HistGradientBoosting, p = 0.25. It's a statistical tie, so our fixed rule decides, and it gives us the smallest and most readable model."

**Point at:** **0.65 t/ha**, **44 %**, the note "The top 5 are within 0.01 t/ha of each other", and the tie-check card.

### 5. Why the model decides · Anvi · Explain · 1:00

**Clicks:** top nav **Explain** → point at **Do the methods agree?** → scroll to **The soil pH paradox**.

**Say:** "We used three different importance methods, and all three rank fertilizer first; their rankings agree with an average Spearman correlation of 0.79. The interesting case is soil pH: its correlation with yield is almost zero, 0.02, so a correlation filter would drop it. But the model's prediction rises and then falls, with a peak around pH 6.6. A hump has no straight-line trend, which is all correlation measures."

**Point at:** **ρ 0.89 / 0.73 / 0.74**, **r = +0.019**, the peak at **pH 6.6** (best range 5.9–7.2).

### 6. Real data: India · Saurav · India · 1:30

**Clicks:** top nav **India** (Simple) → point at the three lessons and **How the model beats the look-up** → switch to **Technical** → point at the KPI row, the ablation chart and **Statistically distinguishable?**

**Say:** "On real Indian records the benchmark story does not transfer. The hard baseline is a simple look-up of each crop's usual yield in each state: it is off by 7.81. Our model, a random forest with time-safe history features, is off by 6.85, so it wins. But be honest about why: just repeating last year's yield, with no model at all, scores 6.95. The ablation shows the gain comes almost entirely from the lag features. And if we only knew data up to 2015, the model would score 9.65, worse than the look-up. Our tie check agrees: across the five validation periods the model is distinguishable from the look-up, p = 0.04, but not from simply repeating last year's yield, p = 0.68."

**Point at:** **6.85 vs 7.81**, **last year's yield 6.95**, the **+ lags** bar (− 0.74 CV), and the strict **9.65**.

### 7. Limitations and future work · Saurav · India → Overview · 1:00

**Clicks:** switch back to **Simple**, top nav **Overview**.

**Say:** "Three limitations. The Kaggle data behaves like synthetic data, so its findings are a controlled benchmark, not agronomy. On India, the inputs are coarse: one rainfall number per state and national fertilizer figures, so most of what any model can learn is the crop, the state and last year. And the India result assumes last year's yield is known when predicting. With more time we would add district-level weather and soil data, report a strict multi-year forecast as the headline, and widen the 90 % range per crop where it under-covers. Thank you, we're happy to take questions."

**Point at:** the Overview "The data" card (the two dataset links).

## Plan B: if the internet fails

1. **Local app (preferred):** switch to the `http://localhost:8501` tab started with `start.bat` during the checklist. It is the same code and the same committed tables, fully offline (fonts are bundled). Follow the same script.
2. **If the laptop's app will not start:** open these screenshots from `docs/screenshots/` full screen, in order, and talk over them with the same script:
   - Overview: `p2_simple_light_overview_1920.png`
   - Predict: `p2_simple_light_predict_1440.png`
   - What-if: `p3_technical_dark_whatif_1440_full.png`
   - Model comparison: `p2_technical_light_model_comparison_1440_full.png`
   - Explain: `p3_technical_light_explain_1440_full.png`
   - India: `p4_simple_dark_india_1440.png`, then `p4_technical_light_india_1440.png`
   - After this phase they predate the tie-check card; say its result from this script.
3. **Numbers to say without any screen:** test RMSE 0.65 t/ha, R² 0.69, 90 % range ± 1.04 t/ha held 88.5 %; India 6.85 vs look-up 7.81, last year's yield 6.95, strict 9.65.
