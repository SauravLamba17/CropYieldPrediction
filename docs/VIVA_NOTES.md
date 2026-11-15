# Viva notes: final Q&A sheet

CSL236 AIML, Team 2: Saurav Lamba (24csu188), Sara Sharma (24csu184), Anvi (24csu024).

Every number comes from the committed tables in `results/tables/` or from a measured run recorded in `PROGRESS.md`. Kaggle errors are in t/ha. India errors are in "yield units": mostly t/ha, but bales per hectare for cotton, jute, mesta and sannhemp.

---

## Data

**Q: Is the Kaggle data real?**
It behaves like synthetic data. The largest correlation between any two inputs is |r| = 0.03, and the histograms are flat. Crop, region and season barely change yield. So it is treated as a controlled benchmark, and every finding is re-checked on real Indian data.

**Q: How big are the datasets and how are they split?**
Kaggle has 4,800 labelled fields after removing duplicates: 3,840 train and 960 test (an 80/20 shuffled split, seed 42). India has 19,412 state × crop × season records, split by time: train 1997–2015 (15,186 rows), test 2016–2020 (4,226 rows).

**Q: Any target leakage in the India data?**
Yes, and it was removed. `Production` is Yield × Area, so it was dropped. Fertilizer and pesticide were farm-size totals, so they became per-hectare values. A test checks that `Production` never reaches a model.

## Preprocessing

**Q: At mid-term you scaled before cross-validation. Did that leak?**
We measured it. Ridge CV RMSE was 0.640771 either way, a difference of 5.5 × 10⁻⁸ t/ha. Every step now sits inside a `Pipeline`, so each fold fits its own scaler and encoder; that is correct by construction.

**Q: What happens with a category the model has never seen?**
One-hot encoding uses `handle_unknown="ignore"`, so an unseen category is all zeros and the model falls back to its baseline. On India this affects 10 of the 4,226 test rows ("Other Summer Pulses"); the look-up falls back to the overall training mean.

## Models

**Q: XGBoost has a lower test RMSE. Why deploy the polynomial model?**
Selection is by CV before the test set is opened: polynomial 0.6349 ± 0.0088 vs XGBoost 0.6362 ± 0.0073. On test, XGBoost is ahead by 0.00006 t/ha, a tie. Choosing on the test set would make the reported score optimistic.

**Q: Is 0.65 t/ha good?**
Test RMSE is 0.646 (R² 0.69) against 1.160 for always guessing the average, 44 % lower. It is near the noise floor: residuals are normal (Jarque-Bera p = 0.60), and six different models land between 0.646 and 0.664.

**Q: Did LightGBM or HistGradientBoosting change the Kaggle result?**
No. With the same folds and search budget, HistGB scores CV 0.639 ± 0.009 and LightGBM 0.647 ± 0.008, against the polynomial's 0.635. They are comparison rows only; the deployed model and its files are unchanged.

## Validation

**Q: Why nested cross-validation for the trees?**
Random Forest and XGBoost are tuned inside each outer fold (`RandomizedSearchCV`, 20 configurations). Tuning on the folds you score on would reuse validation data and flatter the trees. RidgeCV and LassoCV re-tune α inside each fold the same way.

**Q: Are the differences between the top models real? (tie check)**
No. Paired t-test on the 5 CV folds: polynomial vs XGBoost, mean difference −0.0013 ± 0.0072 t/ha, p = 0.71 (Wilcoxon 0.81). Vs HistGradientBoosting, −0.0043 ± 0.0073, p = 0.25 (Wilcoxon 0.19). It is a statistical tie, so the fixed CV rule decides, and it picks the smallest, most readable model.

**Q: Why can the Wilcoxon test never be significant here?**
With 5 folds, an exact two-sided Wilcoxon test has a smallest possible p-value of 2 / 2⁵ = 0.0625, even when all 5 folds agree. So "distinguishable" uses the paired t-test (p < 0.05). Folds also share training rows, which makes both tests optimistic.

**Q: How is the 90 % range computed, and does it cover 90 %?**
Split-conformal on out-of-fold residuals: the half-width is the ⌈(n + 1)·0.9⌉-th absolute residual, ± 1.037 t/ha. On the 960 test fields it covered 88.5 % (850). The standard error is about 1 point, so this is within sampling variation.

**Q: Is the range equally good everywhere?**
No. By crop, coverage runs from 85.9 % (Rice) to 91.1 % (Barley); by region, from 85.9 % (West) to 92.0 % (North). One width fits all fields; a per-group (Mondrian) width would even this out.

## Explainability

**Q: Three importance methods: do they agree?**
All three rank fertilizer first, then pesticide and rainfall. Spearman ρ: permutation vs SHAP 0.895, permutation vs coefficient 0.732, SHAP vs coefficient 0.736 (average 0.79). The biggest disagreement is soil pH: rank 4 (SHAP), 8 (permutation), 13 (coefficient).

**Q: What is the soil pH paradox?**
Pearson r = +0.019 with yield, so a correlation filter would drop it. Yet the model's average prediction rises from 6.14 t/ha at pH 4.5 to 6.41 at pH 6.55, then falls to 6.17 at pH 8.5. A hump has no straight-line trend.

**Q: Why two SHAP methods?**
TreeExplainer is exact and fast for XGBoost (global importance, beeswarm). The deployed polynomial is not a tree, so each prediction is explained with Kernel SHAP in the original feature space; the waterfall adds each factor's push to the average.

## India

**Q: How is the crop × state look-up computed, and can it leak?**
It is the mean log(1 + yield) of the training years (1997–2015) for that crop in that state, with seasons pooled; 147 unseen test pairs fall back to the training mean. It reads no test-year yield. CV 5.97 ± 2.89, test 7.81.

**Q: Did machine learning beat the look-up?**
Under the one-year-ahead assumption, yes. Random forest with all features scores CV 5.12 vs 5.97 and test 6.85 vs 7.81 (R² 0.75 vs 0.68), and beats the look-up for 48 of 54 crops. But repeating last year's yield, with no model, already scores 6.95.

**Q: One-step vs strict: which result is the headline?**
The headline is one-step-ahead (6.85): forecasting year t knows the actual yields up to t − 1, as a seasonal forecast would. Always shown next to it: last year's yield (6.95) and the strict variant, where only 1997–2015 is known, which scores 9.65, worse than the look-up.

**Q: Is the model really better than repeating last year's yield? (tie check)**
Not in CV. Against last year's yield, the mean fold difference is −0.29 ± 1.42, p = 0.68 (Wilcoxon 1.00): a tie. Against the look-up, −0.85 ± 0.63, p = 0.040 (Wilcoxon 0.0625, its floor): distinguishable, but weakly. Against HistGB (all features), p = 0.36. The app's verdict says this.

**Q: Why do the lags dominate?**
Yields persist: each crop × state has a stable level that drifts slowly. The 3-year rolling mean carries 71 % of mean |SHAP| and last year's yield 9 %. In the ablation, lags alone cut CV RMSE by 0.74 (5.89 → 5.14).

**Q: Why did target encoding alone make it worse (5.89 → 6.05)?**
Most likely because the random forest already learns each crop × state level from the one-hot columns, so the encodings add little. Being time-safe, they are also noisy: rows from early years have only a few earlier records to average.

**Q: Why do year and pesticide have zero permutation importance?**
Every test year (2016–2020) lies beyond the training range, and so does every test year's pesticide value. Trees cannot extrapolate: all those rows fall in the same last split, so shuffling them among test rows changes nothing.

**Q: What is the hybrid, and why didn't it win?**
The look-up plus a random forest trained on the look-up's residuals, both time-safe. CV RMSE was 6.06, slightly worse than the look-up alone (5.97): once the look-up removes each crop × state level, the other inputs leave little to learn.

## Deployment and engineering

**Q: How is the app deployed?**
On Streamlit Community Cloud (https://cropyieldprediction-team.streamlit.app/) from the public GitHub repo, Python 3.12. It installs only `requirements.txt`: the app runtime, with CPU-only xgboost on Linux. That is 782 MB, down from 1,097 MB. Nothing is trained on the server; it reads the committed models and tables.

**Q: Does the app send data anywhere?**
Locally the app makes no outside requests. On Streamlit Community Cloud, the platform still sends Streamlit's own usage statistics, which the repo can't switch off. The fonts are bundled in the repo and the app needs no secrets or API keys.

**Q: How do you know it works on Linux?**
GitHub Actions CI on ubuntu-latest, Python 3.12, with no secrets. One job runs ruff and the whole test suite (raw-data tests skip, as on a fresh clone). A smoke job installs only `requirements.txt` and renders every page in both modes.

**Q: Why does the full build take two hours, and what is `--fast`?**
The full build takes 129 min 44 s, of which India is 103 min of nested tree searches and the ablation. `python run.py build --fast` reruns the Kaggle pipeline and notebooks but reuses the committed India tables, in 33 min 35 s. The trade-off: the India numbers are then not regenerated.

**Q: What would you do with more time?**
Add district-level weather and soil data to India; report a strict multi-year forecast as a second headline; use a per-crop (Mondrian) 90 % range; save per-fold CV scores in the pipelines (the tie check had to rerun them); and use more folds for stronger tests.
