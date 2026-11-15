# Viva notes: likely questions, short answers

Every number here comes from the committed results tables (`results/tables/`) or from the measured runs recorded in `PROGRESS.md`. Kaggle errors are in t/ha; India errors are in "yield units" (mostly t/ha, bales per hectare for cotton, jute, mesta and sannhemp).

---

## Model selection

**Q: XGBoost has a lower test RMSE than your deployed Polynomial model. Why deploy Polynomial?**
Because the model is chosen by **cross-validation, before the test set is opened**. On the same 5 folds, Polynomial + Ridge scores CV RMSE **0.6349 ± 0.0088** and XGBoost **0.6362 ± 0.0073**, so Polynomial wins the selection. On the test set XGBoost is ahead by **0.00006 t/ha** (0.6461 vs 0.6462), which is a tie. Picking whichever model looks best on the test set would turn the test set into a second validation set and make the reported score optimistic. As a bonus, the polynomial model is smaller (7.8 KB vs 770 KB), faster, and its effects can be read directly.

**Q: Then what is the test set for?**
It is used **once**, after every choice is made, to report an honest error for the chosen model: test RMSE **0.646 t/ha**, R² **0.69**, against **1.160** for always predicting the average (44 % lower). The app says this on the Model comparison page: "Selected by cross-validation; test set used once for final evaluation."

**Q: Why nested cross-validation for the trees?**
The Random Forest and XGBoost hyperparameters are tuned *inside* each outer fold (`RandomizedSearchCV`, 20 configurations). Tuning on the same folds you score on would reuse the validation data and overstate the trees' accuracy. RidgeCV and LassoCV do the same for their α automatically.

**Q: Is 0.65 t/ha good? Why not better?**
It is close to the data's noise floor. The residuals are normal (Jarque-Bera p = 0.60) with constant spread (Breusch-Pagan p = 0.41), and six quite different models (linear, polynomial, Ridge, Lasso, Random Forest, XGBoost) all land between 0.646 and 0.664 on the test set. When very different models agree like that, the remaining error is mostly noise that no model can explain.

---

## Leakage

**Q: At mid-term you scaled the data before cross-validation. Did that leak?**
In principle yes: every fold's scaler had seen its validation rows. We measured it. Ridge, 5-fold CV RMSE on the 3,840 training rows: **0.640771** with the scaler fitted before CV and **0.640771** with the scaler inside a `Pipeline`, a difference of **5.5 × 10⁻⁸ t/ha**. It is negligible here because each training fold already holds 80 % of near-uniform features, so the mean and spread barely change. Every preprocessing step now sits inside the Pipeline anyway, because that is correct by construction and the effect would not be negligible on smaller or shifting data.

**Q: Any target leakage in the India data?**
Yes, and it was removed. `Production` is Yield × Area, so it was dropped. Fertilizer and Pesticide were raw totals that grow with farm size, so they were converted to per-hectare values. A test checks that `Production` never reaches a model.

---

## India (real-world check)

**Q: Why does a simple crop × state look-up beat every machine-learning model?**
The look-up scores RMSE **7.81**; the best model, Random Forest, scores **8.12**. Most of the predictable signal on real data is simply "what does this crop usually yield in this state". Crop type alone carries **83 %** of the best model's SHAP importance, and yields differ by about 100× between crops. The other inputs are weak: one annual rainfall figure per state, and fertilizer and pesticide that are national yearly averages. Within a single crop, the models largely fail: for **8 of the 10** most common crops, R² is below zero.

**Q: You changed how the India penalties are tuned. What changed and what did it do?**
Before, Ridge/Lasso chose α with folds where later years could inform the choice (RidgeCV's leave-one-out and LassoCV's unshuffled folds). Now α is tuned by a grid search on **time-ordered year blocks**, the same scheme as the evaluation, still on the log scale, so only the folds changed. Effect on the 2016–2020 test years:

| Model | Test RMSE before → after |
|---|---|
| Ridge | 9.5169 → 9.5169 |
| Lasso | 9.5497 → 9.5173 |
| Polynomial (degree 2) | 9.4807 → 9.7175 |
| Polynomial (degree 3) | 9.8373 → 10.0209 |

CV RMSE rose for the tuned models (e.g. polynomial 7.916 → 8.977) because the estimate is now honest. The selected model (Random Forest), the look-up result and every headline conclusion are unchanged. One detail did change: **after making India's penalty tuning time-ordered, degree-2 polynomial no longer beats plain linear regression on the test years** (9.72 vs 9.51).

**Q: What happens with a crop the model has never seen?**
`OneHotEncoder(handle_unknown="ignore")` encodes it as all zeros, so the model falls back to its baseline for that column instead of crashing. On India, **10 of the 4,226** test rows are such a crop ("Other Summer Pulses" only appears in 2016–2020); no state or season is unseen. The look-up baseline falls back to the overall training mean. Both cases are covered by tests.

---

## The 90 % range (conformal interval)

**Q: How is the "90 % range" on the Predict page computed?**
It uses split-conformal prediction on **out-of-fold** residuals. Every training field is predicted by the deployed model trained on the other 4 folds (the same 5 folds as the model comparison). The interval half-width is the ⌈(n + 1) · 0.9⌉-th smallest absolute residual, with n = 3,840: **± 1.037 t/ha**. It is computed once in the pipeline and saved to `results/tables/conformal.json`; the app only reads it. Example: the Typical field predicts **6.52 t/ha (90 % range 5.48–7.56)**.

**Q: Does it actually cover 90 %?**
On the 960 test fields it covered **88.5 %** (850 of 960). That is 1.5 points short of the target. With 960 fields the standard error of a 90 % coverage estimate is about 1 point, so this is within normal sampling variation; we report the measured value, not 90 %. Conformal coverage is guaranteed on average over many datasets, assuming the test fields look like the training fields, and the interval has the same width for every field: it does not adapt to harder fields.

---

## Data and explainability

**Q: Is the Kaggle data real?**
It behaves like synthetic data. The largest correlation between any two inputs is **|r| = 0.03**, the numeric inputs have flat (uniform) histograms, and crop, region and season barely change yield. That is why every finding is re-checked on the real India data.

**Q: What drives yield on the benchmark?**
Fertilizer, then pesticide (negative), then rainfall, with the same top three under coefficients, permutation importance and SHAP. Farming practices make up **64 %** of the model's reliance (mean |SHAP|). Soil pH is the clearest non-linear case: its correlation with yield is about zero, but its effect peaks around pH 6.6. A straight line misses it; SHAP and the what-if curve show it.

**Q: Why two SHAP methods?**
TreeExplainer is exact and fast for XGBoost (used for global importance). The deployed polynomial model is not a tree, so the app explains each prediction with Kernel SHAP in the original feature space. The waterfall on the Predict page starts at the average prediction and adds each factor's push to reach this field's prediction.

---

## Explainability (Phase 3)

**Q: You show three importance methods. What does each measure, and why can they disagree?**
- *Permutation importance* shuffles one input across the 960 test fields and measures how much the test RMSE grows (30 repeats). It sees the model's whole behaviour, curves included. When two inputs carry the same information it can split the credit between them.
- *Mean |SHAP|* is the average size of each input's push on single predictions. It also sees curves and interactions; for the polynomial model it comes from Kernel SHAP, for XGBoost from TreeExplainer.
- *|Standardised coefficient|* is a straight-line slope per 1 sd in the multiple linear model. It cannot see a curve: a hump-shaped effect has a slope near zero.

They disagree for exactly those reasons: non-linearity (soil pH), polynomial terms (squares and products credit an input through several terms) and correlated inputs (shared credit). Here they agree well where it matters. All three rank **fertilizer first**, then pesticide and rainfall. The Spearman rank correlations for the deployed model are **0.895** (permutation vs SHAP), **0.732** (permutation vs coefficient) and **0.736** (SHAP vs coefficient), average **0.79**. The biggest disagreement is **soil pH**: rank **4** by SHAP, **8** by permutation, **13** by coefficient.

**Q: Tell us about the soil pH paradox.**
On the 3,840 training fields soil pH has **Pearson r = +0.019** and **Spearman ρ = +0.018** with yield, so a correlation filter would drop it. Yet the deployed model's average prediction rises from **6.14 t/ha at pH 4.5** to a peak of **6.41 t/ha at pH 6.55**, then falls to **6.17 t/ha at pH 8.5**. Within 10 % of the peak, the optimum runs from pH **5.9 to 7.2**. A hump has almost no straight-line trend, which is all that correlation measures. The quadratic-term test on the training data agrees (p = 4.4 × 10⁻⁴). Lasso keeps pH, but only with a tiny straight-line coefficient (+0.010 t/ha per sd) whose 95 % bootstrap interval crosses zero. The effect is real but modest: about 0.27 t/ha across the whole pH range.

**Q: Your coefficient chart greys out many terms. What does that mean?**
Each linear model was refitted on 500 bootstrap resamples of the training fields; the bar is the middle 95 % of its coefficients. If that interval crosses zero we cannot tell the sign, so the term is "not distinguishable from zero". For the multiple linear model that is **11 of 21** terms, mostly the crop, region and season levels, which matches the finding that those inputs barely matter on this benchmark.

**Q: Why does the two-factor heatmap stop at the edges of the data?**
Each cell is a model prediction. Outside the values seen in training (for example fertilizer above the largest training value), a polynomial model extrapolates its curve and a tree model just repeats its last split, so the numbers would look precise but mean nothing. The grid is 25 × 25 over each input's observed minimum to maximum, and a test checks it never goes beyond them.

## Conformal interval, in more detail

**Q: Your 90 % range covered 88.5 %, not 90 %. Is it wrong?**
No. The guarantee is "at least 90 % on average over many datasets", not on one finite test set. With 960 test fields the coverage estimate has a standard error of about 1 point, so 88.5 % (850 of 960) is within normal sampling variation. We report the measured value rather than claiming 90 %.

**Q: Is it equally good for every crop and region?**
Not exactly, and the app shows this (Predict page, Technical mode). On the test fields, coverage by crop runs from **85.9 %** (Rice) to **91.1 %** (Barley), and by region from **85.9 %** (West) to **92.0 %** (North). No group is below 85 %. One width (± 1.04 t/ha) is used for every field, so groups whose yields are harder to predict are covered less and easier ones more. A per-group width (Mondrian conformal) would even this out; it was left out to keep one simple, global method.

---

## India model upgrade (Phase 4)

**Q: How exactly is the crop × state look-up computed, and could it leak?**
For every test row it predicts the average of log(1 + yield) over all **training-year** rows (1997–2015) of the same crop in the same state, transformed back (so it is a geometric-style mean, matching the log-scale models). All seasons are pooled. A crop × state pair never seen in training falls back to the overall training mean (not the crop's mean). That happens for **147** test rows, including the **10** rows of the unseen crop. It reads no test-year yield, so it does not leak; the audit (notebook 04, section 12) recomputes it from the saved split. On the 5 time-ordered CV folds it scores **5.97 ± 2.89**, and on 2016–2020 **7.81** (unchanged).

**Q: What is target encoding, and why does the naive version leak?**
It replaces a category (crop, state, crop × state) with the average yield seen for it. The naive version averages over all training rows, including the row being encoded. The model then learns to trust an encoding that already contains the answer, and it does worse on new data. Here the encoding is **time-safe**: a row of year *t* is encoded only from rows of years before *t*. Training rows get an expanding window, CV validation rows only their fold's earlier training years, and test rows only 1997–2015. It is smoothed (crop × state toward the crop, crop and state toward the overall mean), and the smoothing strength is tuned by CV. A test checks that changing a year's targets never changes that year's own encodings.

**Q: Your lag features use last year's yield. Isn't that cheating on the test years?**
It is the forecasting assumption, stated in the code and docs: *when predicting year t, actual yields up to t − 1 are known* (a one-year-ahead forecast, as a ministry would make each season). Lags are computed only from strictly earlier years. The models are fitted once on 1997–2015 and never refitted on test years. Under this assumption a 2018 forecast may use the 2017 actual yield. Because that uses test-year yields as inputs (never as training data), every model is also scored in a **strict** variant where only 1997–2015 is known for all five test years; both are reported. A test scrambles all yields from year *t* onward and checks that the lags of year-*t* rows do not change.

**Q: What is the hybrid?**
Take the look-up's prediction and train a tree model (the best family on all features, here random forest) on what it gets wrong (actual − look-up, on the log scale). The prediction is look-up + predicted residual. The look-up inside it is also time-safe: a training row's offset comes from earlier years only, so the residuals the model learns from look like the residuals it will see on new years. It scored CV RMSE **6.06**, slightly worse than the look-up alone (5.97): once the look-up has removed each crop × state level, little is left for the forest to learn from the other inputs.

**Q: How did you choose the final India model without peeking at the test years?**
Same rule as Kaggle: lowest time-ordered CV RMSE, on the same 5 year-block folds for every model, with every model's search re-tuned inside each fold (nested CV, 20 configurations, seeded). The look-up is scored on the same folds. The rule was fixed in code before the test score was computed: if no model beats the look-up's CV RMSE, Random Forest stays deployed and the look-up is reported as the best predictor. Every candidate is then evaluated once on 2016–2020; nothing is re-selected afterwards.

**Q: What did the ablation show?**
Random forest, same folds and search budget, CV RMSE: base inputs 5.89, + target encoding 6.05, **+ lags 5.14**, + year trend 5.81, all features 5.12. Lags are the only group that matters on their own: they cut CV RMSE by 0.74. Target encoding alone does not help in CV, most likely because the random forest already learns crop × state levels from the one-hot columns. The trend helps a little.

**Q: So did machine learning beat the look-up?**
Yes, under the one-year-ahead assumption. The selected random forest (all features) scores CV 5.12 vs 5.97 and test 6.85 vs 7.81 (R² 0.75 vs 0.68). It beats the look-up for 48 of 54 crops and 26 of 30 states. But be honest about where the gain comes from: repeating last year's yield, with no model at all, scores 6.95, so the forest adds only 0.10 on top. In the strict variant (only 1997–2015 known), it scores 9.65, **worse** than the look-up. On the 51 test rows with no earlier record of their crop × state, it is slightly worse than the look-up (1.07 vs 0.97). The real finding is that yields persist from year to year, and the weather and input columns add almost nothing.

**Q: Why is the look-up so hard to beat without history?**
Crops differ in yield by two orders of magnitude, and each crop × state has a stable level. The other inputs carry little signal: rainfall is one annual number per state, and fertilizer and pesticide are one national number per year (every test year's pesticide is above the training maximum, so trees cannot use it). Without lags, a model can at best rediscover the look-up table, and it then pays for its extra variance.

**Q: What does the selected model rely on?**
Mean |SHAP| on 1,000 test rows: the 3-year rolling mean yield 71 %, crop 10 %, last year's yield 9 %, state 7 %, and rainfall, fertilizer and pesticide under 0.5 % each. Permutation importance (30 repeats, test years) gives the same top input: shuffling the rolling mean raises RMSE by 7.07. The two rankings agree with Spearman ρ 0.78. The year and pesticide have exactly zero permutation importance, because every test year lies beyond the training range.

**Q: Did LightGBM or HistGradientBoosting change the Kaggle result?**
No. They were added as comparison rows only, with the same folds and search budget: HistGradientBoosting CV 0.639 ± 0.009 and LightGBM 0.647 ± 0.008, against the deployed polynomial's 0.635 ± 0.009. Neither beats it, so the deployed model, its metrics and its file are unchanged.
