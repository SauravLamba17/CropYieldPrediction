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
