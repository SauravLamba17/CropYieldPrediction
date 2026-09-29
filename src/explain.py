"""Explainability helpers (RQ2): coefficients, permutation importance, SHAP, agreement.

Two SHAP routes are used:
  * TreeExplainer on the best TREE model, in the one-hot (model) feature space;
    one-hot SHAP values are summed back to their original column (SHAP is additive).
  * A model-agnostic Kernel SHAP explainer in the ORIGINAL feature space, used for
    single predictions of any model (e.g. the polynomial best_overall in the app).
"""

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.base import RegressorMixin
from sklearn.inspection import permutation_importance

from src import config
from src.models import final_estimator, fitted_pipeline

PERMUTATION_REPEATS = 10
SHAP_SAMPLE_SIZE = 2000  # TreeExplainer rows; the full set is used if it is smaller
SHAP_BACKGROUND_SIZE = 50  # background rows for Kernel SHAP (cost grows linearly with it)


def original_feature(term: str, features: list[str]) -> str:
    """Map a model input name (e.g. 'crop_type_Rice') back to its original column."""
    if term in features:
        return term
    matches = [f for f in features if term.startswith(f + "_")]
    if not matches:
        raise ValueError(f"Cannot map model input {term!r} to an original feature")
    return max(matches, key=len)  # longest prefix wins (safe if names share a prefix)


def readable_term(term: str, features: list[str], baselines: dict[str, str]) -> str:
    """'crop_type_Rice' -> 'crop_type = Rice (vs Barley)'; numeric names are unchanged."""
    feature = original_feature(term, features)
    if feature == term:
        return term
    level = term[len(feature) + 1 :]
    baseline = baselines.get(feature)
    return f"{feature} = {level}" + (f" (vs {baseline})" if baseline else "")


def standardized_coefficients(model: RegressorMixin, features: list[str]) -> pd.DataFrame:
    """Coefficients of a fitted linear pipeline with readable names.

    Numeric inputs are standardised, so their coefficient is the change in yield
    (t/ha) per one standard deviation. A one-hot coefficient is the difference
    from the dropped baseline level (or from the average level if none dropped).
    """
    pipe = fitted_pipeline(model)
    preprocessor = pipe[0]
    names = preprocessor.get_feature_names_out()
    baselines = {}
    if "cat" in preprocessor.named_transformers_:
        encoder = preprocessor.named_transformers_["cat"]
        if encoder.drop_idx_ is not None:
            for column, levels, idx in zip(
                encoder.feature_names_in_, encoder.categories_, encoder.drop_idx_
            ):
                baselines[column] = levels[idx]
    coef = final_estimator(model).coef_
    table = pd.DataFrame(
        {
            "term": names,
            "feature": [original_feature(t, features) for t in names],
            "label": [readable_term(t, features, baselines) for t in names],
            "coef": coef,
        }
    )
    table["abs_coef"] = table["coef"].abs()
    return table.sort_values("abs_coef", ascending=False).reset_index(drop=True)


def permutation_table(model: RegressorMixin, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Increase in RMSE (t/ha) when each ORIGINAL column is shuffled, mean and std over repeats."""
    result = permutation_importance(
        model,
        X,
        y,
        scoring="neg_root_mean_squared_error",
        n_repeats=PERMUTATION_REPEATS,
        random_state=config.RANDOM_STATE,
        # Serial on purpose: a few seconds here, and parallel workers trigger a
        # harmless but noisy joblib clean-up error when a Jupyter kernel exits on Windows.
        n_jobs=None,
    )
    table = pd.DataFrame(
        {
            "feature": X.columns,
            "rmse_increase": result.importances_mean,
            "rmse_increase_std": result.importances_std,
        }
    )
    return table.sort_values("rmse_increase", ascending=False).reset_index(drop=True)


def tree_shap(
    model: RegressorMixin,
    X: pd.DataFrame,
    features: list[str],
    sample_size: int = SHAP_SAMPLE_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """TreeExplainer SHAP values for a fitted tree pipeline.

    Returns (shap values in model-input space, the matching model inputs,
    shap values summed to original features). Values are in the model's target
    scale (t/ha here; log1p units if the target was log-transformed).
    """
    if len(X) > sample_size:
        X = X.sample(sample_size, random_state=config.RANDOM_STATE)
    pipe = fitted_pipeline(model)
    inputs = pd.DataFrame(
        pipe[0].transform(X), columns=pipe[0].get_feature_names_out(), index=X.index
    )
    values = shap.TreeExplainer(pipe[-1]).shap_values(inputs)
    shap_inputs = pd.DataFrame(values, columns=inputs.columns, index=X.index)
    by_feature = shap_inputs.T.groupby(lambda term: original_feature(term, features)).sum().T
    return shap_inputs, inputs, by_feature[features]


def mean_abs_shap(shap_by_feature: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per original feature, largest first."""
    return shap_by_feature.abs().mean().sort_values(ascending=False)


def grouped_importance(importance: pd.Series, groups: dict[str, list[str]]) -> pd.Series:
    """Sum an importance score over each feature group (e.g. Soil / Weather / Farming / Context)."""
    return pd.Series({group: importance[cols].sum() for group, cols in groups.items()}).sort_values(
        ascending=False
    )


def agreement_table(scores: dict[str, pd.Series], lasso_selected: pd.Series) -> pd.DataFrame:
    """Rank every feature under each importance method (1 = most important)."""
    table = pd.DataFrame(scores)
    ranks = table.rank(ascending=False, method="min").add_suffix("_rank").astype(int)
    out = pd.concat([table, ranks], axis=1)
    out["lasso_selected"] = lasso_selected.reindex(out.index)
    return out.sort_values(ranks.columns[0])


def rank_correlations(scores: dict[str, pd.Series]) -> pd.DataFrame:
    """Spearman rank correlation between every pair of importance methods."""
    table = pd.DataFrame(scores)
    rho = spearmanr(table.to_numpy()).statistic
    return pd.DataFrame(rho, index=table.columns, columns=table.columns)


class RawFeatureExplainer:
    """Kernel SHAP for ANY fitted model, in the original feature space.

    Kernel SHAP is used (not the permutation explainer) because it needs no numba
    compilation, which cost ~20 s on the first call and broke the app's start-up
    budget. It works on numeric arrays, so categoricals are integer-coded for SHAP
    and decoded back to their text labels before calling model.predict.
    """

    def __init__(
        self,
        model: RegressorMixin,
        background: pd.DataFrame,
        numeric: list[str] = config.NUMERIC_FEATURES,
        categorical: list[str] = config.CATEGORICAL_FEATURES,
    ) -> None:
        self.model = model
        self.columns = numeric + categorical
        self.categorical = categorical
        self.levels = {c: sorted(background[c].unique()) for c in categorical}
        sample = background.sample(
            min(SHAP_BACKGROUND_SIZE, len(background)), random_state=config.RANDOM_STATE
        )
        self.explainer = shap.KernelExplainer(self._predict_codes, self.encode(sample))

    def encode(self, X: pd.DataFrame) -> np.ndarray:
        """Text categories -> integer codes (numeric columns unchanged)."""
        coded = X[self.columns].copy()
        for c in self.categorical:
            coded[c] = coded[c].map({level: i for i, level in enumerate(self.levels[c])})
        return coded.to_numpy(dtype=float)

    def decode(self, codes: np.ndarray) -> pd.DataFrame:
        """Integer codes -> a DataFrame with the original text categories."""
        frame = pd.DataFrame(codes, columns=self.columns)
        for c in self.categorical:
            frame[c] = [self.levels[c][round(v)] for v in frame[c]]
        return frame

    def _predict_codes(self, codes: np.ndarray) -> np.ndarray:
        return self.model.predict(self.decode(codes))

    def explain(self, X: pd.DataFrame) -> shap.Explanation:
        """SHAP explanation (t/ha contributions) for the rows of X, with readable feature values.

        Side effect: reseeds NumPy's global random generator, because Kernel SHAP
        draws from it and offers no seed argument.
        """
        # Kernel SHAP samples feature coalitions with NumPy's global generator;
        # seeding it here makes every explanation reproducible.
        np.random.seed(config.RANDOM_STATE)
        values = self.explainer.shap_values(self.encode(X), silent=True)
        return shap.Explanation(
            values=values,
            base_values=np.full(len(X), self.explainer.expected_value),
            data=X[self.columns].to_numpy(dtype=object),
            feature_names=self.columns,
        )
