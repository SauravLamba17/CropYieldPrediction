"""Pure calculations behind the What-if and explainability pages (no Streamlit, unit-tested).

Every function takes a fitted model or saved tables and returns plain DataFrames/numbers.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import RegressorMixin

from src import config

HEATMAP_STEPS = 25
ICE_LINES = 50
OPTIMUM_SHARE = 0.10  # the "optimum region": within 10% of the curve's span below its peak


def observed_grid(low: float, high: float, steps: int, is_integer: bool = False) -> np.ndarray:
    """`steps` evenly spaced values from the observed min to max, never beyond them."""
    if is_integer:
        values = np.unique(np.round(np.linspace(low, high, steps)).astype(int))
        return np.clip(values, int(np.ceil(low)), int(np.floor(high)))
    return np.clip(np.linspace(low, high, steps), low, high)


def heatmap_grid(
    model: RegressorMixin,
    inputs: dict,
    fx: str,
    fy: str,
    bounds: dict[str, dict],
    steps: int = HEATMAP_STEPS,
) -> pd.DataFrame:
    """Predicted yield over a steps × steps grid of two inputs; every other input as in `inputs`.

    bounds: {feature: {"min", "max", "is_integer"}} from the training data (app_metadata).
    """
    gx = observed_grid(bounds[fx]["min"], bounds[fx]["max"], steps, bounds[fx]["is_integer"])
    gy = observed_grid(bounds[fy]["min"], bounds[fy]["max"], steps, bounds[fy]["is_integer"])
    xx, yy = np.meshgrid(gx, gy)
    rows = pd.DataFrame([inputs] * xx.size)
    rows[fx], rows[fy] = xx.ravel(), yy.ravel()
    return pd.DataFrame({"x": xx.ravel(), "y": yy.ravel(), "pred": model.predict(rows[config.FEATURES])})


def ice_frame(model: RegressorMixin, rows: pd.DataFrame, feature: str, grid: np.ndarray) -> pd.DataFrame:
    """Individual conditional expectation: one prediction curve per row as `feature` sweeps."""
    repeated = rows.loc[rows.index.repeat(len(grid))].copy()
    repeated[feature] = np.tile(grid, len(rows))
    pred = model.predict(repeated[config.FEATURES])
    return pd.DataFrame(
        {"line": np.repeat(np.arange(len(rows)), len(grid)), "x": np.tile(grid, len(rows)), "y": pred}
    )


def shap_difference(a: dict[str, float], b: dict[str, float]) -> pd.Series:
    """SHAP(B) − SHAP(A) per original feature, largest absolute gap first.

    With the same explainer (same average prediction) the gaps sum to pred(B) − pred(A).
    """
    diff = pd.Series(b) - pd.Series(a)
    return diff.reindex(diff.abs().sort_values(ascending=False).index)


def method_scores(
    permutation: pd.DataFrame, shap: pd.DataFrame, coefficients: pd.DataFrame, model: str, linear: str
) -> pd.DataFrame:
    """One importance score per original feature and method (permutation, mean |SHAP|, |coef|).

    |coef| is the largest absolute standardised coefficient among a feature's terms in the
    linear model `linear` (for a category: its strongest level).
    """
    perm = permutation.query("model == @model").set_index("feature")["importance_mean"]
    sh = shap.query("model == @model").set_index("feature")["mean_abs_shap"]
    coef = coefficients.query("model == @linear").assign(a=lambda d: d["coef"].abs()).groupby("feature")["a"].max()
    return pd.DataFrame({"permutation": perm, "shap": sh, "coefficient": coef}).reindex(config.FEATURES)


def method_ranks(scores: pd.DataFrame) -> pd.DataFrame:
    """Rank per method (1 = most important)."""
    return scores.rank(ascending=False, method="min").astype(int)


def method_agreement(scores: pd.DataFrame) -> pd.DataFrame:
    """Spearman rank correlation for every pair of methods."""
    cols = list(scores.columns)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            rows.append({"a": a, "b": b, "rho": float(spearmanr(scores[a], scores[b]).statistic)})
    return pd.DataFrame(rows)


def correlations(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """(Pearson r, Spearman ρ)."""
    return float(pearsonr(x, y).statistic), float(spearmanr(x, y).statistic)


def optimum_region(curve: pd.DataFrame, share: float = OPTIMUM_SHARE) -> dict:
    """Peak of a response curve and the x-range within `share` of its span below the peak.

    Returns peak x/y, the region [low, high], and whether the peak is interior (the curve
    falls on BOTH sides by more than `share` of its span: a genuine optimum, not an edge).
    """
    x, y = curve["x"].to_numpy(), curve["y"].to_numpy()
    span = float(y.max() - y.min())
    i = int(y.argmax())
    near = x[y >= y.max() - share * span]
    interior = 0 < i < len(y) - 1 and (y[i] - y[0]) > share * span and (y[i] - y[-1]) > share * span
    return {
        "peak_x": float(x[i]),
        "peak_y": float(y[i]),
        "low": float(near.min()),
        "high": float(near.max()),
        "span": span,
        "interior": bool(interior),
    }


def group_coverage_flags(coverage: pd.DataFrame, floor: float = 0.85) -> pd.DataFrame:
    """Add an 'under' flag to the saved per-group coverage table."""
    return coverage.assign(under=coverage["coverage"] < floor)
