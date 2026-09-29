"""Feature-signal statistics used by the EDA notebook and by model selection."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

from src import config

N_BINS = 10  # quantile bins for binned-means plots and the linearity test
MAX_DISCRETE_LEVELS = 10  # a numeric column with this few values is treated as discrete for MI


def _bins(x: pd.Series, n_bins: int = N_BINS) -> pd.Series:
    """Quantile bins for continuous x; one bin per value for discrete x (e.g. 1-6 per week)."""
    if x.nunique() <= MAX_DISCRETE_LEVELS:
        return x
    return pd.qcut(x, q=n_bins, duplicates="drop")


def binned_means(x: pd.Series, y: pd.Series, n_bins: int = N_BINS) -> pd.DataFrame:
    """Mean of y within quantile bins of x, with a normal-approximation 95% CI."""
    bins = _bins(x, n_bins)
    g = y.groupby(bins, observed=True)
    out = pd.DataFrame(
        {"x_mid": x.groupby(bins, observed=True).median(), "mean": g.mean(), "n": g.size()}
    )
    half_width = 1.96 * g.std() / np.sqrt(out["n"])
    out["ci_low"], out["ci_high"] = out["mean"] - half_width, out["mean"] + half_width
    return out.reset_index(drop=True)


def linearity_test(x: pd.Series, y: pd.Series, n_bins: int = N_BINS) -> tuple[float, float]:
    """Compare a straight line against bin means (correlation ratio eta^2 vs r^2).

    The linear model is nested inside the bin-means model, so an F-test on the
    extra variance explained by the bins tests for departure from linearity.
    Binning discards the slope within each bin, so for a strongly linear feature
    eta^2 can fall slightly below r^2 (then p = 1). Returns (eta_squared, p_value).
    """
    bins = _bins(x, n_bins)
    k = bins.nunique()
    n = len(y)
    ss_total = ((y - y.mean()) ** 2).sum()
    ss_bins_resid = ((y - y.groupby(bins, observed=True).transform("mean")) ** 2).sum()
    eta2 = 1 - ss_bins_resid / ss_total
    r2 = x.corr(y) ** 2
    df_extra, df_resid = k - 2, n - k
    if df_extra <= 0:
        return float(eta2), float("nan")
    f_stat = ((eta2 - r2) / df_extra) / ((1 - eta2) / df_resid)
    return float(eta2), float(stats.f.sf(f_stat, df_extra, df_resid))


def quadratic_term_pvalue(x: pd.Series, y: pd.Series) -> float:
    """p-value of the x^2 term in y ~ x + x^2: a 1-df test for an optimum or U-shape."""
    z = (x - x.mean()) / x.std()  # centring avoids collinearity between x and x^2
    design = sm.add_constant(np.column_stack([z, z**2]))
    return float(sm.OLS(y.to_numpy(), design).fit().pvalues[2])


def feature_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson r, Spearman rho, mutual information and linearity test for each numeric feature.

    mi_linear is the MI a bivariate normal with the same Pearson r would have,
    -0.5 * ln(1 - r^2); MI well above it suggests a non-linear relationship.
    mi_noise_floor is the MI against a shuffled target (no real relationship).
    """
    X = df[config.NUMERIC_FEATURES]
    y = df[config.TARGET]
    discrete = [X[c].nunique() <= MAX_DISCRETE_LEVELS for c in X.columns]
    mi = mutual_info_regression(X, y, discrete_features=discrete, random_state=config.RANDOM_STATE)
    y_shuffled = y.sample(frac=1, random_state=config.RANDOM_STATE).to_numpy()
    mi_null = mutual_info_regression(
        X, y_shuffled, discrete_features=discrete, random_state=config.RANDOM_STATE
    )
    rows = []
    for i, col in enumerate(X.columns):
        r = X[col].corr(y)
        eta2, p_nonlinear = linearity_test(X[col], y)
        rows.append(
            {
                "feature": col,
                "pearson_r": r,
                "spearman_rho": X[col].corr(y, method="spearman"),
                "mutual_info": mi[i],
                "mi_linear": -0.5 * np.log(1 - r**2),
                "mi_noise_floor": mi_null[i],
                "r_squared": r**2,
                "eta_squared": eta2,
                "p_nonlinear": p_nonlinear,
                "p_quadratic": quadratic_term_pvalue(X[col], y),
            }
        )
    out = pd.DataFrame(rows).set_index("feature")
    return out.sort_values("pearson_r", key=np.abs, ascending=False)


def top_pearson_feature(
    df: pd.DataFrame,
    numeric: list[str] = config.NUMERIC_FEATURES,
    target: pd.Series | None = None,
) -> str:
    """Numeric feature with the highest |Pearson r| with the target (pass TRAINING data only).

    target defaults to df[config.TARGET]; pass it explicitly to use another
    target or its modelling scale (e.g. log1p yield for the India data).
    """
    target = df[config.TARGET] if target is None else target
    return str(df[numeric].corrwith(target).abs().idxmax())
