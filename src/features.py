"""Preprocessing: scaling, one-hot encoding and optional polynomial expansion.

Everything is returned as an unfitted ColumnTransformer so it is fitted inside
each model's Pipeline, i.e. only ever on the training fold (no leakage).
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


def build_preprocessor(
    numeric: list[str],
    categorical: list[str],
    poly_degree: int | None = None,
    drop_first: bool = False,
) -> ColumnTransformer:
    """StandardScaler on numeric columns, one-hot on categorical columns.

    poly_degree: if set, expand the NUMERIC columns only (categoricals are
        one-hot, never expanded), then re-scale because x^2 terms have a
        different variance and the Ridge penalty is scale-sensitive.
    drop_first: drop one level per categorical. Needed for plain OLS, where a
        full one-hot set plus an intercept is perfectly collinear (dummy trap).
    No imputer: Phase 1 found zero missing values.
    """
    if poly_degree is None:
        numeric_steps = make_pipeline(StandardScaler())
    else:
        numeric_steps = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=poly_degree, include_bias=False),
            StandardScaler(),
        )
    encoder = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False, drop="first" if drop_first else None
    )
    transformers = [("num", numeric_steps, numeric)]
    if categorical:
        transformers.append(("cat", encoder, categorical))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def transformed_frame(preprocessor: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
    """Apply a FITTED preprocessor and return a DataFrame with readable column names."""
    return pd.DataFrame(
        preprocessor.transform(X), columns=preprocessor.get_feature_names_out(), index=X.index
    )
