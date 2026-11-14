"""Display names and units: the one mapping from data columns to what the UI shows.

Every chart axis, slider, table header, tooltip and report row goes through here, so no
snake_case column name reaches the screen (tests/test_labels.py checks every feature).
"""

from src import config

# column -> (technical name, plain name for Simple mode, unit)
FEATURES = {
    "soil_ph": ("Soil pH", "Soil pH", "pH"),
    "soil_moisture": ("Soil moisture", "Soil moisture", "%"),
    "nitrogen_content": ("Nitrogen content", "Soil nitrogen", "%"),
    "phosphorus_content": ("Phosphorus content", "Soil phosphorus", "%"),
    "potassium_content": ("Potassium content", "Soil potassium", "%"),
    "avg_temperature": ("Average temperature", "Temperature", "°C"),
    "total_rainfall": ("Rainfall", "Rainfall", "mm"),
    "sunlight_hours": ("Sunlight hours", "Sunshine", "hours"),
    "fertilizer_amount": ("Fertilizer", "Fertilizer", "kg/ha"),
    "pesticide_usage": ("Pesticide", "Pesticide use", "L/ha"),
    "irrigation_frequency": ("Irrigation frequency", "Irrigation", "times/week"),
    "crop_type": ("Crop type", "Crop", ""),
    "region": ("Region", "Region", ""),
    "season": ("Season", "Season", ""),
    config.TARGET: ("Yield", "Yield", "t/ha"),
    # India (state-level records)
    config.INDIA_RAINFALL: ("Annual rainfall", "Rainfall", "mm"),
    config.INDIA_FERTILIZER_PER_HA: ("Fertilizer per hectare", "Fertilizer", "kg/ha"),
    config.INDIA_PESTICIDE_PER_HA: ("Pesticide per hectare", "Pesticide", ""),
    config.INDIA_LOG_AREA: ("Crop area (log)", "Crop area", ""),
    config.INDIA_CROP: ("Crop", "Crop", ""),
    config.INDIA_SEASON: ("Season", "Season", ""),
    config.INDIA_STATE: ("State", "State", ""),
    config.INDIA_YEAR: ("Year", "Year", ""),
    config.INDIA_TARGET: ("Yield", "Yield", "yield units"),
}

GROUPS = {  # config.FEATURE_GROUPS key -> (technical, plain)
    "Soil": ("Soil", "Soil conditions"),
    "Weather": ("Weather", "Weather"),
    "Farming": ("Farming", "Farming practices"),
    "Context": ("Context", "Crop, region and season"),
}

MODELS = {  # model key -> (technical, plain)
    "dummy_mean": ("Mean baseline", "Guessing the average"),
    "simple_lr": ("Simple linear (fertilizer only)", "Fertilizer-only line"),
    "multiple_lr": ("Multiple linear", "Straight-line formula"),
    "polynomial": ("Polynomial + Ridge", "Curved formula"),
    "polynomial_deg3": ("Polynomial (degree 3) + Ridge", "Extra-curved formula"),
    "ridge": ("Ridge", "Straight-line formula (smoothed)"),
    "lasso": ("Lasso", "Straight-line formula (trimmed)"),
    "random_forest": ("Random Forest", "Forest of decision trees"),
    "gradient_boost": ("XGBoost", "Boosted decision trees"),
}

COLUMNS = {  # results-table column -> header
    "model": "Model",
    "predictor": "Predictor",
    "crop": "Crop",
    "feature": "Feature",
    "rmse": "RMSE",
    "mae": "MAE",
    "r2": "R²",
    "train_rmse": "Train RMSE",
    "test_rmse": "Test RMSE",
    "test_mae": "Test MAE",
    "test_r2": "Test R²",
    "rmse_mean": "CV RMSE",
    "rmse_std": "CV RMSE (sd)",
    "mae_mean": "CV MAE",
    "mae_std": "CV MAE (sd)",
    "r2_mean": "CV R²",
    "r2_std": "CV R² (sd)",
    "fit_time_s": "Fit time (s)",
    "alpha_median": "Chosen α (median)",
    "rmse_with": "RMSE with",
    "rmse_without": "RMSE without",
    "delta_rmse": "Δ RMSE",
    "r2_with": "R² with",
    "r2_without": "R² without",
    "delta_r2": "Δ R²",
    "n_test_rows": "Test rows",
    "mean_yield": "Mean yield",
    "relative_rmse": "RMSE / mean yield",
    "median_abs_error": "Median |error|",
    "train_min": "Train min",
    "train_max": "Train max",
    "test_min": "Test min",
    "test_max": "Test max",
    "share_test_rows_above_train_max": "Test rows above train max",
    "share_test_rows_below_train_min": "Test rows below train min",
    "mean_abs_shap": "Mean |SHAP|",
    "share": "Share",
    "coverage": "Coverage",
    "n": "Test fields",
    "group": "Group",
}


def name(col: str, simple: bool = False) -> str:
    """'nitrogen_content' -> 'Nitrogen content' (Technical) or 'Soil nitrogen' (Simple)."""
    technical, plain, _ = FEATURES[col]
    return plain if simple else technical


def unit(col: str) -> str:
    """'total_rainfall' -> 'mm'; '' when the column has no unit."""
    return FEATURES[col][2]


def label(col: str, simple: bool = False) -> str:
    """Name with unit: 'Rainfall (mm)', 'Soil pH' (the pH unit is implied), 'Crop type'."""
    u = unit(col)
    text = name(col, simple)
    return f"{text} ({u})" if u and u != "pH" else text


def group(key: str, simple: bool = False) -> str:
    technical, plain = GROUPS[key]
    return plain if simple else technical


def model(key: str, simple: bool = False) -> str:
    technical, plain = MODELS[key]
    return plain if simple else technical


def column(col: str) -> str:
    """Results-table header; falls back to the feature names, then to 'Sentence case'."""
    if col in COLUMNS:
        return COLUMNS[col]
    if col in FEATURES:
        return label(col)
    text = col.replace("_", " ")
    return text[:1].upper() + text[1:]


def level(value: str) -> str:
    """A category level as shown: 'Other  Rabi pulses' -> 'Other Rabi pulses'."""
    return " ".join(str(value).split())


def lower(text: str) -> str:
    """Lower-case the first letter only, so 'Soil pH' reads 'soil pH' inside a sentence."""
    return text[:1].lower() + text[1:]


def term(term_name: str, feature: str) -> str:
    """A model term as shown: 'crop_type_Rice' -> 'Crop type = Rice'; 'soil_ph' -> 'Soil pH'."""
    if term_name == feature:
        return name(feature)
    return f"{name(feature)} = {level(term_name[len(feature) + 1 :])}"
