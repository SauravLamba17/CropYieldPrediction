"""Single source of truth for settings, paths and column names."""

from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# --- Paths (all relative to the project root, resolved from this file) ---
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_KAGGLE_DIR = DATA_DIR / "raw" / "kaggle"
RAW_INDIA_DIR = DATA_DIR / "raw" / "india"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = ROOT / "models"
SUBMISSIONS_DIR = ROOT / "submissions"

KAGGLE_TRAIN_CSV = RAW_KAGGLE_DIR / "crop_yield_train.csv"
KAGGLE_COMPETITION_TEST_CSV = RAW_KAGGLE_DIR / "crop_yield_test.csv"
KAGGLE_SAMPLE_SUBMISSION_CSV = RAW_KAGGLE_DIR / "sample_submission.csv"
KAGGLE_TRAIN_PARQUET = PROCESSED_DIR / "kaggle_train.parquet"
KAGGLE_TEST_PARQUET = PROCESSED_DIR / "kaggle_test.parquet"
APP_METADATA_JSON = MODELS_DIR / "app_metadata.json"  # slider ranges for the Streamlit app

# --- Kaggle columns ---
TARGET = "yield_tpha"  # tons per hectare (confirmed on the competition data page)
ID_COLUMN = "id"
HARVEST_DATE_COLUMN = "harvest_date"
FIELD_ID_COLUMN = "field_id"

# id: row identifier. harvest_date, field_id: undocumented, and Phase 1 showed
# both are random (every field_id spans several crops/regions; harvest month is
# independent of season), so they carry no agronomic information.
DROP_COLUMNS = [ID_COLUMN, HARVEST_DATE_COLUMN, FIELD_ID_COLUMN]

SOIL_FEATURES = [
    "soil_ph",
    "soil_moisture",
    "nitrogen_content",
    "phosphorus_content",
    "potassium_content",
]
WEATHER_FEATURES = ["avg_temperature", "total_rainfall", "sunlight_hours"]
# irrigation_frequency is ordinal (1-6 times/week); treated as numeric because
# it has a real unit and equal spacing.
FARMING_FEATURES = ["fertilizer_amount", "pesticide_usage", "irrigation_frequency"]
CONTEXT_FEATURES = ["crop_type", "region", "season"]

NUMERIC_FEATURES = SOIL_FEATURES + WEATHER_FEATURES + FARMING_FEATURES
CATEGORICAL_FEATURES = CONTEXT_FEATURES
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FEATURE_GROUPS = {
    "Soil": SOIL_FEATURES,
    "Weather": WEATHER_FEATURES,
    "Farming": FARMING_FEATURES,
    "Context": CONTEXT_FEATURES,
}

# Units from the competition data page, used for axis labels.
UNITS = {
    TARGET: "t/ha",
    "soil_ph": "pH",
    "soil_moisture": "%",
    "avg_temperature": "°C",
    "total_rainfall": "mm",
    "fertilizer_amount": "kg/ha",
    "pesticide_usage": "L/ha",
    "sunlight_hours": "hours",
    "nitrogen_content": "%",
    "phosphorus_content": "%",
    "potassium_content": "%",
    "irrigation_frequency": "times/week",
}

# EDA interaction check: does the slope of the strongest feature differ by these?
INTERACTION_MODERATORS = ["crop_type", "irrigation_frequency"]
# GroupKFold check: can a model predict a region it never saw in training?
GROUP_CV_COLUMN = "region"

# --- India state-level data (Mendeley ncw2vbcgnk, CC BY 4.0): real-world validation ---
INDIA_CSV = RAW_INDIA_DIR / "crop_yield.csv"
INDIA_TRAIN_PARQUET = PROCESSED_DIR / "india_train.parquet"
INDIA_TEST_PARQUET = PROCESSED_DIR / "india_test.parquet"
INDIA_TARGET = "Yield"
# Crop_Year is used only to split by time and is never passed to a model directly.
# NOTE: Fertilizer_per_ha / Pesticide_per_ha take one value per year, so the models
# still see the year implicitly through them (legitimate: the year is known when predicting).
INDIA_YEAR = "Crop_Year"
INDIA_PRODUCTION = "Production"  # Yield = Production / Area, so it must never be a feature
INDIA_AREA = "Area"
INDIA_FERTILIZER = "Fertilizer"  # raw column is a total (kg), scales with area
INDIA_PESTICIDE = "Pesticide"  # raw column is a total, scales with area
INDIA_RAINFALL = "Annual_Rainfall"  # one value per state-year (mm)
INDIA_CROP = "Crop"
INDIA_SEASON = "Season"
INDIA_STATE = "State"
INDIA_LOG_AREA = "log_Area"
INDIA_FERTILIZER_PER_HA = "Fertilizer_per_ha"
INDIA_PESTICIDE_PER_HA = "Pesticide_per_ha"
INDIA_TRAIN_LAST_YEAR = 2015  # train on <= 2015 ...
INDIA_TEST_FIRST_YEAR = 2016  # ... test on >= 2016 (never let the model see the future)
# Coconut yield is reported in nuts per hectare (median ~8,500 vs <= 53 for every
# other crop), so it is not comparable with t/ha and would dominate every RMSE.
INDIA_EXCLUDED_CROPS = ["Coconut"]
INDIA_NUMERIC_FEATURES = [
    INDIA_RAINFALL,
    INDIA_FERTILIZER_PER_HA,
    INDIA_PESTICIDE_PER_HA,
    INDIA_LOG_AREA,
]
INDIA_CATEGORICAL_FEATURES = [INDIA_CROP, INDIA_SEASON, INDIA_STATE]
INDIA_FEATURES = INDIA_NUMERIC_FEATURES + INDIA_CATEGORICAL_FEATURES
# Ablation (CODE_REVIEW H1): these two columns hold one national value per year, i.e.
# they encode the year. The India suite is re-run without them to measure their effect.
INDIA_YEAR_INDEX_FEATURES = [INDIA_FERTILIZER_PER_HA, INDIA_PESTICIDE_PER_HA]
INDIA_FEATURE_GROUPS = {
    "Weather": [INDIA_RAINFALL],
    "Farming": [INDIA_FERTILIZER_PER_HA, INDIA_PESTICIDE_PER_HA, INDIA_LOG_AREA],
    "Context": INDIA_CATEGORICAL_FEATURES,
}
INDIA_TOP_CROPS_FOR_REPORT = 10
