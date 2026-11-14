"""Display names: every column the UI can show has a readable name (no snake_case leaks)."""

import re

import pandas as pd
import pytest

from src import config
from ui import labels
from ui.components import table_view

SNAKE = re.compile(r"[a-z]+_[a-z]+")
ALL_COLUMNS = (
    config.FEATURES
    + [config.TARGET]
    + config.INDIA_FEATURES
    + [config.INDIA_TARGET, config.INDIA_YEAR]
)
TABLES = [
    "test_results",
    "cv_all_models",
    "india_test",
    "india_cv",
    "india_ablation",
    "india_reference_baselines",
    "india_per_crop_rmse",
]


@pytest.mark.parametrize("col", ALL_COLUMNS)
@pytest.mark.parametrize("simple", [False, True])
def test_every_feature_has_a_readable_label(col, simple):
    text = labels.label(col, simple)
    assert text and not SNAKE.search(text) and "_" not in text, text


def test_units_follow_the_name():
    assert labels.label("total_rainfall") == "Rainfall (mm)"
    assert labels.label("fertilizer_amount") == "Fertilizer (kg/ha)"
    assert labels.label("soil_ph") == "Soil pH"  # the unit 'pH' is implied, not repeated


def test_every_model_and_group_has_both_names():
    for key in labels.MODELS:
        assert "_" not in labels.model(key) and "_" not in labels.model(key, simple=True)
    for key in config.FEATURE_GROUPS:
        assert labels.group(key) and labels.group(key, simple=True)


@pytest.mark.parametrize("table", TABLES)
def test_results_table_headers_and_rows_are_readable(table):
    frame = pd.read_csv(config.TABLES_DIR / f"{table}.csv")
    shown = table_view(frame).data
    assert all("_" not in h for h in shown.columns), list(shown.columns)
    if "Model" in shown:
        assert all("_" not in m for m in shown["Model"]), list(shown["Model"])
