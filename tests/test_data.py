import hashlib

import pandas as pd
import pytest

from src import config
from src.data import load_kaggle_train, load_split, make_split

TRACKED_SPLIT = [config.KAGGLE_TRAIN_PARQUET, config.KAGGLE_TEST_PARQUET]  # committed copies
# The raw Kaggle CSV is git-ignored, so a fresh clone skips these (like the India tests).
needs_kaggle_csv = pytest.mark.skipif(
    not config.KAGGLE_TRAIN_CSV.exists(), reason="Kaggle CSV not downloaded"
)


@pytest.fixture(autouse=True)
def isolated_split(tmp_path, monkeypatch):
    """make_split() writes the split to parquet; send it to a temp dir, never to data/processed/."""
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(config, "KAGGLE_TRAIN_PARQUET", tmp_path / "kaggle_train.parquet")
    monkeypatch.setattr(config, "KAGGLE_TEST_PARQUET", tmp_path / "kaggle_test.parquet")
    return tmp_path


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@needs_kaggle_csv
def test_make_split_does_not_touch_tracked_files(isolated_split):
    before = [sha256(p) for p in TRACKED_SPLIT]
    make_split()
    assert [sha256(p) for p in TRACKED_SPLIT] == before
    assert (isolated_split / "kaggle_train.parquet").exists()
    assert (isolated_split / "kaggle_test.parquet").exists()


@needs_kaggle_csv
def test_split_sizes_sum_to_deduplicated_original():
    train, test = make_split()
    assert len(train) + len(test) == len(load_kaggle_train())


@needs_kaggle_csv
def test_no_row_in_both_splits():
    train, test = make_split()
    assert not set(train.index) & set(test.index)
    # Also compare full row contents, not just the index.
    assert len(train.merge(test, how="inner")) == 0


def test_target_not_in_features():
    assert config.TARGET not in config.NUMERIC_FEATURES
    assert config.TARGET not in config.CATEGORICAL_FEATURES


@needs_kaggle_csv
def test_config_columns_exist_in_data():
    df = load_kaggle_train()
    for col in config.FEATURES + [config.TARGET]:
        assert col in df.columns, col
    for col in config.DROP_COLUMNS:
        assert col not in df.columns, col


@needs_kaggle_csv
def test_make_split_is_deterministic():
    train1, test1 = make_split()
    train2, test2 = make_split()
    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(test1, test2)


@needs_kaggle_csv
def test_parquet_round_trip():
    train, test = make_split()
    train_back, test_back = load_split()
    pd.testing.assert_frame_equal(train, train_back, check_dtype=False)
    pd.testing.assert_frame_equal(test, test_back, check_dtype=False)
