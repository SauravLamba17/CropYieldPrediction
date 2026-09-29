import pandas as pd

from src import config
from src.data import load_kaggle_train, load_split, make_split


def test_split_sizes_sum_to_deduplicated_original():
    train, test = make_split()
    assert len(train) + len(test) == len(load_kaggle_train())


def test_no_row_in_both_splits():
    train, test = make_split()
    assert not set(train.index) & set(test.index)
    # Also compare full row contents, not just the index.
    assert len(train.merge(test, how="inner")) == 0


def test_target_not_in_features():
    assert config.TARGET not in config.NUMERIC_FEATURES
    assert config.TARGET not in config.CATEGORICAL_FEATURES


def test_config_columns_exist_in_data():
    df = load_kaggle_train()
    for col in config.FEATURES + [config.TARGET]:
        assert col in df.columns, col
    for col in config.DROP_COLUMNS:
        assert col not in df.columns, col


def test_make_split_is_deterministic():
    train1, test1 = make_split()
    train2, test2 = make_split()
    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(test1, test2)


def test_parquet_round_trip():
    train, test = make_split()
    train_back, test_back = load_split()
    pd.testing.assert_frame_equal(train, train_back, check_dtype=False)
    pd.testing.assert_frame_equal(test, test_back, check_dtype=False)
