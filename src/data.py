"""Loading the Kaggle data and creating the locked-away train/test split."""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from src import config

logger = logging.getLogger(__name__)


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV, failing with a clear message if it has not been downloaded."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the data first (see Phase 1 in the build report)."
        )
    return pd.read_csv(path)


def load_kaggle_train() -> pd.DataFrame:
    """Load the labelled Kaggle training file, drop DROP_COLUMNS and exact duplicates."""
    df = _read_csv(config.KAGGLE_TRAIN_CSV).drop(columns=config.DROP_COLUMNS)
    # Deduplicate AFTER dropping id, otherwise the unique id hides duplicate rows.
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Removed %d exact duplicate rows", n_before - len(df))
    return df


def load_kaggle_competition_test() -> pd.DataFrame:
    """Load the unlabelled competition test file (used only for the submission file)."""
    return _read_csv(config.KAGGLE_COMPETITION_TEST_CSV)


def make_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle-split the training data 80/20 and save both parts to parquet."""
    df = load_kaggle_train()
    train_df, test_df = train_test_split(
        df, test_size=config.TEST_SIZE, shuffle=True, random_state=config.RANDOM_STATE
    )
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(config.KAGGLE_TRAIN_PARQUET)
    test_df.to_parquet(config.KAGGLE_TEST_PARQUET)
    return train_df, test_df


def load_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the saved train/test split back from parquet."""
    for path in (config.KAGGLE_TRAIN_PARQUET, config.KAGGLE_TEST_PARQUET):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run src.data.make_split() first.")
    return pd.read_parquet(config.KAGGLE_TRAIN_PARQUET), pd.read_parquet(config.KAGGLE_TEST_PARQUET)


def make_kfold() -> KFold:
    """The single shuffled KFold used for every model, so all models see identical folds."""
    return KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix (config.FEATURES only) from the target."""
    return df[config.FEATURES], df[config.TARGET]
