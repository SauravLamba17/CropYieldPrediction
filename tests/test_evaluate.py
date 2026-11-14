import json

import pandas as pd

from src import config
from src.evaluate import leaderboard


def test_leaderboard_ranks_by_cv_not_by_test():
    cv = pd.DataFrame(
        {"model": ["a", "b", "c"], "rmse_mean": [0.70, 0.60, 0.65], "rmse_std": [0.01] * 3}
    )
    # "a" has the best TEST error but the worst CV error: it must still rank last.
    test = pd.DataFrame({"model": ["a", "b", "c"], "rmse": [0.50, 0.66, 0.64], "r2": [0.9] * 3})
    board = leaderboard(cv, test)
    assert list(board["model"]) == ["b", "c", "a"]
    assert board.set_index("model").loc["a", "rmse"] == 0.50  # test columns carried unchanged


def test_saved_tables_agree_with_the_deployed_model():
    cv = pd.read_csv(config.TABLES_DIR / "cv_all_models.csv")
    test = pd.read_csv(config.TABLES_DIR / "test_results.csv")
    names = json.loads((config.MODELS_DIR / "model_names.json").read_text())
    board = leaderboard(cv, test)
    assert board.loc[board["model"] != "dummy_mean", "model"].iloc[0] == names["best_overall"]
    assert list(test["model"]) == list(board["model"])  # the CSV itself is in CV order
    assert {"train_rmse", "rmse", "mae", "r2"} <= set(test.columns)
