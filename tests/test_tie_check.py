"""Paired statistical tie check: the maths on toy inputs and the saved table."""

import pytest


def test_paired_test_on_a_toy_input():
    import math

    from src.tie_check import paired_test

    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [0.9, 1.8, 2.7, 3.6, 4.5]  # differences 0.1 ... 0.5
    r = paired_test(a, b)
    assert r["n_folds"] == 5
    assert r["mean_diff"] == pytest.approx(0.3)
    assert r["sd_diff"] == pytest.approx(math.sqrt(0.025))
    t = 0.3 / (math.sqrt(0.025) / math.sqrt(5))  # 4.2426
    x = t / math.sqrt(t * t + 4)
    assert r["t_test_p"] == pytest.approx(1 - x * (3 - x * x) / 2)  # closed form, 4 df: 0.0132
    assert r["wilcoxon_p"] == pytest.approx(2 / 2**5)  # all 5 signs agree: the exact minimum
    assert r["distinguishable"] is True


def test_paired_test_identical_and_noisy_scores_are_not_distinguishable():
    from src.tie_check import paired_test

    same = paired_test([1, 2, 3], [1, 2, 3])
    assert same["mean_diff"] == 0 and same["t_test_p"] == 1.0 and not same["distinguishable"]
    noisy = paired_test([1.0, 2.0, 3.0, 4.0, 5.0], [1.2, 1.8, 3.3, 3.6, 5.1])
    assert not noisy["distinguishable"]


def test_tie_table_is_well_formed():
    from src import config
    from src.tie_check import INDIA_PAIRS, KAGGLE_PAIRS

    path = config.TABLES_DIR / "tie_check.csv"
    if not path.exists():
        pytest.skip("tie_check.csv not built")
    import pandas as pd

    table = pd.read_csv(path)
    assert len(table) == len(KAGGLE_PAIRS) + len(INDIA_PAIRS)
    assert table["n_folds"].eq(5).all()
    assert table["distinguishable"].eq(table["t_test_p"] < 0.05).all()
    assert (table["wilcoxon_p"] >= 0.0625 - 1e-12).all()  # the exact floor with 5 folds


def test_verdict_wording_follows_the_tie_check():
    from ui import plain

    verdict = {"model": "Random Forest (all features)", "model_cv": 5.12, "lookup_cv": 5.97,
               "model_test": 6.85, "lookup_test": 7.81, "persistence_test": 6.95, "strict_test": 9.65,
               "group": "lags", "gain": 0.74, "beats_cv": True}
    tied = {**verdict, "tie_lookup": (0.0402, True), "tie_persistence": (0.6762, False)}
    technical = plain.india_verdict(tied, simple=False)[1]
    assert "is distinguishable from the look-up (p = 0.040)" in technical
    assert "<b>not distinguishable</b> from last year's yield (p = 0.676)" in technical
    assert "within fold-to-fold noise" in technical
    assert "statistically a tie" in plain.india_verdict(tied, simple=True)[1]
    # without the table, or when the model is distinguishable, no tie wording
    assert "Paired t-test" not in plain.india_verdict(verdict, simple=False)[1]
    clear = {**tied, "tie_persistence": (0.01, True)}
    assert "statistically a tie" not in plain.india_verdict(clear, simple=True)[1]
    assert "fold-to-fold" not in plain.india_verdict(clear, simple=False)[1]


def test_kaggle_tie_note():
    from ui import plain

    note = plain.kaggle_tie_note({"XGBoost": 0.7117, "HistGB": 0.2541}, {"XGBoost": False, "HistGB": False})
    assert note.startswith("On the 5 CV folds the selected model is not statistically distinguishable")
    assert "XGBoost (p = 0.71)" in note and "a tie" in note
    assert "not statistically" not in plain.kaggle_tie_note({"X": 0.01}, {"X": True})
    assert plain.kaggle_tie_note({}, {}) == ""
