"""The dashboard's plain-language sentences contain the right, sensibly rounded numbers."""

import numpy as np

from ui import plain


def test_model_headline_rounds_the_real_test_numbers():
    # Saved best model (polynomial) on the 960 held-out fields; mean test yield 6.28 t/ha.
    text = plain.model_headline(rmse=0.6462, mean_yield=6.2819, r2=0.6898, baseline_rmse=1.1604)
    assert "about <b>0.65 t/ha</b>" in text
    assert "<b>10%</b> of an average harvest" in text
    assert "about <b>69%</b> of the differences" in text
    assert "<b>44%</b> better than always guessing" in text


def test_predict_headline_rating_and_share():
    assert plain.rating(6.52, 5.45, 6.29, 7.09) == "Good"
    assert plain.rating(5.0, 5.45, 6.29, 7.09) == "Low"
    assert plain.rating(8.0, 5.45, 6.29, 7.09) == "Excellent"
    text = plain.predict_headline(6.5199, "Good", 0.564, 3840)
    assert "<b>6.52 t/ha</b>" in text and "a <b>good</b> yield" in text
    assert "<b>56%</b> of the 3,840 fields" in text
    # never rounds up to "higher than 100%"
    assert "<b>99%</b>" in plain.predict_headline(9.15, "Excellent", 0.996, 3840)
    assert "an <b>excellent</b>" in plain.predict_headline(9.15, "Excellent", 0.996, 3840)


def test_explanation_sentence_names_top_boost_and_drag():
    text = plain.explanation_sentence(
        {"Fertilizer": 0.1027, "Pesticide use": -0.0180, "Crop": 0.02}
    )
    assert "<b>Fertilizer</b> is adding about <b>+0.10 t/ha</b>" in text
    assert "<b>Pesticide use</b> is taking away about <b>−0.02 t/ha</b>" in text
    assert plain.explanation_sentence({"A": 0.0, "B": 0.0}).startswith("No factor")


def test_drivers_headline_marks_negative_association():
    text = plain.drivers_headline(
        "Farming", 0.6434, ["fertilizer", "pesticide use", "rainfall"], {"pesticide use"}
    )
    assert "<b>Farming practices</b> matter most (<b>64%</b>" in text
    assert "(more pesticide use is associated with lower yield)" in text
    assert "fertilizer</b>, then" in text and "(more fertilizer" not in text


def test_india_sentences():
    assert "<b>83%</b>" in plain.lesson_crop(0.8337)[1]
    big, text = plain.lesson_lookup(7.8075, 8.1195, beats_every_model=True)
    assert big == "7.81 vs 8.12" and "It beats every model." in text
    assert "<b>8 of the 10</b>" in plain.lesson_seasons(8, 10)[1]
    assert " beats the best model" in plain.india_headline(7.8075, 8.1195, 0.8337)
    assert "does not beat" in plain.india_headline(9.0, 8.1195, 0.8337)


def test_tie_note_counts_models_within_a_hundredth():
    assert plain.tie_note([0.6461, 0.6462, 0.6526, 0.6533, 0.6533, 0.6635]).startswith("The top 5")
    assert plain.tie_note([0.60, 0.70]) == ""


def test_curve_shapes_and_captions():
    x = np.linspace(4.5, 8.5, 41)  # step 0.1, so 6.5 is a grid point
    assert plain.curve_shape(x, -((x - 6.5) ** 2)) == ("peaks", x[np.argmax(-((x - 6.5) ** 2))])
    caption = plain.curve_caption("Soil pH", "pH", x, 6.4 - 0.1 * (x - 6.5) ** 2)
    assert "peaks around <b>pH 6.5</b>" in caption
    assert plain.curve_shape(x, 0.5 * x)[0] == "rises"
    assert plain.curve_shape(x, -0.5 * x)[0] == "falls"
    assert plain.curve_shape(x, 0.01 * x)[0] == "barely changes"
    assert plain.curve_shape(x, (x - 6.5) ** 2)[0] == "dips"
    shape, at = plain.curve_shape(x, np.minimum(x, 6.0))  # rises, then flat from 6.0
    assert shape == "rises then levels off" and 5.5 < at <= 6.0
    rise = plain.curve_caption("Fertilizer", "kg/ha", [50, 300], [4.547, 7.599])
    assert "<b>+3.05 t/ha</b> from lowest to highest" in rise
