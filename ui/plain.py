"""Plain-language text for the dashboard, always generated from real numbers.

No Streamlit here, so every sentence can be unit-tested (tests/test_plain.py).
Sentences are HTML: key numbers are wrapped in <b>, which the theme colours indigo.
Wording stays descriptive ("the model predicts", "is associated with"), never causal.
"""

import numpy as np

MINUS = "−"

FRIENDLY = {  # feature -> everyday name (Simple mode charts and sentences)
    "soil_ph": "Soil pH",
    "soil_moisture": "Soil moisture",
    "nitrogen_content": "Soil nitrogen",
    "phosphorus_content": "Soil phosphorus",
    "potassium_content": "Soil potassium",
    "avg_temperature": "Temperature",
    "total_rainfall": "Rainfall",
    "sunlight_hours": "Sunshine",
    "fertilizer_amount": "Fertilizer",
    "pesticide_usage": "Pesticide use",
    "irrigation_frequency": "Irrigation",
    "crop_type": "Crop",
    "region": "Region",
    "season": "Season",
}
GROUP_PLAIN = {
    "Farming": "Farming practices",
    "Soil": "Soil conditions",
    "Weather": "Weather",
    "Context": "Crop, region and season",
}
PLAIN_MODEL_NAMES = {
    "gradient_boost": "Boosted decision trees",
    "polynomial": "Curved formula",
    "polynomial_deg3": "Extra-curved formula",
    "lasso": "Straight-line formula (trimmed)",
    "ridge": "Straight-line formula (smoothed)",
    "multiple_lr": "Straight-line formula",
    "random_forest": "Forest of decision trees",
    "simple_lr": "Fertilizer-only line",
    "dummy_mean": "Guessing the average",
}
TIPS = {  # one-line explanations behind every ⓘ (both modes)
    "t/ha": "Tonnes of harvest per hectare. A hectare is 10,000 m², about 1.4 football pitches.",
    "average": (
        "The model's average prediction over the fields it learned from: the starting "
        "point before this field's own conditions are counted."
    ),
    "contributions": (
        "How much each factor pushes this prediction above (+) or below (−) the average "
        "prediction, in t/ha (SHAP values)."
    ),
    "gauge": (
        "Bands are the quarters of the yields the model learned from: Low is the bottom 25%, "
        "Typical up to the middle, Good up to 75%, Excellent the top 25%."
    ),
    "whatif": (
        "Sweeps one factor from its lowest to its highest value in the data while every other "
        "input stays as set above. It shows the model's predictions, not a guaranteed effect."
    ),
    "response": (
        "Partial dependence: the average prediction over 300 real training fields as one factor "
        "is swept across its range. It shows what the model learned, not cause and effect."
    ),
    "rmse": (
        "RMSE (root mean squared error): the typical size of a prediction's miss, in t/ha. "
        "Big misses count extra. Lower is better."
    ),
    "mae": "MAE (mean absolute error): the average size of a miss, in t/ha. Lower is better.",
    "metrics": (
        "RMSE: typical miss, big misses count extra. MAE: average miss. R²: share of the "
        "differences explained (1 = all, 0 = no better than guessing the average)."
    ),
    "r2": (
        "R²: the share of the differences in yield that the model accounts for. 1 = all of "
        "them, 0 = no better than guessing the average, below 0 = worse than that."
    ),
    "baseline": (
        "Baseline: a 'model' that predicts the same average yield every time. Any useful "
        "model has to beat it."
    ),
    "test": (
        "Held-out test set: fields kept aside while the models were built and used once, "
        "at the end, for the final score."
    ),
    "shap": (
        "SHAP splits each prediction into pushes from each factor. The average size of those "
        "pushes (mean |SHAP|) ranks how much the model relies on each factor."
    ),
    "cv": (
        "Cross-validation: train on part of the data and score on the rest, several times "
        "over, to check a score is not down to luck."
    ),
    "lookup": (
        "No machine learning: predict each crop's average yield in that state over the "
        "training years."
    ),
    "ablation": (
        "Ablation: re-run every model without the fertilizer and pesticide columns to see "
        "how much they help (Δ = change)."
    ),
    "time_cv": (
        "Cross-validation where every check scores the model only on years later than the "
        "ones it trained on. sd = spread across the checks."
    ),
    "relative": (
        "The crop's typical error divided by its average yield. Under 50% is labelled 'works well'."
    ),
    "improvement": "How much smaller the best model's typical error is than guessing the average.",
    "beeswarm": (
        "Each dot is one field: its position shows how much that factor pushed the "
        "prediction; its colour shows whether the factor's value was high or low."
    ),
}
INPUT_TIPS = {  # slider help (both modes)
    "soil_ph": "How acidic (below 7) or alkaline (above 7) the soil is.",
    "soil_moisture": "How much water the soil holds, in %.",
    "nitrogen_content": "Share of this plant nutrient in the soil, in %.",
    "phosphorus_content": "Share of this plant nutrient in the soil, in %.",
    "potassium_content": "Share of this plant nutrient in the soil, in %.",
    "sunlight_hours": "Total hours of sunshine, as recorded in the data.",
    "fertilizer_amount": "Kilograms of fertilizer per hectare (kg/ha).",
    "pesticide_usage": "Litres of pesticide per hectare (L/ha).",
    "irrigation_frequency": "How many times a week the field is watered.",
}
GLOSSARY = {  # "How it works" tab
    "RMSE": (
        "Root mean squared error: the typical size of a prediction's miss, in t/ha. Big misses "
        "count extra. Lower is better."
    ),
    "R²": (
        "The share of the differences in yield that the model accounts for. 1 = all of them, "
        "0 = no better than guessing the average."
    ),
    "SHAP": TIPS["shap"],
    "Cross-validation": (
        "Train on part of the data and score on the rest, several times over, to check a "
        "score is not down to luck."
    ),
    "Baseline": (
        "A 'model' that predicts the same average yield every time. Any useful model has to "
        "beat it."
    ),
}

FLAT_RANGE = 0.1  # t/ha: a sweep that moves the prediction less than this "barely changes" it
WORKS_WELL = 0.5  # per-crop typical error under 50% of the crop's average yield = "works well"


# --- numbers --------------------------------------------------------------------------


def signed(value: float, decimals: int = 2) -> str:
    """'+0.10' / '−0.15' with a typographic minus."""
    return f"{value:+.{decimals}f}".replace("-", MINUS)


def pct(share: float) -> str:
    """0.6412 -> '64%'."""
    return f"{share:.0%}"


def decimals_for_span(span: float) -> int:
    """Sensible rounding for a factor value, from the width of its range."""
    return 2 if span <= 3 else 1 if span <= 20 else 0


def value_text(value: float, unit: str, decimals: int) -> str:
    """'6.6' + 'pH' -> 'pH 6.6'; '805' + 'mm' -> '805 mm'."""
    number = f"{value:.{decimals}f}"
    if unit == "pH":
        return f"pH {number}"
    return f"{number} {unit}" if unit else number


def rating(prediction: float, p25: float, p50: float, p75: float) -> str:
    """Low / Typical / Good / Excellent against the quartiles of the training yields."""
    if prediction < p25:
        return "Low"
    if prediction < p50:
        return "Typical"
    return "Good" if prediction < p75 else "Excellent"


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


# --- curves ---------------------------------------------------------------------------


def curve_shape(grid, values) -> tuple[str, float | None]:
    """Classify a prediction curve; returns (shape, the x-value that matters or None).

    Shapes: barely changes / peaks / dips / rises / falls / rises (falls) then levels off.
    """
    x, y = np.asarray(grid, dtype=float), np.asarray(values, dtype=float)
    span = y.max() - y.min()
    if span < FLAT_RANGE:
        return "barely changes", None
    last = len(y) - 1
    edge = 0.1 * last  # a turning point within the outer 10% counts as an end, not a peak
    i, j = int(y.argmax()), int(y.argmin())
    if edge < i < last - edge and min(y[i] - y[0], y[i] - y[-1]) > 0.25 * span:
        return "peaks", float(x[i])
    if edge < j < last - edge and min(y[0] - y[j], y[-1] - y[j]) > 0.25 * span:
        return "dips", float(x[j])
    rising = y[-1] > y[0]
    third = max(len(y) // 3, 1)
    early, late = abs(y[third] - y[0]), abs(y[-1] - y[-1 - third])
    if late < 0.25 * early:  # the last third barely moves compared with the first third
        target = y[0] + 0.9 * (y[-1] - y[0])  # where 90% of the total change is reached
        reached = y >= target if rising else y <= target
        return ("rises" if rising else "falls") + " then levels off", float(x[reached.argmax()])
    return ("rises" if rising else "falls"), None


def curve_caption(name: str, unit: str, grid, values) -> str:
    """One plain sentence describing a prediction curve, e.g. 'Soil pH: … peaks around pH 6.6.'"""
    shape, at = curve_shape(grid, values)
    y = np.asarray(values, dtype=float)
    decimals = decimals_for_span(float(np.ptp(np.asarray(grid, dtype=float))))
    where = value_text(at, unit, decimals) if at is not None else ""
    change = f"{signed(y[-1] - y[0])} t/ha"
    text = {
        "barely changes": (
            f"predicted yield barely changes (less than <b>{FLAT_RANGE} t/ha</b> across the "
            "whole range)"
        ),
        "peaks": f"predicted yield peaks around <b>{where}</b>",
        "dips": f"predicted yield is lowest around <b>{where}</b>",
        "rises": f"predicted yield rises across the whole range (<b>{change}</b> from lowest to highest)",
        "falls": f"predicted yield falls as it goes up (<b>{change}</b> across the range)",
        "rises then levels off": f"predicted yield rises, then levels off above about <b>{where}</b>",
        "falls then levels off": f"predicted yield falls, then levels off above about <b>{where}</b>",
    }[shape]
    return f"{name}: {text}."


# --- headline sentences ---------------------------------------------------------------


def predict_headline(prediction: float, rating_word: str, share_below: float, n_fields: int) -> str:
    word = rating_word.lower()
    higher_than = f"{int(share_below * 100)}%"  # rounded down: never claim 100% for 99.6%
    return (
        f"For these field conditions the model predicts <b>{prediction:.2f} t/ha</b>: "
        f"{_article(word)} <b>{word}</b> yield, higher than <b>{higher_than}</b> of the "
        f"{n_fields:,} fields it learned from."
    )


def explanation_sentence(contributions: dict[str, float]) -> str:
    """'<top boost> is adding about +x.xx t/ha; <top drag> is taking away about −x.xx t/ha.'"""
    boost = max(contributions, key=contributions.get)
    drag = min(contributions, key=contributions.get)
    parts = []
    if contributions[boost] > 0:
        parts.append(f"<b>{boost}</b> is adding about <b>{signed(contributions[boost])} t/ha</b>")
    if contributions[drag] < 0:
        parts.append(
            f"<b>{drag}</b> is taking away about <b>{signed(contributions[drag])} t/ha</b>"
        )
    if not parts:
        return "No factor moves this prediction away from the average."
    return "; ".join(parts) + "."


def model_headline(rmse: float, mean_yield: float, r2: float, baseline_rmse: float) -> str:
    return (
        f"Our best model's predictions are typically off by about <b>{rmse:.2f} t/ha</b>, "
        f"roughly <b>{pct(rmse / mean_yield)}</b> of an average harvest, and it explains about "
        f"<b>{pct(r2)}</b> of the differences between fields. That's "
        f"<b>{pct(1 - rmse / baseline_rmse)}</b> better than always guessing the average."
    )


def tie_note(errors: list[float]) -> str:
    """How many of the top models are within 0.01 t/ha of the best (a practical tie)."""
    close = sum(e - errors[0] < 0.01 for e in errors)
    return (
        f"The top {close} are within 0.01 t/ha of each other: practically a tie."
        if close > 1
        else ""
    )


def drivers_headline(
    top_group: str, share: float, top_features: list[str], negative: set[str]
) -> str:
    """top_features: the three biggest (friendly, lower-case); negative: those whose higher
    values are associated with lower yield."""

    def named(f: str) -> str:
        note = f" (more {f} is associated with lower yield)" if f in negative else ""
        return f"<b>{f}</b>{note}"

    first, second, third = top_features
    return (
        f"<b>{GROUP_PLAIN.get(top_group, top_group)}</b> matter most "
        f"(<b>{pct(share)}</b> of what the model relies on). The biggest single factor is "
        f"{named(first)}, then {named(second)} and {named(third)}."
    )


def india_headline(lookup_rmse: float, best_rmse: float, crop_share: float) -> str:
    comparison = "beats" if lookup_rmse < best_rmse else "does not beat"
    return (
        f"On real Indian crop records, the crop grown explains most of the yield "
        f"(<b>{pct(crop_share)}</b>), and a simple crop-by-state look-up "
        f"(<b>{lookup_rmse:.2f} t/ha</b> typical error) {comparison} the best model "
        f"(<b>{best_rmse:.2f} t/ha</b>)."
    )


def lesson_crop(crop_share: float) -> tuple[str, str]:
    return pct(crop_share), (
        f"On real farms, <b>which crop you grow</b> matters most: crop type makes up "
        f"<b>{pct(crop_share)}</b> of what the best model relies on."
    )


def lesson_lookup(lookup_rmse: float, best_rmse: float, beats_every_model: bool) -> tuple[str, str]:
    verdict = "It beats every model." if beats_every_model else "It comes close to the models."
    return f"{lookup_rmse:.2f} vs {best_rmse:.2f}", (
        f"A simple <b>look-up</b> of each crop's usual yield in each state is typically off by "
        f"<b>{lookup_rmse:.2f} t/ha</b>, against <b>{best_rmse:.2f} t/ha</b> for the best "
        f"model. {verdict}"
    )


def lesson_seasons(n_negative: int, n_crops: int) -> tuple[str, str]:
    return f"{n_negative} of {n_crops}", (
        "The models tell crops apart well but struggle with good vs bad seasons for the same "
        f"crop: for <b>{n_negative} of the {n_crops}</b> most common crops, they predict that "
        "crop's yield no better than simply guessing its average every time (R² below zero)."
    )
