"""Plain-language text for the dashboard, always generated from real numbers.

No Streamlit here, so every sentence can be unit-tested (tests/test_plain.py).
Sentences are HTML: key numbers are wrapped in <b>, which the theme colours green.
Wording stays descriptive ("the model predicts", "is associated with"), never causal.
"""

import numpy as np

from src import config
from ui import labels

MINUS = "−"
# The India data mixes units across crops, so its errors are never labelled t/ha.
INDIA_UNIT = "yield units"

FRIENDLY = {c: labels.name(c, simple=True) for c in config.FEATURES}  # Simple-mode names
GROUP_PLAIN = {g: labels.group(g, simple=True) for g in labels.GROUPS}
PLAIN_MODEL_NAMES = {m: labels.model(m, simple=True) for m in labels.MODELS}
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
    "feature_ablation": (
        "Feature ablation: the same model family is re-tuned with one feature group added at a time "
        "(target encoding = past average yields; lags = last years' yields; trend = the year), then "
        "with all of them. Every row is scored on the same 5 time-ordered folds."
    ),
    "persistence": (
        "A second no-ML reference: predict each crop × state with its own yield of the previous "
        "year (the look-up when it was not grown last year)."
    ),
    "segments": (
        "Rows the model finds hard: crop × state pairs never grown before, crops never seen in "
        "training, and rows with no record in the previous year."
    ),
    "time_cv": (
        "Cross-validation where every check scores the model only on years later than the "
        "ones it trained on. sd = spread across the checks."
    ),
    "india_units": (
        "The India records use each crop's own unit: most crops are in tonnes per hectare, "
        "but in Indian crop statistics cotton is counted in bales of 170 kg and jute, mesta "
        "and sannhemp in bales of 180 kg (coconut, counted in nuts, is left out). One error "
        "across all crops therefore has no single unit."
    ),
    "interval": (
        "90% range: if the model sees fields like this one many times, the real yield should "
        "land inside this range about 9 times in 10. It was checked on 960 fields the model "
        "never saw."
    ),
    "interval_method": (
        "Split-conformal prediction interval: the model's prediction ± the 90th-percentile "
        "(finite-sample rank) of its absolute out-of-fold cross-validation errors on the "
        "3,840 training fields."
    ),
    "permutation": (
        "Permutation importance: shuffle one input across the test fields and see how much the "
        "typical error grows. Bigger growth = the model relies on it more."
    ),
    "agreement": (
        "Three different ways to rank the inputs. If they put the same inputs at the top, the "
        "ranking is not an artefact of one method."
    ),
    "spearman": "Spearman ρ compares two rankings: 1 = identical order, 0 = unrelated.",
    "ph": (
        "Correlation measures how well a straight line fits. A hump-shaped effect (best in the "
        "middle) can matter a lot and still have a correlation near zero."
    ),
    "bootstrap": (
        "The model is refitted on 500 resamples of the training fields; the bar spans the middle "
        "95% of the coefficients it gets. A bar crossing 0 means the sign is not certain."
    ),
    "ice": (
        "Each thin line is one real field with only this input changed; the thick line is their "
        "average. Lines that run parallel mean the effect is the same for every field."
    ),
    "heatmap": (
        "Every cell is a prediction for the current field with these two inputs changed; all other "
        "inputs stay as set. The grid only covers values seen in the training data."
    ),
    "coverage_groups": (
        "Share of the 960 test fields in each group whose real yield fell inside the 90% range. "
        "One width is used for every field, so some groups are covered more and some less."
    ),
    "abtest": (
        "Each bar is how much one input's push changes between the two scenarios (SHAP of B minus "
        "SHAP of A). The bars add up to the difference between the two predictions."
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
SIMPLE_TIPS = {  # Simple mode: the same ⓘ in everyday words, no technical terms
    "rmse": (
        "Typical error: how far a prediction usually lands from the real yield, in t/ha. "
        "Big misses count extra. Lower is better."
    ),
    "r2": (
        "How much of the difference between fields the model explains: 100% = all of it, "
        "0% = no better than guessing the average every time."
    ),
    "contributions": (
        "How much each factor pushed this prediction above (+) or below (−) the average "
        "prediction, in t/ha."
    ),
    "abtest": (
        "Each bar is how much one factor's push changes between the two scenarios. The bars "
        "add up to the difference between the two predictions."
    ),
    "permutation": (
        "Scramble one factor across fields the model never saw and see how much the typical "
        "error grows. Bigger growth = the model relies on it more."
    ),
    "shap": (
        "Each prediction is split into how much each factor pushed it. The average size of "
        "those pushes ranks how much the model relies on each factor."
    ),
    "cv": (
        "A fair test: the model is scored on fields it did not learn from, several times over, "
        "so a good score is not down to luck."
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
BY_FAR = 2.0  # the top factor "matters most, by far" when it is at least twice the second
CLOSE_BEHIND = 1.05  # a runner-up within 5% of the best is "close behind"


def tip_text(key: str, simple: bool) -> str:
    """The ⓘ text for `key`: the everyday version in Simple mode when there is one."""
    return SIMPLE_TIPS.get(key, TIPS[key]) if simple else TIPS[key]


# --- numbers --------------------------------------------------------------------------


def signed(value: float, decimals: int = 2) -> str:
    """'+0.10' / '−0.15' with a typographic minus; a value that rounds to zero is '0.00'."""
    text = f"{value:+.{decimals}f}"
    return text[1:] if float(text) == 0 else text.replace("-", MINUS)


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


# --- chart headlines (Simple mode: one line above each chart, from the real numbers) ----


def curve_headline(name: str, unit: str, grid, values) -> str:
    """'Fertilizer: more of it, higher predicted yield' style takeaway for a prediction curve."""
    shape, at = curve_shape(grid, values)
    where = value_text(at, unit, decimals_for_span(float(np.ptp(np.asarray(grid, dtype=float))))) if at else ""
    lower = labels.lower(name)
    return {
        "barely changes": f"{name} barely changes the prediction",
        "peaks": f"{name} works best around {where}",
        "dips": f"{name} is least helpful around {where}",
        "rises": f"More {lower}, higher predicted yield",
        "falls": f"More {lower}, lower predicted yield",
        "rises then levels off": f"{name} helps up to about {where}, then levels off",
        "falls then levels off": f"{name} lowers the prediction up to about {where}, then levels off",
    }[shape]


def importance_headline(ranked: list[tuple[str, float]]) -> str:
    """ranked: (friendly name, importance), largest first -> 'Fertilizer matters most, by far'."""
    (first, a), (second, b) = ranked[0], ranked[1]
    if b <= 0 or a >= BY_FAR * b:
        return f"{first} matters most, by far"
    return f"{first} matters most, then {labels.lower(second)}"


def contribution_headline(contributions: dict[str, float]) -> str:
    """The factor with the biggest push on this prediction, up or down."""
    top = max(contributions, key=lambda k: abs(contributions[k]))
    value = contributions[top]
    if abs(value) < 0.005:
        return "No factor moves this prediction much"
    verb = "helps this field most" if value > 0 else "holds this field back most"
    return f"{top} {verb} ({signed(value)} t/ha)"


def gauge_headline(rating_word: str, share_below: float) -> str:
    word = rating_word.lower()
    return (
        f"{_article(word).capitalize()} {word} yield: higher than {int(share_below * 100)}% "
        "of the fields the model learned from"
    )


def gap_headline(diff: float) -> str:
    """A vs B: which scenario the model expects to yield more."""
    if abs(diff) < 0.005:
        return "B and A should yield about the same"
    return f"B should yield {abs(diff):.2f} t/ha {'more' if diff > 0 else 'less'} than A"


def lowest_error_headline(errors: dict[str, float]) -> str:
    """errors: plain predictor phrase -> typical error. Names the smallest and a close runner-up."""
    ranked = sorted(errors.items(), key=lambda kv: kv[1])
    (best, low), (second, next_low) = ranked[0], ranked[1]
    text = f"{best[:1].upper()}{best[1:]} makes the smallest errors"
    if next_low <= CLOSE_BEHIND * low:
        text += f", with {second} close behind"
    return text


def crops_headline(n_well: int, n_crops: int) -> str:
    return f"The model works well for {n_well} of the {n_crops} most common crops"


def result_sentence(prediction: float, lo: float | None, hi: float | None) -> str:
    """Predict, Simple mode: the whole result in one sentence."""
    text = f"This field should produce about <b>{prediction:.1f} tonnes per hectare</b>"
    if lo is not None and hi is not None:
        text += f" (usually between <b>{lo:.1f}</b> and <b>{hi:.1f}</b>)"
    return text + "."


def ph_sentence_simple(r: float, optimum: dict) -> str:
    """The soil pH surprise without statistics words."""
    link = (
        f"A straight-line check finds almost no link between soil pH and yield (<b>{r:+.2f}</b> on a "
        "scale from −1 to 1)."
    )
    if not optimum["interior"]:
        return link + (
            " And the model's prediction is highest at the edge of the range "
            f"(pH {optimum['peak_x']:.1f}), so this data shows no hidden curve."
        )
    return link + (
        f" Yet the model's prediction rises and then falls, best around pH <b>{optimum['low']:.1f}"
        f"–{optimum['high']:.1f}</b>: a hump has no straight-line trend, so that check misses it."
    )


def methods_headline_simple(top: str | None, mean_rho: float, model_name: str) -> str:
    """What matters, Simple mode: the top factor and whether three ways of measuring agree."""
    order = ("nearly the same order" if mean_rho >= 0.8 else "a similar order" if mean_rho >= 0.5
             else "quite different orders")
    if top:
        return (f"<b>{top}</b> matters most to the {model_name} model: three different ways of "
                f"measuring all put it first, and they rank the other factors in {order}.")
    return (f"Three different ways of measuring disagree on the top factor for the {model_name} model, "
            f"and rank the factors in {order}.")


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
    """How many models are within 0.01 t/ha of the first one (a practical tie).

    errors: test errors in leaderboard order (the first is the selected model).
    """
    close = sum(abs(e - errors[0]) < 0.01 for e in errors)
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
        f"(<b>{lookup_rmse:.2f} {INDIA_UNIT}</b> typical error) {comparison} the best model "
        f"(<b>{best_rmse:.2f} {INDIA_UNIT}</b>)."
    )


def lesson_crop(crop_share: float, history_share: float = 0.0) -> tuple[str, str]:
    if history_share > crop_share:  # Phase 4 model: the lag features lead
        return pct(history_share), (
            f"On real farms, the best guide is <b>the recent past</b>: the last few years' yields make up "
            f"<b>{pct(history_share)}</b> of what the chosen model relies on, crop type <b>{pct(crop_share)}</b>."
        )
    return pct(crop_share), (
        f"On real farms, <b>which crop you grow</b> matters most: crop type makes up "
        f"<b>{pct(crop_share)}</b> of what the best model relies on."
    )


def lesson_lookup(lookup_rmse: float, best_rmse: float, lookup_wins: bool) -> tuple[str, str]:
    verdict = "It beats the chosen model." if lookup_wins else "The chosen model beats it."
    return f"{lookup_rmse:.2f} vs {best_rmse:.2f}", (
        f"A simple <b>look-up</b> of each crop's usual yield in each state is typically off by "
        f"<b>{lookup_rmse:.2f} {INDIA_UNIT}</b>, against <b>{best_rmse:.2f} {INDIA_UNIT}</b> "
        "for the chosen "
        f"model. {verdict}"
    )


def lesson_seasons(n_negative: int, n_crops: int) -> tuple[str, str]:
    lead = ("The models tell crops apart well but struggle with good vs bad seasons for the same crop"
            if 2 * n_negative > n_crops else "Good vs bad seasons for the same crop are the hardest part")
    return f"{n_negative} of {n_crops}", (
        f"{lead}: for <b>{n_negative} of the {n_crops}</b> most common crops, they predict that "
        "crop's yield no better than simply guessing its average every time."
    )


def interval_sentence(lo: float, hi: float, coverage: float, n_test: int) -> str:
    """Simple-mode reading of the 90% range, with its measured test-set coverage."""
    return (
        f"For fields like this, the real yield usually lands between <b>{lo:.2f}</b> and "
        f"<b>{hi:.2f} t/ha</b>: on <b>{n_test}</b> fields the model never saw, the range held "
        f"<b>{coverage:.0%}</b> of the time (aim: 90%)."
    )


def interval_method(q: float, n_cal: int, coverage: float, n_test: int) -> str:
    """Technical-mode description of how the 90% range is computed."""
    return (
        f"Split-conformal 90% interval: prediction ± <b>{q:.3f} t/ha</b>, the "
        f"⌈(n+1)·0.9⌉-th smallest absolute out-of-fold 5-fold CV residual of the deployed model "
        f"(n = {n_cal:,} training fields). Empirical coverage on the {n_test} held-out test fields: "
        f"<b>{coverage:.1%}</b>."
    )


def ph_sentence(r: float, rho: float, optimum: dict, lasso_kept: bool) -> str:
    """The soil pH paradox in one or two sentences, from the measured numbers.

    Says plainly when the model's curve is NOT hump-shaped, instead of forcing the story.
    """
    corr = (
        f"Soil pH barely correlates with yield (Pearson r = <b>{r:+.2f}</b>, Spearman ρ = "
        f"<b>{rho:+.2f}</b>), so a correlation filter would have dropped it."
    )
    if not optimum["interior"]:
        return corr + (
            " But the model's response to pH is not hump-shaped either: it is highest at the edge "
            f"of the range (pH {optimum['peak_x']:.1f}), so this data shows no hidden curved effect."
        )
    lasso = (
        "Lasso kept it, but only with a tiny straight-line coefficient"
        if lasso_kept
        else "Lasso, which only fits straight lines, set it to zero"
    )
    return corr + (
        f" Yet the model's prediction rises and then falls, best around pH <b>{optimum['low']:.1f}"
        f"–{optimum['high']:.1f}</b> (peak at pH {optimum['peak_x']:.1f}): a hump has almost no "
        f"straight-line trend, which is all that correlation measures. {lasso}."
    )


def india_verdict(r: dict, simple: bool) -> tuple[str, str]:
    """(title, html) for the "beat the look-up?" card, every number from the Phase 4 tables.

    r: model (display name), model_cv, lookup_cv, model_test, lookup_test, persistence_test,
    strict_test, group (feature set that helps most), gain (its CV RMSE cut), beats_cv.
    """
    u = INDIA_UNIT
    won_test = r["model_test"] < r["lookup_test"]
    if r["beats_cv"] and won_test:
        title = "How the model beats the look-up"
    elif r["beats_cv"]:
        title = "Better in cross-validation, not on the test years"
    else:
        title = "Why can't machine learning beat a look-up?"
    if simple:
        text = (
            f"On the 2016–2020 years it never saw, the model is typically off by <b>{r['model_test']:.2f} {u}</b>, "
            f"the look-up by <b>{r['lookup_test']:.2f}</b>. "
        )
        text += (
            f"What helps most is <b>{r['group']}</b>. " if r["gain"] > 0 else ""
        )
        if r["persistence_test"] < r["lookup_test"]:
            close = "as good as" if r["persistence_test"] <= r["model_test"] else "nearly as good as"
            text += (
                f"But just repeating last year's yield is off by <b>{r['persistence_test']:.2f}</b>, {close} "
                "the model: knowing last year matters more than the weather or farm inputs. "
            )
        tie = r.get("tie_persistence")  # (p, distinguishable) from tie_check.csv, or None
        if tie is not None and not tie[1] and r["persistence_test"] < r["lookup_test"]:
            text += ("In the five validation periods, the model and repeating last year's yield are "
                     "<b>statistically a tie</b>. ")
        text += "This assumes last year's real yield is known when predicting."
        return title, text
    text = (
        f"Selected by time-ordered cross-validation, <b>{r['model']}</b> scores CV RMSE "
        f"<b>{r['model_cv']:.2f}</b> against <b>{r['lookup_cv']:.2f}</b> for the crop × state look-up "
        f"on the same folds; on the 2016–2020 test years (used once) <b>{r['model_test']:.2f}</b> against "
        f"<b>{r['lookup_test']:.2f}</b> {u}. "
    )
    if not r["beats_cv"]:
        text += (
            "Yields persist strongly by crop and state, and the other inputs carry little signal: one annual "
            "rainfall figure per state, and fertilizer and pesticide that are national yearly averages. "
        )
    if r["gain"] > 0:
        text += (
            f"The single feature group that closes most of the gap is <b>{r['group']}</b>: added alone to the "
            f"base inputs it cuts CV RMSE by <b>{r['gain']:.2f}</b>. "
        )
    if r["persistence_test"] <= r["model_test"]:
        text += (
            f"A caution: simply repeating last year's yield (no ML) scores <b>{r['persistence_test']:.2f}</b> on "
            "the test years, at least as good as the model, so the gain comes from recent history rather than "
            "from the weather or input columns. "
        )
    elif r["persistence_test"] < r["lookup_test"]:
        text += (
            f"Repeating last year's yield (no ML) also beats the look-up (<b>{r['persistence_test']:.2f}</b>); "
            f"the model adds <b>{r['persistence_test'] - r['model_test']:.2f}</b> on top of that. "
        )
    text += (
        "These scores assume last year's actual yields are known when predicting (one year ahead). "
        f"If only 1997–2015 were known for every test year, the model would score <b>{r['strict_test']:.2f}</b>"
    )
    text += (f", worse than the look-up ({r['lookup_test']:.2f})." if r["strict_test"] >= r["lookup_test"]
             else ", still better than the look-up.")
    text += tie_sentence(r.get("tie_lookup"), r.get("tie_persistence"))
    return title, text


def tie_sentence(lookup: tuple | None, persistence: tuple | None) -> str:
    """The paired CV tie check in one or two sentences; each argument is (p, distinguishable)."""
    if lookup is None and persistence is None:
        return ""
    parts = [
        f"{'distinguishable' if test[1] else '<b>not distinguishable</b>'} from {name} (p = {test[0]:.3f})"
        for name, test in (("the look-up", lookup), ("last year's yield", persistence)) if test is not None
    ]
    text = " Paired t-test on the 5 CV folds: the model is " + "; ".join(parts) + "."
    if persistence is not None and not persistence[1]:
        text += " So the gain over simply repeating last year's yield is within fold-to-fold noise."
    return text


def kaggle_tie_note(p_values: dict[str, float], distinguishable: dict[str, bool]) -> str:
    """Model comparison: are the selected model's CV wins over the runners-up real? Names are display names."""
    if not p_values:
        return ""
    ties = [n for n in p_values if not distinguishable[n]]
    shown = " or ".join(f"{n} (p = {p_values[n]:.2f})" for n in p_values)
    if len(ties) == len(p_values):
        return f"On the 5 CV folds the selected model is not statistically distinguishable from {shown}: a tie."
    return f"Paired t-test on the 5 CV folds against {shown}."


def india_phase4_headline(r: dict, simple: bool) -> str:
    """Page headline after Phase 4: does the selected model beat the crop x state look-up?"""
    better = r["model_test"] < r["lookup_test"]
    if simple:
        return (
            f"On real Indian crop records, the chosen model is typically off by <b>{r['model_test']:.2f} {INDIA_UNIT}</b> "
            f"on years it never saw, against <b>{r['lookup_test']:.2f}</b> for a simple look-up of each crop's usual "
            f"yield in each state: the {'model' if better else 'look-up'} is better."
        )
    return (
        f"Selected by time-ordered CV, <b>{r['model']}</b> scores test RMSE <b>{r['model_test']:.2f}</b> on 2016–2020, "
        f"{'beating' if better else 'losing to'} the crop × state look-up (<b>{r['lookup_test']:.2f}</b>); "
        f"CV RMSE <b>{r['model_cv']:.2f}</b> vs <b>{r['lookup_cv']:.2f}</b> on the same folds."
    )
