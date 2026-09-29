"""Streamlit dashboard: predict crop yield, see what drives it, and check it on real data.

Run from the project root: python run.py app   (or: python -m streamlit run app.py)
Requires python -m src.pipeline to have been run (models, tables and figures).
No model is trained here; every number comes from the saved model, results/tables/ or
data/processed/. Simple mode adds plain-language sentences (ui/plain.py), each generated
from those numbers; Technical mode shows the exact metrics and tables.
"""

import json
from functools import partial

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import RegressorMixin

from src import config
from src.data import load_split
from src.explain import RawFeatureExplainer
from ui import charts, plain
from ui.plain import FRIENDLY, PLAIN_MODEL_NAMES, TIPS
from ui.theme import INDIGO, card_title, headline_card, inject_theme, kpi_card, tip

st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾", layout="wide")
inject_theme()

TOP_CONTRIBUTIONS = 8  # the explanation chart shows the 8 largest; the rest are summed
TIE_TOLERANCE = 0.001  # models within 0.001 t/ha of the best test RMSE count as tied best
CURVE_POINTS = 40  # what-if and response curves sweep each factor over 40 values
CURVE_SAMPLE = 300  # response curves average the prediction over 300 training fields
SIMPLE, TECHNICAL = "Simple", "Technical"
WHY, WHAT_IF = "Why this prediction", "What if I change…"
PRESETS = ["Typical field", "Ideal field", "Drought year", "Low-input farm"]
MINUS = "−"  # typographic minus sign for signed numbers
MODEL_LABELS = {
    "dummy_mean": "Mean baseline",
    "simple_lr": "Simple linear (fertilizer only)",
    "multiple_lr": "Multiple linear",
    "polynomial": "Polynomial + Ridge",
    "polynomial_deg3": "Polynomial (degree 3) + Ridge",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "random_forest": "Random Forest",
    "gradient_boost": "XGBoost",
}
COLUMN_LABELS = {
    "model": "Model",
    "predictor": "Predictor",
    "crop": "Crop",
    "rmse": "RMSE",
    "mae": "MAE",
    "r2": "R²",
    "rmse_mean": "CV RMSE",
    "rmse_std": "CV RMSE (sd)",
    "mae_mean": "CV MAE",
    "mae_std": "CV MAE (sd)",
    "r2_mean": "CV R²",
    "r2_std": "CV R² (sd)",
    "fit_time_s": "Fit time (s)",
    "alpha_median": "Chosen α (median)",
    "rmse_with": "RMSE with",
    "rmse_without": "RMSE without",
    "delta_rmse": "Δ RMSE",
    "r2_with": "R² with",
    "r2_without": "R² without",
    "delta_r2": "Δ R²",
    "n_test_rows": "Test rows",
    "mean_yield": "Mean yield",
    "relative_rmse": "RMSE / mean yield",
    "median_abs_error": "Median |error|",
}
INDIA_TABLES = {  # pill label -> (saved table, one-sentence caption, ⓘ text)
    "Test results": (
        "india_test",
        "Every model trained on 1997–2015 and evaluated once on the 2016–2020 test years.",
        TIPS["metrics"],
    ),
    "Look-up baseline": (
        "india_reference_baselines",
        "Simple historical-average look-ups (no machine learning) compared with the best model.",
        TIPS["lookup"],
    ),
    "Ablation": (
        "india_ablation",
        (
            "Test error with vs without the per-hectare fertilizer and pesticide inputs "
            "(Δ > 0 means the inputs helped)."
        ),
        TIPS["ablation"],
    ),
    "Time-ordered CV": (
        "india_cv",
        (
            "Cross-validation on the training years, where each validation block is later "
            "than its training years."
        ),
        TIPS["time_cv"],
    ),
    "Per-crop errors": (
        "india_per_crop_rmse",
        "Test error of the best model for the 10 most common crops.",
        TIPS["relative"],
    ),
}


# --- data -------------------------------------------------------------------------


@st.cache_resource
def load_artifacts() -> tuple[RegressorMixin, dict, dict, RawFeatureExplainer]:
    """Load the saved best model, slider metadata and a SHAP explainer once per server."""
    required = [
        config.MODELS_DIR / "best_overall.joblib",
        config.MODELS_DIR / "model_names.json",
        config.APP_METADATA_JSON,
        config.KAGGLE_TRAIN_PARQUET,
    ]
    missing = [str(p.relative_to(config.ROOT)) for p in required if not p.exists()]
    if missing:
        st.error(
            f"Missing files: {missing}. Run `python run.py` (or python -m src.pipeline) first."
        )
        st.stop()
    model = joblib.load(config.MODELS_DIR / "best_overall.joblib")
    names = json.loads((config.MODELS_DIR / "model_names.json").read_text())
    metadata = json.loads(config.APP_METADATA_JSON.read_text())
    background = load_split()[0][config.FEATURES]  # training rows only, as SHAP background
    return model, names, metadata, RawFeatureExplainer(model, background)


@st.cache_data
def load_table(name: str) -> pd.DataFrame | None:
    """A saved results table, or None if it has not been generated."""
    path = config.TABLES_DIR / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def data_summary() -> dict:
    """Numbers from data/processed/ behind the plain-language sentences and presets."""
    train, test = load_split()
    yields = train[config.TARGET]
    return {
        "n_train": len(train),
        "n_test": len(test),
        "test_mean": float(test[config.TARGET].mean()),
        "sorted_yields": np.sort(yields.to_numpy()),
        "quartiles": [float(q) for q in yields.quantile([0.25, 0.5, 0.75])],
        "min": float(yields.min()),
        "max": float(yields.max()),
        "q10": train[config.NUMERIC_FEATURES].quantile(0.10).to_dict(),
    }


@st.cache_data
def held_out_predictions() -> pd.DataFrame:
    """The saved best model's predictions for the held-out test fields (no SHAP)."""
    test = load_split()[1]
    return pd.DataFrame(
        {
            "real": test[config.TARGET].to_numpy(),
            "predicted": load_artifacts()[0].predict(test[config.FEATURES]),
            "crop": test["crop_type"].to_numpy(),
            "region": test["region"].to_numpy(),
        }
    )


def sweep_grid(feature: str, metadata: dict) -> np.ndarray:
    """40 evenly spaced values from the training min to max (every whole number if integer)."""
    m = metadata["numeric"][feature]
    if m["is_integer"]:
        return np.arange(int(m["min"]), int(m["max"]) + 1)
    return np.linspace(m["min"], m["max"], CURVE_POINTS)


def _predict_sweep(rows: pd.DataFrame, feature: str, grid: np.ndarray) -> np.ndarray:
    """Mean prediction over `rows` at each grid value of `feature`."""
    repeated = rows.loc[rows.index.repeat(len(grid))].copy()
    repeated[feature] = np.tile(grid, len(rows))
    predictions = load_artifacts()[0].predict(repeated)
    return predictions.reshape(len(rows), len(grid)).mean(axis=0)


@st.cache_data
def response_curve(feature: str) -> pd.DataFrame:
    """Partial dependence: the average prediction over 300 training fields as `feature` sweeps."""
    sample = load_split()[0][config.FEATURES].sample(CURVE_SAMPLE, random_state=config.RANDOM_STATE)
    grid = sweep_grid(feature, load_artifacts()[2])
    return pd.DataFrame({"x": grid, "y": _predict_sweep(sample, feature, grid)})


@st.cache_data
def whatif_curve(feature: str, inputs: tuple) -> pd.DataFrame:
    """The prediction for this one field as `feature` sweeps; other inputs held as set."""
    row = pd.DataFrame([dict(zip(config.FEATURES, inputs))])
    grid = sweep_grid(feature, load_artifacts()[2])
    return pd.DataFrame({"x": grid, "y": _predict_sweep(row, feature, grid)})


# --- formatting -------------------------------------------------------------------


def readable(name: str) -> str:
    """'nitrogen_content' -> 'Nitrogen content', 'soil_ph' -> 'Soil pH', 'log_Area' -> 'log(Area)'."""
    if name.startswith("log_"):
        return f"log({name[4:]})"
    text = name.replace("_", " ")
    return (text[:1].upper() + text[1:]).replace(" ph", " pH")


def signed(value: float, decimals: int = 2) -> str:
    """'+0.10' / '−0.15' with a typographic minus."""
    return f"{value:+.{decimals}f}".replace("-", MINUS)


def input_label(col: str) -> str:
    """Readable input label with its unit."""
    unit = config.UNITS.get(col)
    return readable(col) + (f" ({unit})" if unit else "")


def decimals_for(col: str) -> int:
    """Display precision: R² to 2 decimals, row counts as integers, everything else 3."""
    return 2 if "r2" in col else 0 if col == "n_test_rows" else 3


def number(value: float, decimals: int) -> str:
    """Fixed decimals with a typographic minus; a value that rounds to zero shows as 0.00."""
    text = f"{value:.{decimals}f}"
    if float(text) == 0:
        text = f"{0:.{decimals}f}"
    return text.replace("-", MINUS)


def table_view(frame: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Readable headers and numbers, best value per metric column in indigo.

    "Best" is judged at the displayed precision, so values that look tied are all
    highlighted (e.g. two models that both show 0.646).
    """
    df = frame.copy()
    if "model" in df:
        df["model"] = df["model"].map(lambda m: MODEL_LABELS.get(m, m))
    if "predictor" in df:
        df["predictor"] = df["predictor"].str.replace(" x ", " × ").str.capitalize()
    numeric = list(df.select_dtypes("number"))
    metric_cols = [c for c in numeric if not c.endswith("_std") and not c.startswith("delta")]
    lower_better = [c for c in metric_cols if any(t in c for t in ("rmse", "mae", "median_abs"))]
    higher_better = [c for c in metric_cols if "r2" in c]
    best = {}
    for col in lower_better + higher_better:
        shown = df[col].round(decimals_for(col))
        best[col] = shown.min() if col in lower_better else shown.max()
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for col, target in best.items():
        hit = df[col].round(decimals_for(col)) == target
        styles.loc[hit, col] = f"color: {INDIGO}; font-weight: 700"
    # st.table ignores Styler header relabelling, so rename the columns themselves.
    labels = {c: COLUMN_LABELS.get(c, readable(c)) for c in df.columns}
    formats = {labels[c]: partial(number, decimals=decimals_for(c)) for c in numeric}
    styles = styles.rename(columns=labels)
    shown = df.rename(columns=labels)
    styler = shown.style.format(formats, na_rep="—").apply(lambda _: styles, axis=None)
    return styler.hide(axis="index")


def show_table(frame: pd.DataFrame) -> None:
    """A static HTML table (white cells, exact height; st.dataframe paints the canvas colour)."""
    st.table(table_view(frame))


# --- inputs -----------------------------------------------------------------------


def default_inputs(metadata: dict) -> dict:
    """Training medians for numeric inputs and the first level for categorical ones."""
    values = {}
    for col in config.NUMERIC_FEATURES:
        m = metadata["numeric"][col]
        values[col] = int(m["median"]) if m["is_integer"] else m["median"]
    for col in config.CATEGORICAL_FEATURES:
        values[col] = metadata["categorical"][col][0]
    return values


def preset_values(name: str, defaults: dict, metadata: dict) -> dict:
    """Input values for a scenario preset, derived from the training data.

    Crop, region and season always stay at the Typical defaults, so presets differ only in
    growing conditions and farming inputs.
    - Typical field: the defaults (training medians).
    - Ideal field: every numeric input at the value the model favours, i.e. where its
      response curve (average prediction over 300 training fields) is highest.
    - Drought year: rainfall and soil moisture at their 10th percentile in the training data;
      everything else Typical.
    - Low-input farm: fertilizer, pesticide and irrigation at their 10th percentile;
      everything else Typical.
    """
    values = dict(defaults)
    q10 = data_summary()["q10"]
    if name == "Ideal field":
        for col in config.NUMERIC_FEATURES:
            curve = response_curve(col)
            values[col] = curve["x"].iloc[curve["y"].idxmax()]
    elif name == "Drought year":
        values.update({c: q10[c] for c in ("total_rainfall", "soil_moisture")})
    elif name == "Low-input farm":
        values.update(
            {c: q10[c] for c in ("fertilizer_amount", "pesticide_usage", "irrigation_frequency")}
        )
    for col in config.NUMERIC_FEATURES:
        is_integer = metadata["numeric"][col]["is_integer"]
        values[col] = round(float(values[col])) if is_integer else float(values[col])
    return values


def set_inputs(values: dict) -> None:
    for col, value in values.items():
        st.session_state[f"in_{col}"] = value


def reset_inputs(defaults: dict) -> None:
    """Button callback: put every input back to its default."""
    set_inputs(defaults)
    st.session_state["preset"] = PRESETS[0]


def apply_preset(defaults: dict, metadata: dict) -> None:
    """Pill callback: load the chosen scenario into the inputs."""
    if st.session_state["preset"]:
        set_inputs(preset_values(st.session_state["preset"], defaults, metadata))


def clear_preset() -> None:
    """Moving an input by hand means no preset describes the field any more."""
    st.session_state["preset"] = None


def input_widget(col: str, metadata: dict) -> None:
    """One slider (numeric) or selectbox (categorical); the value lives in session state."""
    key, help_text = f"in_{col}", plain.INPUT_TIPS.get(col)
    if col in metadata["numeric"]:
        m = metadata["numeric"][col]
        low, high = (int(m["min"]), int(m["max"])) if m["is_integer"] else (m["min"], m["max"])
        st.slider(input_label(col), low, high, key=key, help=help_text, on_change=clear_preset)
    else:
        st.selectbox(
            input_label(col), metadata["categorical"][col], key=key, on_change=clear_preset
        )


# --- page -------------------------------------------------------------------------

model, names, metadata, explainer = load_artifacts()
summary = data_summary()
defaults = default_inputs(metadata)
for feature, default in defaults.items():  # initialise the input state once per session
    st.session_state.setdefault(f"in_{feature}", default)
st.session_state.setdefault("preset", PRESETS[0])

st.html(
    '<p class="app-title">🌾 Crop Yield Prediction</p>'
    '<p class="app-subtitle">Predict crop yield and see what drives it.</p>'
)
mode = st.segmented_control(
    "Mode",
    [SIMPLE, TECHNICAL],
    default=SIMPLE,
    required=True,
    key="mode",
    label_visibility="collapsed",
)
simple = mode != TECHNICAL
tab_predict, tab_models, tab_drivers, tab_real, tab_how = st.tabs(
    ["Predict", "Model comparison", "What drives yield", "Real-world validation", "How it works"]
)

with tab_predict:
    X_input = pd.DataFrame([{f: st.session_state[f"in_{f}"] for f in config.FEATURES}])
    prediction = float(model.predict(X_input)[0])
    explanation = explainer.explain(X_input)[0]
    base_value = float(explanation.base_values)
    contributions = pd.Series(explanation.values, index=explanation.feature_names)
    q25, q50, q75 = summary["quartiles"]
    rating = plain.rating(prediction, q25, q50, q75)
    share_below = np.searchsorted(summary["sorted_yields"], prediction) / summary["n_train"]
    headline_card(plain.predict_headline(prediction, rating, share_below, summary["n_train"]))

    with st.container(key="card-inputs"):
        head, scenario, button = st.columns([30, 50, 17], vertical_alignment="center")
        with head:
            card_title("Field conditions", "Move any input or pick a scenario.", inline=True)
        with scenario:
            st.pills(
                "Scenario",
                PRESETS,
                key="preset",
                on_change=apply_preset,
                args=(defaults, metadata),
                label_visibility="collapsed",
                width="stretch",
            )
        with button:
            st.button(
                "Reset to defaults",
                type="primary",
                icon=":material/restart_alt:",
                on_click=reset_inputs,
                args=(defaults,),
                key="reset",
                width="stretch",
            )
        # Soil (5 inputs) spans two sub-columns (3 + 2), so every group is at most 3 rows tall.
        columns = st.columns([2, 1, 1.15, 0.85], gap="large")
        for column, (group, features) in zip(columns, config.FEATURE_GROUPS.items()):
            with column:
                st.html(f'<p class="eyebrow">{group}</p>')
                if len(features) > 3:
                    for sub, part in zip(st.columns(2, gap="large"), (features[:3], features[3:])):
                        with sub:
                            for feature in part:
                                input_widget(feature, metadata)
                else:
                    for feature in features:
                        input_widget(feature, metadata)

    ranked = contributions.reindex(contributions.abs().sort_values(ascending=False).index)
    top, rest = ranked.iloc[:TOP_CONTRIBUTIONS], ranked.iloc[TOP_CONTRIBUTIONS:]
    name_of = (lambda f: FRIENDLY[f]) if simple else readable
    rows = [{"feature": name_of(f), "value": float(v)} for f, v in top.items()]
    other = "Other factors" if simple else "Other features"
    rows.append({"feature": f"{other} ({len(rest)})", "value": float(rest.sum())})
    chart_data = pd.DataFrame(rows).assign(order=range(len(rows)))

    hero_col, chart_col = st.columns([35, 65], gap="medium")
    if simple:
        with hero_col, st.container(key="hero-gauge", height="stretch"):
            st.html(f'<p class="gauge-top">Predicted yield{tip(TIPS["gauge"])}</p>')
            bands = [
                ("Low", summary["min"], q25),
                ("Typical", q25, q50),
                ("Good", q50, q75),
                ("Excellent", q75, summary["max"]),
            ]
            st.altair_chart(
                charts.gauge_chart(prediction, summary["min"], summary["max"], bands, height=96),
                theme=None,
            )
            st.html(
                f'<p class="gauge-value"><span class="num">{prediction:.2f}</span> '
                f'<span class="unit">t/ha</span></p>'
                f'<p class="gauge-rating"><b>{rating}</b>&nbsp; average prediction '
                f"{base_value:.2f} t/ha ({signed(prediction - base_value)})"
                f"{tip(TIPS['average'], left=True)}</p>"
            )
        with chart_col, st.container(key="card-explain"):
            toggle, pick = st.columns([11, 9], vertical_alignment="center")
            with toggle:
                view = st.segmented_control(
                    "Explain",
                    [WHY, WHAT_IF],
                    default=WHY,
                    required=True,
                    key="explain_view",
                    label_visibility="collapsed",
                )
            if view == WHAT_IF:
                with pick:
                    factor = st.selectbox(
                        "Factor",
                        config.NUMERIC_FEATURES,
                        format_func=lambda f: FRIENDLY[f],
                        key="whatif_feature",
                        label_visibility="collapsed",
                    )
                inputs = tuple(st.session_state[f"in_{f}"] for f in config.FEATURES)
                curve = whatif_curve(factor, inputs)
                unit = config.UNITS.get(factor, "")
                st.altair_chart(
                    charts.curve_chart(
                        curve, st.session_state[f"in_{factor}"], f"{FRIENDLY[factor]} ({unit})", 150
                    ),
                    theme=None,
                )
                st.html(
                    '<p class="plain-text">'
                    f"{plain.curve_caption(FRIENDLY[factor], unit, curve['x'], curve['y'])}"
                    f"{tip(TIPS['whatif'], left=True)}</p>"
                )
            else:
                with pick:
                    card_title("What moved it (t/ha)", help_text=TIPS["contributions"])
                st.altair_chart(charts.contribution_chart(chart_data, height=162), theme=None)
                friendly = {FRIENDLY[f]: float(v) for f, v in contributions.items()}
                st.html(f'<p class="plain-text">{plain.explanation_sentence(friendly)}</p>')
    else:
        boost, drag = contributions.idxmax(), contributions.idxmin()
        boost_text = (
            f"{readable(boost)} ({signed(contributions[boost])})"
            if contributions[boost] > 0
            else "none"
        )
        drag_text = (
            f"{readable(drag)} ({signed(contributions[drag])})"
            if contributions[drag] < 0
            else "none"
        )
        with hero_col, st.container(key="hero-box", height="stretch"):
            st.html(
                '<div class="hero">'
                '<p class="label">Predicted yield</p>'
                f'<p class="value">{prediction:.2f} <span>t/ha</span></p>'
                f'<p class="vs">vs benchmark average {base_value:.2f} t/ha '
                f"({signed(prediction - base_value)})</p><hr>"
                f'<p class="line">Biggest boost: <b>{boost_text}</b></p>'
                f'<p class="line">Biggest drag: <b>{drag_text}</b></p>'
                "</div>"
            )
        with chart_col, st.container(key="card-explain"):
            card_title("What moved this prediction (t/ha)", help_text=TIPS["contributions"])
            st.altair_chart(charts.contribution_chart(chart_data, height=176), theme=None)

with tab_models:
    results = load_table("test_results")
    if results is None:
        st.info("test_results.csv not found — run `python run.py` first.")
    else:
        ranked = results.sort_values("rmse").reset_index(drop=True)
        models_only = ranked[ranked["model"] != "dummy_mean"]
        best = models_only.iloc[0]
        best_r2 = models_only.loc[models_only["r2"].idxmax()]
        baseline = ranked.loc[ranked["model"] == "dummy_mean", "rmse"].iloc[0]
        tied = set(models_only.loc[models_only["rmse"] - best["rmse"] <= TIE_TOLERANCE, "model"])
        tied_labels = " · ".join(MODEL_LABELS[m] for m in models_only["model"] if m in tied)
        saved_best = results.set_index("model").loc[names["best_overall"]]  # the app's model
        headline_card(
            plain.model_headline(
                saved_best["rmse"], summary["test_mean"], saved_best["r2"], baseline
            )
        )
        k1, k2, k3, k4 = st.columns(4)
        if simple:
            with k1:
                kpi_card(
                    "Typical error",
                    f"{saved_best['rmse']:.2f}",
                    "t/ha",
                    f"{PLAIN_MODEL_NAMES[names['best_overall']]}, our best model",
                    accent=True,
                    help_text=TIPS["rmse"],
                )
            with k2:
                kpi_card(
                    "Explains",
                    plain.pct(saved_best["r2"]),
                    "",
                    "of the differences between fields",
                    help_text=TIPS["r2"],
                )
            with k3:
                kpi_card(
                    "Guessing the average",
                    f"{baseline:.2f}",
                    "t/ha",
                    "Typical error of one guess for all",
                    help_text=TIPS["baseline"],
                )
            with k4:
                kpi_card(
                    "Improvement",
                    plain.pct(1 - saved_best["rmse"] / baseline),
                    "",
                    "Smaller error than guessing",
                    help_text=TIPS["improvement"],
                    help_left=True,
                )
            board_side, scatter_side = st.columns([9, 11], gap="medium")
            with board_side, st.container(key="card-board", height="stretch"):
                card_title(
                    "Leaderboard",
                    f"Typical error on {summary['n_test']} fields the models never saw",
                    help_text=TIPS["test"],
                )
                medals = ["🥇", "🥈", "🥉"]
                rows = []
                for i, row in enumerate(ranked.itertuples()):
                    css = "first" if i == 0 else "muted" if row.model == "dummy_mean" else ""
                    mark = (
                        f'<td class="medal">{medals[i]}</td>'
                        if i < len(medals)
                        else f'<td class="rank">{i + 1}</td>'
                    )
                    name = PLAIN_MODEL_NAMES[row.model]
                    rows.append(
                        f'<tr class="{css}">{mark}<td>{name}'
                        f"{tip('Technical name: ' + MODEL_LABELS[row.model])}</td>"
                        f'<td class="err">{row.rmse:.2f} t/ha</td></tr>'
                    )
                note = plain.tie_note(list(models_only["rmse"]))
                st.html(
                    f'<table class="board">{"".join(rows)}</table>'
                    + (f'<p class="card-caption">{note}</p>' if note else "")
                )
            with scatter_side, st.container(key="card-scatter", height="stretch"):
                card_title(
                    "Predicted vs real yield",
                    f"{summary['n_test']} held-out fields · on the dashed line = a perfect "
                    "prediction · hover a dot",
                )
                st.altair_chart(charts.scatter_chart(held_out_predictions(), 280), theme=None)
        else:
            with k1:
                kpi_card(
                    "Best test error (RMSE)",
                    f"{best['rmse']:.3f}",
                    "t/ha",
                    tied_labels,
                    accent=True,
                    help_text=TIPS["rmse"],
                )
            with k2:
                kpi_card(
                    "Best R²",
                    f"{best_r2['r2']:.2f}",
                    "",
                    f"{MODEL_LABELS[best_r2['model']]} on {summary['n_test']} held-out fields",
                    help_text=TIPS["r2"],
                )
            with k3:
                kpi_card(
                    "Baseline error",
                    f"{baseline:.3f}",
                    "t/ha",
                    "Always predicting the average",
                    help_text=TIPS["baseline"],
                )
            with k4:
                kpi_card(
                    "Improvement over baseline",
                    f"{1 - best['rmse'] / baseline:.0%}",
                    "",
                    "Lower test RMSE than the mean baseline",
                    help_text=TIPS["improvement"],
                    help_left=True,
                )
            chart_side, table_side = st.columns([11, 10], gap="medium")
            with chart_side, st.container(key="card-model-chart", height="stretch"):
                card_title(
                    "Held-out test error by model",
                    "RMSE in t/ha — lower is better",
                    help_text=TIPS["rmse"],
                )
                chart_data = ranked.assign(label=ranked["model"].map(MODEL_LABELS))
                st.altair_chart(
                    charts.ranked_bar_chart(
                        chart_data,
                        "label",
                        "rmse",
                        "Test RMSE (t/ha)",
                        ".3f",
                        height=300,
                        highlight={MODEL_LABELS[m] for m in tied},
                        muted={MODEL_LABELS["dummy_mean"]},
                    ),
                    theme=None,
                )
            with table_side, st.container(key="card-model-table", height="stretch"):
                card_title(
                    "Test-set results",
                    f"{summary['n_test']} held-out fields · best value per column highlighted",
                    help_text=TIPS["metrics"],
                )
                show_table(ranked[["model", "rmse", "mae", "r2"]])

with tab_drivers:
    agreement = load_table("importance_agreement")
    if agreement is None:
        st.info("importance_agreement.csv not found — run the notebooks (`python run.py build`).")
    else:
        shap_by_feature = agreement.set_index("feature")["mean_abs_shap"].sort_values(
            ascending=False
        )
        group_totals = pd.Series(
            {g: shap_by_feature[cols].sum() for g, cols in config.FEATURE_GROUPS.items()}
        ).sort_values(ascending=False)
        shares = group_totals / group_totals.sum()
        signal = load_table("feature_signal")
        rho = signal.set_index("feature")["spearman_rho"] if signal is not None else pd.Series()
        top3 = list(shap_by_feature.index[:3])
        negative = {FRIENDLY[f].lower() for f in top3 if rho.get(f, 0) < 0}
        headline_card(
            plain.drivers_headline(
                shares.index[0], shares.iloc[0], [FRIENDLY[f].lower() for f in top3], negative
            )
        )
        group_side, feature_side = st.columns([2, 3], gap="medium")
        if simple:
            with group_side, st.container(key="card-groups", height="stretch"):
                card_title(
                    "Which kinds of factors matter",
                    "Share of what the model relies on",
                    help_text=TIPS["shap"],
                )
                donut = pd.DataFrame(
                    {
                        "label": [f"{plain.GROUP_PLAIN[g]}  {s:.0%}" for g, s in shares.items()],
                        "share": shares.to_numpy(),
                    }
                )
                st.altair_chart(charts.donut_chart(donut, height=320), theme=None)
            with feature_side, st.container(key="card-response", height="stretch"):
                head, pick = st.columns([3, 2], vertical_alignment="center")
                with head:
                    card_title(
                        "How yield responds to each factor",
                        "Average prediction over 300 real fields",
                        help_text=TIPS["response"],
                    )
                with pick:
                    factor = st.selectbox(
                        "Factor",
                        config.NUMERIC_FEATURES,
                        index=config.NUMERIC_FEATURES.index(shap_by_feature.index[0]),
                        format_func=lambda f: FRIENDLY[f],
                        key="response_feature",
                        label_visibility="collapsed",
                    )
                curve = response_curve(factor)
                unit = config.UNITS.get(factor, "")
                st.altair_chart(
                    charts.curve_chart(curve, None, f"{FRIENDLY[factor]} ({unit})", 270),
                    theme=None,
                )
                st.html(
                    '<p class="plain-text">'
                    f"{plain.curve_caption(FRIENDLY[factor], unit, curve['x'], curve['y'])}</p>"
                )
        else:
            with group_side, st.container(key="card-groups"):
                card_title(
                    "Importance by factor group",
                    "Share of mean |SHAP| (XGBoost, held-out fields)",
                    help_text=TIPS["shap"],
                )
                group_data = pd.DataFrame({"group": shares.index, "share": shares.to_numpy()})
                st.altair_chart(
                    charts.ranked_bar_chart(
                        group_data,
                        "group",
                        "share",
                        "Share of importance",
                        ".0%",
                        height=330,
                        highlight={shares.index[0]},
                        axis_format="%",
                    ),
                    theme=None,
                )
            with feature_side, st.container(key="card-features"):
                head, toggle = st.columns([3, 2], vertical_alignment="center")
                with toggle:
                    view = st.segmented_control(
                        "View",
                        ["Ranking", "Detailed view"],
                        default="Ranking",
                        key="drivers_view",
                        label_visibility="collapsed",
                    )
                with head:
                    if view == "Detailed view":
                        card_title(
                            "Importance by feature",
                            "Original SHAP figure · hover to enlarge",
                            help_text=TIPS["beeswarm"],
                        )
                    else:
                        card_title(
                            "Importance by feature", "Mean |SHAP| in t/ha", help_text=TIPS["shap"]
                        )
                if view == "Detailed view":
                    beeswarm = config.FIGURES_DIR / "explain_shap_beeswarm.png"
                    if beeswarm.exists():
                        st.image(str(beeswarm), width=370)
                    else:
                        st.info("explain_shap_beeswarm.png not found — run the notebooks.")
                else:
                    feature_data = pd.DataFrame(
                        {
                            "feature": [readable(f) for f in shap_by_feature.index],
                            "shap": shap_by_feature.to_numpy(),
                        }
                    )
                    st.altair_chart(
                        charts.ranked_bar_chart(
                            feature_data,
                            "feature",
                            "shap",
                            "Mean |SHAP| (t/ha)",
                            ".3f",
                            height=330,
                            highlight={feature_data["feature"].iloc[0]},
                        ),
                        theme=None,
                    )

with tab_real:
    india_test = load_table("india_test")
    lookups = load_table("india_reference_baselines")
    shap_shares = load_table("india_shap_shares")
    per_crop = load_table("india_per_crop_rmse")
    if india_test is None or lookups is None or shap_shares is None or per_crop is None:
        st.info(
            "The real-world validation has not been run yet — place the India CSV in "
            "data/raw/india/ and run `python run.py`."
        )
    else:
        india_names = json.loads((config.MODELS_DIR / "india_model_names.json").read_text())
        india_best = india_test.set_index("model").loc[india_names["best_overall"]]
        lookup = lookups[lookups["predictor"].str.contains("state")].iloc[0]
        crop = shap_shares.set_index("feature").loc[config.INDIA_CROP, "share"]
        models_india = india_test[india_test["model"] != "dummy_mean"]
        headline_card(plain.india_headline(lookup["rmse"], india_best["rmse"], crop))
        if simple:
            lessons = [
                plain.lesson_crop(crop),
                plain.lesson_lookup(
                    lookup["rmse"], india_best["rmse"], lookup["rmse"] < models_india["rmse"].min()
                ),
                plain.lesson_seasons(int((per_crop["r2"] < 0).sum()), len(per_crop)),
            ]
            with st.container(key="lessons"):
                for i, (column, (big, text)) in enumerate(zip(st.columns(3), lessons), start=1):
                    with column:
                        st.html(
                            f'<div class="lesson"><p class="l-num"><span>{i}</span>{big}</p>'
                            f'<p class="l-text">{text}</p></div>'
                        )
            compare_side, crop_side = st.columns([1, 1], gap="medium")
            with compare_side, st.container(key="card-india-compare", height="stretch"):
                card_title(
                    "Typical error on the 2016–2020 test years",
                    "t/ha · lower is better",
                    help_text=TIPS["rmse"],
                )
                guess = india_test.set_index("model").loc["dummy_mean", "rmse"]
                best_label = f"Best model ({PLAIN_MODEL_NAMES[india_names['best_overall']]})"
                compare = pd.DataFrame(
                    {
                        "label": [
                            "Look-up: crop's usual yield in the state",
                            best_label,
                            "Guessing the average",
                        ],
                        "rmse": [lookup["rmse"], india_best["rmse"], guess],
                    }
                ).sort_values("rmse")
                st.altair_chart(
                    charts.ranked_bar_chart(
                        compare,
                        "label",
                        "rmse",
                        "Typical error (t/ha)",
                        ".2f",
                        height=250,
                        highlight={compare["label"].iloc[0]},
                        muted={"Guessing the average"},
                    ),
                    theme=None,
                )
            with crop_side, st.container(key="card-india-crops", height="stretch"):
                card_title(
                    f"How well it works for the {len(per_crop)} most common crops",
                    "Typical error as a share of each crop's average yield",
                    help_text=TIPS["relative"],
                )
                crops = per_crop.sort_values("relative_rmse")
                verdicts = pd.DataFrame(
                    {
                        "crop": crops["crop"],
                        "relative": crops["relative_rmse"],
                        "verdict": np.where(
                            crops["relative_rmse"] < plain.WORKS_WELL, "works well", "struggles"
                        ),
                    }
                )
                st.altair_chart(charts.verdict_bar_chart(verdicts, height=250), theme=None)
        else:
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                kpi_card(
                    "Best model error",
                    f"{india_best['rmse']:.2f}",
                    "RMSE",
                    f"{MODEL_LABELS[india_names['best_overall']]}, test years 2016–2020",
                    help_text=TIPS["rmse"],
                )
            with k2:
                kpi_card(
                    "Crop × state look-up error",
                    f"{lookup['rmse']:.2f}",
                    "RMSE",
                    "No ML — beats every model",
                    accent=True,
                    help_text=TIPS["lookup"],
                )
            with k3:
                kpi_card(
                    "Crop type share of importance",
                    f"{crop:.0%}",
                    "",
                    "Mean |SHAP| of the best model",
                    help_text=TIPS["shap"],
                )
            with k4:
                best_r2_model = india_test.loc[india_test["r2"].idxmax(), "model"]
                kpi_card(
                    "Best R²",
                    f"{india_test['r2'].max():.2f}",
                    "",
                    MODEL_LABELS[best_r2_model],
                    help_text=TIPS["r2"],
                    help_left=True,
                )
            with st.container(key="card-india-tables"):
                choice = (
                    st.segmented_control(
                        "Table",
                        list(INDIA_TABLES),
                        default="Test results",
                        key="india_table",
                        label_visibility="collapsed",
                    )
                    or "Test results"
                )
                table_name, caption, help_text = INDIA_TABLES[choice]
                frame = load_table(table_name)
                # The selected pill already names the table, so only its caption is shown.
                st.html(f'<p class="card-caption">{caption}{tip(help_text)}</p>')
                if frame is None:
                    st.info(f"{table_name}.csv not found — run `python run.py build`.")
                else:
                    show_table(frame)

with tab_how:
    results = load_table("test_results")
    agreement = load_table("importance_agreement")
    if results is None or agreement is None:
        st.info("Results not found — run `python run.py` first.")
    else:
        saved_best = results.set_index("model").loc[names["best_overall"]]
        top_shap = agreement.set_index("feature")["mean_abs_shap"].sort_values(ascending=False)
        n_fields = summary["n_train"] + summary["n_test"]
        headline_card(
            f"The model learned from <b>{summary['n_train']:,}</b> example fields, was tested on "
            f"<b>{summary['n_test']}</b> fields it had never seen, and every finding was "
            "re-checked on real Indian crop records."
        )
        steps = [
            (
                "Collect data",
                (
                    f"Records of fields: {len(config.FEATURES)} details about soil, weather "
                    "and farming, and the yield each field produced."
                ),
                f"{n_fields:,}",
                "fields",
            ),
            (
                f"Train {len(results)} models",
                (
                    "From 'guess the average' to boosted decision trees, each tuned with "
                    f"{config.CV_FOLDS}-fold cross-validation."
                ),
                f"{len(results)}",
                "approaches compared",
            ),
            (
                "Test on unseen fields",
                (
                    f"{summary['n_test']} fields ({config.TEST_SIZE:.0%}) were kept aside; the "
                    "predictions are compared with their real yields."
                ),
                f"{saved_best['rmse']:.2f} t/ha",
                "typical error",
            ),
            (
                "Explain the predictions",
                (
                    "SHAP splits every prediction into the push from each factor, so we can "
                    "see what the model relies on."
                ),
                f"{top_shap.iloc[0]:.2f} t/ha",
                f"average push from {FRIENDLY[top_shap.index[0]].lower()}",
            ),
        ]
        cards = []
        for i, (title, text, big, unit) in enumerate(steps, start=1):
            cards.append(
                f'<div class="step"><span class="s-no">{i}</span>'
                f'<p class="s-title">{title}</p><p class="s-text">{text}</p>'
                f'<p class="s-num">{big} <span>{unit}</span></p></div>'
            )
        st.html('<div class="flow">' + '<div class="arrow">→</div>'.join(cards) + "</div>")
        with st.container(key="card-glossary"):
            card_title("Glossary", "The technical words used in Technical mode, in plain words")
            st.html(
                '<div class="glossary">'
                + "".join(f"<p><b>{term}</b>{text}</p>" for term, text in plain.GLOSSARY.items())
                + "</div>"
            )
        st.html(
            '<p class="honest">The main benchmark is a controlled dataset; every finding is '
            "re-checked on real Indian crop records.</p>"
        )
