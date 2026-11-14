"""Everything the pages read: saved models, results tables, data, and the shared input state.

No model is trained here. Models are cached with st.cache_resource, tables and data with
st.cache_data, so switching pages or moving a slider never reloads a file.
"""

import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import RegressorMixin

from src import config
from src.data import load_split

APP_VERSION = "2.0.0"
SIMPLE, TECHNICAL = "Simple", "Technical"
PRESETS = ["Typical field", "Ideal field", "Drought year", "Low-input farm"]
CURVE_POINTS = 40  # what-if and response curves sweep each factor over 40 values
CURVE_SAMPLE = 300  # response curves average the prediction over 300 training fields
MODEL_FILES = {"best_overall": "best_overall.joblib", "best_tree": "best_tree.joblib"}
REQUIRED = [
    config.MODELS_DIR / "best_overall.joblib",
    config.MODELS_DIR / "model_names.json",
    config.APP_METADATA_JSON,
    config.KAGGLE_TRAIN_PARQUET,
    config.KAGGLE_TEST_PARQUET,
]
# Widget values that must survive page switches (Streamlit forgets the state of widgets
# that are not drawn on the current page unless it is re-assigned every run).
PERSISTENT = [f"in_{f}" for f in config.FEATURES] + [
    "preset",
    "explain_view",
    "whatif_feature",
    "response_feature",
    "drivers_view",
    "india_table",
    "board_model",
    "explore_feature",
    "corr_method",
    "explain_model",
    "coef_model",
    "pdp_feature",
    "heat_x",
    "heat_y",
] + [f"b_{f}" for f in config.FEATURES]


WIDGET_DEFAULTS = {
    "explain_view": "Why this prediction",
    "drivers_view": "Ranking",
    "india_table": "Test results",
    "board_model": "best_overall",
    "corr_method": "Pearson",
    "whatif_feature": config.NUMERIC_FEATURES[0],
    "explore_feature": config.NUMERIC_FEATURES[0],
    "explain_model": "best_overall",
    "coef_model": "multiple_lr",
    "pdp_feature": "soil_ph",
    "heat_x": "fertilizer_amount",
    "heat_y": "total_rainfall",
}


def is_simple() -> bool:
    return st.session_state.get("mode", SIMPLE) != TECHNICAL


def missing_files() -> list[str]:
    return [str(p.relative_to(config.ROOT)) for p in REQUIRED if not p.exists()]


# --- models -----------------------------------------------------------------------


@st.cache_resource
def load_model(key: str = "best_overall") -> RegressorMixin:
    import joblib  # imported here: pages that show only tables never pay for it

    return joblib.load(config.MODELS_DIR / MODEL_FILES[key])


@st.cache_data
def model_names() -> dict:
    return json.loads((config.MODELS_DIR / "model_names.json").read_text())


@st.cache_data
def app_metadata() -> dict:
    return json.loads(config.APP_METADATA_JSON.read_text())


@st.cache_resource
def explainer():
    """Kernel SHAP in the original feature space for the deployed model (lazy: imports shap)."""
    from src.explain import RawFeatureExplainer

    return RawFeatureExplainer(load_model(), split()[0][config.FEATURES])


@st.cache_data
def tree_shap_frame() -> pd.DataFrame:
    """Per-field SHAP values of the saved XGBoost model on the held-out fields (long format)."""
    from src.explain import tree_shap

    test = split()[1]
    _, _, by_feature = tree_shap(load_model("best_tree"), test[config.FEATURES], config.FEATURES)
    long = by_feature.reset_index(names="row").melt(
        id_vars="row", var_name="feature", value_name="shap"
    )
    values = test[config.NUMERIC_FEATURES].rank(pct=True).reset_index(names="row")
    values = values.melt(id_vars="row", var_name="feature", value_name="value_rank")
    return long.merge(values, on=["row", "feature"], how="left")


# --- tables and data ----------------------------------------------------------------


@st.cache_data
def split() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_split()


@st.cache_data
def load_table(name: str) -> pd.DataFrame | None:
    """A saved results table, or None if it has not been generated."""
    path = config.TABLES_DIR / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_json(name: str) -> dict | None:
    path = config.TABLES_DIR / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def data_summary() -> dict:
    """Numbers from data/processed/ behind the plain-language sentences and presets."""
    train, test = split()
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
def held_out_predictions(key: str = "best_overall") -> pd.DataFrame:
    """A saved model's predictions for the 960 held-out test fields."""
    test = split()[1]
    predicted = load_model(key).predict(test[config.FEATURES])
    return pd.DataFrame(
        {
            "real": test[config.TARGET].to_numpy(),
            "predicted": predicted,
            "residual": test[config.TARGET].to_numpy() - predicted,
            "crop": test["crop_type"].to_numpy(),
            "region": test["region"].to_numpy(),
        }
    )


@st.cache_data
def raw_harvest_months() -> pd.DataFrame | None:
    """Harvest month and yield of the training fields (needs the git-ignored raw CSV)."""
    if not config.KAGGLE_TRAIN_CSV.exists():
        return None
    train = split()[0]
    raw = pd.read_csv(config.KAGGLE_TRAIN_CSV).loc[train.index]
    month = pd.to_datetime(raw[config.HARVEST_DATE_COLUMN], errors="coerce").dt.month
    return pd.DataFrame({"month": month, "yield": train[config.TARGET]}).dropna()


# --- curves -------------------------------------------------------------------------


def sweep_grid(feature: str) -> np.ndarray:
    """40 evenly spaced values from the training min to max (every whole number if integer)."""
    m = app_metadata()["numeric"][feature]
    if m["is_integer"]:
        return np.arange(int(m["min"]), int(m["max"]) + 1)
    return np.linspace(m["min"], m["max"], CURVE_POINTS)


def _predict_sweep(rows: pd.DataFrame, feature: str, grid: np.ndarray) -> np.ndarray:
    repeated = rows.loc[rows.index.repeat(len(grid))].copy()
    repeated[feature] = np.tile(grid, len(rows))
    predictions = load_model().predict(repeated)
    return predictions.reshape(len(rows), len(grid)).mean(axis=0)


@st.cache_data
def response_curve(feature: str) -> pd.DataFrame:
    """Partial dependence: the average prediction over 300 training fields as `feature` sweeps."""
    sample = split()[0][config.FEATURES].sample(CURVE_SAMPLE, random_state=config.RANDOM_STATE)
    grid = sweep_grid(feature)
    return pd.DataFrame({"x": grid, "y": _predict_sweep(sample, feature, grid)})


@st.cache_data
def whatif_curve(feature: str, inputs: tuple) -> pd.DataFrame:
    """The prediction for this one field as `feature` sweeps; other inputs held as set."""
    row = pd.DataFrame([dict(zip(config.FEATURES, inputs))])
    grid = sweep_grid(feature)
    return pd.DataFrame({"x": grid, "y": _predict_sweep(row, feature, grid)})


# --- inputs, presets, state ---------------------------------------------------------


def default_inputs() -> dict:
    """Training medians for numeric inputs and the first level for categorical ones."""
    metadata = app_metadata()
    values = {}
    for col in config.NUMERIC_FEATURES:
        m = metadata["numeric"][col]
        values[col] = int(m["median"]) if m["is_integer"] else m["median"]
    for col in config.CATEGORICAL_FEATURES:
        values[col] = metadata["categorical"][col][0]
    return values


def preset_values(name: str) -> dict:
    """Input values for a scenario preset, derived from the training data.

    Crop, region and season stay at the Typical defaults, so presets differ only in growing
    conditions and farming inputs. Ideal field: every numeric input where its response
    curve peaks. Drought year: rainfall and soil moisture at their training 10th percentile.
    Low-input farm: fertilizer, pesticide and irrigation at their 10th percentile.
    """
    values = default_inputs()
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
    metadata = app_metadata()
    for col in config.NUMERIC_FEATURES:
        is_integer = metadata["numeric"][col]["is_integer"]
        values[col] = round(float(values[col])) if is_integer else float(values[col])
    return values


def init_state() -> None:
    """Initialise inputs once per session and keep page-specific widget values alive."""
    for feature, default in default_inputs().items():
        st.session_state.setdefault(f"in_{feature}", default)
    st.session_state.setdefault("preset", PRESETS[0])
    for key, value in WIDGET_DEFAULTS.items():  # widget defaults live here, not on the widgets,
        st.session_state.setdefault(key, value)  # because their state is re-assigned below
    for key in PERSISTENT:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def set_inputs(values: dict) -> None:
    for col, value in values.items():
        st.session_state[f"in_{col}"] = value


def reset_inputs() -> None:
    """Button callback: put every input back to its default."""
    set_inputs(default_inputs())
    st.session_state["preset"] = PRESETS[0]


def apply_preset() -> None:
    """Pill callback: load the chosen scenario into the inputs."""
    if st.session_state["preset"]:
        set_inputs(preset_values(st.session_state["preset"]))


def clear_preset() -> None:
    """Moving an input by hand means no preset describes the field any more."""
    st.session_state["preset"] = None


def current_inputs() -> pd.DataFrame:
    return pd.DataFrame([{f: st.session_state[f"in_{f}"] for f in config.FEATURES}])


def interval() -> dict | None:
    """The saved 90% conformal interval of the deployed model (results/tables/conformal.json)."""
    return load_json("conformal")


@st.cache_data
def app_commit() -> str:
    """Short git commit of the running code, or 'unknown' outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


@st.cache_data(max_entries=256)
def prediction(inputs: tuple) -> float:
    return float(load_model().predict(pd.DataFrame([dict(zip(config.FEATURES, inputs))]))[0])


@st.cache_data(max_entries=256)
def explanation(inputs: tuple) -> tuple[float, dict[str, float]]:
    """(average prediction, {feature: SHAP push in t/ha}) for one field, cached per input set."""
    result = explainer().explain(pd.DataFrame([dict(zip(config.FEATURES, inputs))]))[0]
    return float(result.base_values), dict(zip(result.feature_names, map(float, result.values)))


def input_tuple() -> tuple:
    return tuple(st.session_state[f"in_{f}"] for f in config.FEATURES)


# --- Phase 3: heatmap, PDP/ICE, scenario B -------------------------------------------


@st.cache_data(max_entries=64)
def heatmap(fx: str, fy: str, inputs: tuple) -> pd.DataFrame:
    """25 × 25 predicted yields over two inputs' observed ranges (cached per inputs)."""
    from ui.analysis import heatmap_grid

    return heatmap_grid(
        load_model(), dict(zip(config.FEATURES, inputs)), fx, fy, app_metadata()["numeric"]
    )


@st.cache_data
def ice(feature: str) -> pd.DataFrame:
    """50 ICE lines (training fields drawn with the fixed seed) for the deployed model."""
    from ui.analysis import ICE_LINES, ice_frame

    rows = split()[0][config.FEATURES].sample(ICE_LINES, random_state=config.RANDOM_STATE)
    return ice_frame(load_model(), rows.reset_index(drop=True), feature, sweep_grid(feature))


def init_scenario_b() -> None:
    """Scenario B starts as a copy of the current scenario (A)."""
    for f in config.FEATURES:
        st.session_state.setdefault(f"b_{f}", st.session_state[f"in_{f}"])


def b_tuple() -> tuple:
    return tuple(st.session_state[f"b_{f}"] for f in config.FEATURES)


def reset_b_to_a() -> None:
    for f in config.FEATURES:
        st.session_state[f"b_{f}"] = st.session_state[f"in_{f}"]


def swap_a_b() -> None:
    for f in config.FEATURES:
        a, b = st.session_state[f"in_{f}"], st.session_state[f"b_{f}"]
        st.session_state[f"in_{f}"], st.session_state[f"b_{f}"] = b, a
    st.session_state["preset"] = None
