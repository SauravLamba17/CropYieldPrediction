"""Phase 9, simple by default: first visit in Simple, 5 pages in the nav (7 with details), no
technical words anywhere on the 5 Simple pages, at most 2 charts per Simple page, and Predict
showing 4 inputs with the rest in a closed "More field conditions" expander.
"""

import html
import json
import re

import pytest
import streamlit as st
from streamlit.dataframe_util import convert_arrow_bytes_to_pandas_df
from streamlit.testing.v1 import AppTest

from src import config
from ui import data

TIMEOUT = 120
SIMPLE_PAGES = {  # nav title -> script
    "Home": "app_pages/overview.py",
    "Predict": "app_pages/predict.py",
    "What-if": "app_pages/whatif.py",
    "What matters": "app_pages/explain.py",
    "Real-world check": "app_pages/india.py",
}
DETAIL_PAGES = {"Model comparison", "Data explorer"}
FORBIDDEN = {  # word -> pattern (case-insensitive)
    "RMSE": r"\bRMSE\b",
    "MAE": r"\bMAE\b",
    "R²": r"\bR²|\bR2\b",
    "CV": r"\bCV\b",
    "cross-validation": r"cross[- ]?validat",
    "SHAP": r"\bSHAP\b",
    "permutation": r"permutation",
    "conformal": r"conformal",
    "p-value": r"p-value|\bp = ",
    "lag": r"\blags?\b",
    "target encoding": r"target[- ]encod",
    "polynomial": r"polynomial",
    "Ridge": r"\bridge\b",
    "ablation": r"ablation",
    "baseline": r"baseline",
}
MAIN_INPUTS = {"in_fertilizer_amount", "in_pesticide_usage", "in_total_rainfall", "in_crop_type"}


def walk(node):
    yield node
    children = getattr(node, "children", None)
    if isinstance(children, dict):
        for child in children.values():
            yield from walk(child)


def chart_strings(spec, out: list[str]) -> None:
    """Titles and literal texts in a Vega-Lite spec (axis, legend, tooltip and mark titles)."""
    if isinstance(spec, dict):
        for key, value in spec.items():
            if key in ("title", "text") and isinstance(value, str):
                out.append(value)
            elif key == "title" and isinstance(value, list):
                out.extend(v for v in value if isinstance(v, str))
            else:
                chart_strings(value, out)
    elif isinstance(spec, list):
        for value in spec:
            chart_strings(value, out)


def rendered_text(at: AppTest) -> str:
    """Everything a visitor can read: page HTML (with the ⓘ tooltips), widget labels, help and
    options, expander labels, and each chart's titles and data labels."""
    parts: list[str] = []
    for node in walk(at._tree):
        kind = getattr(node, "type", "")
        proto = getattr(node, "proto", None)
        if kind == "html":
            body = re.sub(r"<(style|script)\b.*?</\1>", " ", node.proto.body, flags=re.DOTALL)  # not visible
            parts += [html.unescape(t) for t in re.findall(r'data-tip="([^"]*)"', body)]
            parts.append(html.unescape(re.sub(r"<[^>]+>", " ", body)))
        elif kind == "vega_lite_chart":
            spec = json.loads(proto.spec)
            chart_strings(spec, parts)
            shown = set(re.findall(r'"field": "([^"]+)"', proto.spec))  # columns the chart draws
            for dataset in proto.datasets:
                frame = convert_arrow_bytes_to_pandas_df(dataset.data.data)
                for col in frame.select_dtypes(exclude="number"):
                    if col in shown:
                        parts += [str(v) for v in frame[col].unique()]
        elif proto is not None:
            for field in ("label", "help", "body", "placeholder"):
                value = getattr(proto, field, None)
                if isinstance(value, str):
                    parts.append(value)
            options = getattr(proto, "options", None)
            if options is not None and not isinstance(options, str):
                parts += [str(getattr(o, "content", o)) for o in options]  # already formatted
    return "\n".join(parts)


def jargon(text: str) -> dict[str, list[str]]:
    hits = {}
    for word, pattern in FORBIDDEN.items():
        found = [text[max(m.start() - 40, 0) : m.end() + 40].replace("\n", " ")
                 for m in re.finditer(pattern, text, flags=re.IGNORECASE)]
        if found:
            hits[word] = found[:3]
    return hits


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, at.exception
    return at


def test_simple_is_the_default_on_a_first_visit():
    at = AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()
    switch = at.segmented_control(key="mode")
    assert switch.value == data.SIMPLE == "Simple"
    assert list(switch.options) == ["Simple", "Show details"]
    assert data.TECHNICAL == "Show details"


def test_the_mode_choice_is_kept_across_pages(app):
    app.segmented_control(key="mode").set_value("Show details").run()
    app.switch_page("app_pages/whatif.py").run()
    assert app.segmented_control(key="mode").value == "Show details"
    app.segmented_control(key="mode").set_value("Simple").run()
    app.switch_page("app_pages/india.py").run()
    assert app.segmented_control(key="mode").value == "Simple"


def test_nav_shows_5_pages_in_simple_and_7_with_details(monkeypatch):
    seen = []
    real = st.navigation

    def spy(pages, **kwargs):
        seen.append({p.title: p.visibility for p in pages})
        return real(pages, **kwargs)

    monkeypatch.setattr(st, "navigation", spy)
    at = AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()
    visible = [t for t, v in seen[-1].items() if v == "visible"]
    assert visible == list(SIMPLE_PAGES)
    assert {t for t, v in seen[-1].items() if v == "hidden"} == DETAIL_PAGES  # registered: URLs work
    at.segmented_control(key="mode").set_value("Show details").run()
    assert len(seen[-1]) == 7 and all(v == "visible" for v in seen[-1].values())


def test_hidden_pages_still_open_by_url_in_simple_mode(app):
    app.segmented_control(key="mode").set_value("Simple").run()
    for page in ("app_pages/models.py", "app_pages/explorer.py"):
        app.switch_page(page).run()
        assert not app.exception, [e.value for e in app.exception]


@pytest.mark.parametrize("title", list(SIMPLE_PAGES))
def test_simple_pages_have_no_jargon_and_at_most_2_charts(app, title):
    app.segmented_control(key="mode").set_value("Simple").run()
    app.switch_page(SIMPLE_PAGES[title]).run()
    assert not app.exception, [e.value for e in app.exception]
    assert jargon(rendered_text(app)) == {}
    charts = [n for n in walk(app._tree) if getattr(n, "type", "") == "vega_lite_chart"]
    assert len(charts) <= 2, f"{title}: {len(charts)} charts in Simple mode"
    headlines = [n for n in walk(app._tree)
                 if getattr(n, "type", "") == "html" and 'class="chart-headline"' in n.proto.body]
    assert len(headlines) >= len(charts), f"{title}: a chart without its one-line headline"


def test_the_scan_catches_jargon(app):
    """The scan itself works: the details view of What matters is full of technical words."""
    app.segmented_control(key="mode").set_value("Show details").run()
    app.switch_page("app_pages/explain.py").run()
    hits = jargon(rendered_text(app))
    assert {"SHAP", "permutation"} <= set(hits)
    app.segmented_control(key="mode").set_value("Simple").run()


def test_predict_shows_4_inputs_and_the_rest_in_a_closed_expander(app):
    app.segmented_control(key="mode").set_value("Simple").run()
    app.switch_page("app_pages/predict.py").run()
    more = [e for e in app.status if e.label == "More field conditions"]
    assert len(more) == 1 and more[0].proto.expanded is False
    inside = {w.key for w in list(more[0].slider) + list(more[0].selectbox)}
    every = {w.key for w in list(app.slider) + list(app.selectbox) if w.key and w.key.startswith("in_")}
    assert every == {f"in_{f}" for f in config.FEATURES}
    assert every - inside == MAIN_INPUTS
    headline = " ".join(str(h.proto.body) for h in app.get("html"))
    assert "This field should produce about" in headline and "tonnes per hectare" in headline


def test_hidden_inputs_keep_their_values(app):
    app.segmented_control(key="mode").set_value("Show details").run()
    app.switch_page("app_pages/predict.py").run()
    app.slider(key="in_soil_ph").set_value(7.2).run()
    app.segmented_control(key="mode").set_value("Simple").run()
    assert app.session_state["in_soil_ph"] == pytest.approx(7.2)
    app.button(key="reset").click().run()


# --- the plain sentences behind the chart headlines (no app needed) -------------------


def test_simple_tips_have_no_jargon():
    from ui import plain

    for key, text in plain.SIMPLE_TIPS.items():
        assert jargon(text) == {}, key
        assert plain.tip_text(key, True) == text and plain.tip_text(key, False) == plain.TIPS[key]
    assert plain.tip_text("interval", True) == plain.TIPS["interval"]  # no simple version: the same


def test_chart_headlines_follow_the_numbers():
    import numpy as np

    from ui import plain

    x = np.linspace(0, 10, 40)
    assert plain.curve_headline("Fertilizer", "kg/ha", x, 5 + 0.3 * x) == "More fertilizer, higher predicted yield"
    peak = np.linspace(0, 10, 41)  # a grid point at exactly 5
    assert plain.curve_headline("Soil pH", "pH", peak, 6 - (peak - 5) ** 2 / 10) == "Soil pH works best around pH 5.0"
    assert plain.curve_headline("Sunshine", "hours", x, np.full(40, 6.0)) == "Sunshine barely changes the prediction"
    assert plain.importance_headline([("Fertilizer", 0.73), ("Pesticide use", 0.12)]) == "Fertilizer matters most, by far"
    assert plain.importance_headline([("Crop", 0.30), ("Rainfall", 0.20)]) == "Crop matters most, then rainfall"
    assert plain.contribution_headline({"Fertilizer": 0.10, "Pesticide use": -0.02}) == (
        "Fertilizer helps this field most (+0.10 t/ha)")
    assert plain.contribution_headline({"Rainfall": -0.40, "Crop": 0.10}) == (
        "Rainfall holds this field back most (−0.40 t/ha)")
    assert plain.gap_headline(0.001) == "B and A should yield about the same"
    assert plain.gap_headline(-0.31) == "B should yield 0.31 t/ha less than A"
    assert plain.lowest_error_headline({"the chosen model": 6.85, "repeating last year's yield": 6.95,
                                        "the look-up": 7.81}) == (
        "The chosen model makes the smallest errors, with repeating last year's yield close behind")
    assert plain.crops_headline(4, 10) == "The model works well for 4 of the 10 most common crops"
    assert plain.gauge_headline("Good", 0.567) == "A good yield: higher than 56% of the fields the model learned from"


def test_result_sentence_is_one_sentence():
    from ui import plain

    text = plain.result_sentence(6.5212, 5.4845, 7.5579)
    assert text == ("This field should produce about <b>6.5 tonnes per hectare</b> "
                    "(usually between <b>5.5</b> and <b>7.6</b>).")
    assert plain.result_sentence(6.52, None, None) == "This field should produce about <b>6.5 tonnes per hectare</b>."
