"""Every page renders without an exception, in both modes (Streamlit AppTest, headless)."""

import pytest
from streamlit.testing.v1 import AppTest

from src import config

PAGES = [
    "app_pages/overview.py",
    "app_pages/predict.py",
    "app_pages/whatif.py",
    "app_pages/models.py",
    "app_pages/explain.py",
    "app_pages/explorer.py",
    "app_pages/india.py",
]
TIMEOUT = 120  # the first Predict render builds the SHAP explainer


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, at.exception
    return at


@pytest.mark.parametrize("mode", ["Simple", "Technical"])
@pytest.mark.parametrize("page", PAGES)
def test_page_renders(app, page, mode):
    app.segmented_control(key="mode").set_value(mode).run()
    app.switch_page(page).run()
    assert not app.exception, [e.value for e in app.exception]


def test_predict_shows_prediction_with_range_and_report(app):
    app.segmented_control(key="mode").set_value("Simple").run()
    app.switch_page("app_pages/predict.py").run()
    html = " ".join(str(h.proto.body) for h in app.get("html"))
    assert "90% range" in html and "t/ha" in html
    assert {b.key for b in app.get("download_button")} == {"dl_csv", "dl_pdf"}


def test_preset_changes_inputs(app):
    app.switch_page("app_pages/predict.py").run()
    before = app.session_state["in_total_rainfall"]
    app.button_group(key="preset").set_value("Drought year").run()
    assert not app.exception
    assert app.session_state["in_total_rainfall"] < before


def test_scenario_b_changes_the_gap(app):
    app.segmented_control(key="mode").set_value("Technical").run()
    app.switch_page("app_pages/whatif.py").run()
    app.button(key="ab_reset").click().run()
    app.slider(key="b_fertilizer_amount").set_value(app.session_state["in_fertilizer_amount"] + 50).run()
    assert not app.exception
    assert "Changed in B" in " ".join(str(h.proto.body) for h in app.get("html"))
    app.button(key="ab_swap").click().run()
    assert not app.exception


def test_explain_page_without_tables_shows_a_setup_card(tmp_path, monkeypatch):
    """If the explainability tables are missing, the page explains how to build them."""
    from ui import data

    at = AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()
    monkeypatch.setattr(config, "TABLES_DIR", tmp_path)  # no tables there
    data.load_table.clear()
    at.switch_page("app_pages/explain.py").run()
    data.load_table.clear()
    assert not at.exception
    assert "explain_tables" in " ".join(str(h.proto.body) for h in at.get("html"))


def test_india_page_without_phase4_tables(tmp_path, monkeypatch):
    """With only the pre-Phase-4 India tables, the page still renders and says how to build the rest."""
    import shutil

    from ui import data

    for name in ("india_test", "india_reference_baselines", "india_shap_shares", "india_per_crop_rmse"):
        shutil.copy(config.TABLES_DIR / f"{name}.csv", tmp_path)
    at = AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()
    monkeypatch.setattr(config, "TABLES_DIR", tmp_path)
    for mode in ("Technical", "Simple"):
        data.load_table.clear()
        data.load_json.clear()
        at.segmented_control(key="mode").set_value(mode).run()
        at.switch_page("app_pages/india.py").run()
        assert not at.exception, [e.value for e in at.exception]
    html = " ".join(str(h.proto.body) for h in at.get("html"))
    assert "not built yet" in html
    data.load_table.clear()
    data.load_json.clear()
