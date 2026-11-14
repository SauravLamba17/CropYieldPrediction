"""Every page renders without an exception, in both modes (Streamlit AppTest, headless)."""

import pytest
from streamlit.testing.v1 import AppTest

from src import config

PAGES = [
    "app_pages/overview.py",
    "app_pages/predict.py",
    "app_pages/whatif.py",
    "app_pages/models.py",
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
