"""Streamlit dashboard: predict crop yield, see what drives it, and check it on real data.

Run from the project root: python run.py app   (or: python -m streamlit run app.py)

This file is the entry point: theme, header, the Simple / Technical switch (global and kept
across pages) and the page navigation. Each page is a script in app_pages/. No model is
trained here; every number comes from the saved model, results/tables/ or data/processed/.
"""

import streamlit as st

from ui import data
from ui.components import setup_card
from ui.theme import inject_theme

st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾", layout="wide")
inject_theme()

PAGES = [
    st.Page("app_pages/overview.py", title="Overview", icon=":material/home:", default=True),
    st.Page("app_pages/predict.py", title="Predict", icon=":material/agriculture:"),
    st.Page("app_pages/whatif.py", title="What-if", icon=":material/tune:"),
    st.Page("app_pages/models.py", title="Model comparison", icon=":material/leaderboard:"),
    st.Page("app_pages/explain.py", title="Explain", icon=":material/psychology:"),
    st.Page("app_pages/explorer.py", title="Data explorer", icon=":material/query_stats:"),
    st.Page("app_pages/india.py", title="India", icon=":material/public:"),
]
page = st.navigation(PAGES, position="top")

head, switch = st.columns([3, 1], vertical_alignment="center")
with head:
    st.html(
        '<p class="app-title"><span class="leaf">🌾</span> Crop Yield Prediction</p>'
        '<p class="app-subtitle">Predict crop yield, see what drives it, and check it on real '
        "data.</p>"
    )
with switch, st.container(horizontal=True, horizontal_alignment="right"):
    st.segmented_control(
        "Mode",
        [data.SIMPLE, data.TECHNICAL],
        default=data.SIMPLE,
        required=True,
        key="mode",
        label_visibility="collapsed",
        help="Simple: plain words and everyday names. Technical: exact metrics and tables.",
    )

missing = data.missing_files()
if missing:
    setup_card(
        "Setup needed: the trained models are missing",
        "Run this once from the project folder (or double-click <code>start.bat</code> on "
        "Windows). It trains the models in about 26 minutes and then opens this dashboard. "
        "Missing: " + ", ".join(f"<code>{m}</code>" for m in missing),
        command="python run.py",
    )
    st.stop()

absent = data.missing_tables()
if absent:  # health check: the pages still run, each hiding what it cannot show
    setup_card(
        f"Health check: {len(absent)} results table{'s are' if len(absent) > 1 else ' is'} missing",
        "The parts of the app that read them are hidden until they are rebuilt. Missing: "
        + ", ".join(f"<code>{m}</code>" for m in absent)
        + ". Rebuild with: "
        + " and ".join(f"<code>{c}</code>" for c in data.rebuild_commands(absent)),
    )

data.init_state()
page.run()
