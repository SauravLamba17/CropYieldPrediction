"""'Field & Harvest' design tokens: the single source of truth for CSS, charts and figures.

Both modes live in TOKENS. Nothing here depends on which mode is active:
  * CSS uses `light-dark(light, dark)`, which the browser resolves from the
    color-scheme Streamlit sets on .stApp, so a live theme switch restyles instantly
    (Streamlit does not rerun the script when the viewer changes theme).
  * Charts colour marks by *series name* (SERIES); .streamlit/config.toml maps the same
    series, in the same order, to `chartCategoricalColors` for each mode
    (tests/test_theme.py keeps config.toml and TOKENS in sync).
  * matplotlib figures (the PDF report) use mpl_style(mode).
"""

from html import escape

import streamlit as st
from cycler import cycler

TOKENS = {
    "light": {
        "bg": "#F6F4EC",
        "surface": "#FFFFFF",
        "surface_2": "#EFEBDD",  # widget fills, table headers
        "text": "#1F2A1F",
        "muted": "#5B6656",
        "border": "#DAD6C8",
        "green": "#2E7D32",  # primary
        "gold": "#E0A526",  # accent fills behind dark text
        "gold_chart": "#A87A12",  # gold as a chart mark (3:1 on white)
        "gold_text": "#8A5A00",  # gold as text
        "soil": "#8D6E63",
        "soil_text": "#6D4C41",
        "sky": "#3B82C4",
        "sky_text": "#2563A8",
        "shap_pos": "#1B7F5B",
        "shap_neg": "#C2711C",  # #D9822B was 2.9:1 on white; darkened to pass 3:1
        "neutral": "#7A8475",
        "faint": "#C9C6B8",
        "on_primary": "#FFFFFF",
        "hero_a": "#2E7D32",
        "hero_b": "#1B5E20",
    },
    "dark": {
        "bg": "#121611",
        "surface": "#1B211A",
        "surface_2": "#232B22",
        "text": "#E8EDE4",
        "muted": "#A9B4A4",
        "border": "#33402F",
        "green": "#66BB6A",
        "gold": "#F2C14E",
        "gold_chart": "#F2C14E",
        "gold_text": "#F2C14E",
        "soil": "#BCAAA4",
        "soil_text": "#BCAAA4",
        "sky": "#7FB3E6",
        "sky_text": "#7FB3E6",
        "shap_pos": "#4CC79A",  # #1B7F5B is too dark on #1B211A; brighter twin
        "shap_neg": "#F0A04B",
        "neutral": "#8E9A89",
        "faint": "#3A4538",
        "on_primary": "#121611",  # white on #66BB6A is only 2.4:1
        "hero_a": "#2E7D32",
        "hero_b": "#14361A",
    },
}

# Chart colours are looked up by series name; config.toml lists them in this order per mode.
SERIES = [
    "green",
    "sky",
    "soil",
    "gold_chart",
    "shap_pos",
    "shap_neg",
    "neutral",
    "faint",
    "text",
]
SEQUENTIAL = {  # chartSequentialColors (10 steps) for heatmaps and quantitative colour
    "light": [
        "#F1F5EC", "#DCEBD3", "#C3DDB7", "#A6CD98", "#87BC78",
        "#68A85B", "#4C9244", "#357B33", "#236427", "#164D1C",
    ],
    "dark": [
        "#1E2A1D", "#25391F", "#2D4A25", "#365C2C", "#416F34",
        "#4E833E", "#5E984B", "#73AD5D", "#8DC274", "#AAD78F",
    ],
}
GROUP_SERIES = {"Soil": "soil", "Weather": "sky", "Farming": "green", "Context": "gold_chart"}
GROUP_TEXT = {"Soil": "soil_text", "Weather": "sky_text", "Farming": "green", "Context": "gold_text"}

# Text/background pairs that must pass WCAG AA (4.5:1 normal text, 3:1 large text or marks).
CONTRAST_PAIRS = [
    ("text", "bg", 4.5),
    ("text", "surface", 4.5),
    ("muted", "bg", 4.5),
    ("muted", "surface", 4.5),
    ("green", "bg", 4.5),
    ("green", "surface", 4.5),
    ("gold_text", "surface", 4.5),
    ("soil_text", "surface", 4.5),
    ("sky_text", "surface", 4.5),
    ("on_primary", "green", 4.5),
    ("text", "surface_2", 4.5),
    ("shap_pos", "surface", 3.0),
    ("shap_neg", "surface", 3.0),
    ("gold_chart", "surface", 3.0),
    ("soil", "surface", 3.0),
    ("sky", "surface", 3.0),
    ("neutral", "surface", 3.0),
]
ON_HERO = "#FFFFFF"  # hero cards are a dark green gradient in both modes


def luminance(hex_colour: str) -> float:
    """WCAG relative luminance of a #RRGGBB colour."""
    h = hex_colour.lstrip("#")
    channels = [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(a: str, b: str) -> float:
    """WCAG contrast ratio between two colours (1 to 21)."""
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def contrast_report() -> list[dict]:
    """Every CONTRAST_PAIRS check, in both modes, plus white on the hero gradient."""
    rows = []
    for mode, t in TOKENS.items():
        for fg, bg, need in CONTRAST_PAIRS:
            ratio = contrast(t[fg], t[bg])
            rows.append({"mode": mode, "pair": f"{fg} on {bg}", "ratio": ratio, "need": need})
        for stop in ("hero_a", "hero_b"):
            ratio = contrast(ON_HERO, t[stop])
            rows.append({"mode": mode, "pair": f"white on {stop}", "ratio": ratio, "need": 4.5})
    return rows


def series_colours(mode: str) -> list[str]:
    """chartCategoricalColors for one mode, in SERIES order."""
    return [TOKENS[mode][name] for name in SERIES]


def mpl_style(mode: str = "light") -> dict:
    """matplotlib rcParams from the tokens (the PDF report uses the light set: it is paper)."""
    t = TOKENS[mode]
    return {
        "figure.facecolor": t["surface"],
        "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"],
        "text.color": t["text"],
        "axes.labelcolor": t["text"],
        "axes.edgecolor": t["border"],
        "xtick.color": t["muted"],
        "ytick.color": t["muted"],
        "axes.prop_cycle": cycler(color=series_colours(mode)[:8]),
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "DejaVu Sans", "Arial"],
    }


def _var(name: str) -> str:
    return f"light-dark({TOKENS['light'][name]}, {TOKENS['dark'][name]})"


def _variables() -> str:
    names = sorted(TOKENS["light"])
    return "\n".join(f"    --cy-{n.replace('_', '-')}: {_var(n)};" for n in names)


CARD_KEY_PREFIX = "card-"  # every card is a st.container(key="card-...")
SELECTOR_KEYS = ["india_table", "drivers_view", "explain_view", "preset", "mode", "board_model"]
DEFAULT_LIGHT_FLAG = "cyp-default-light-v1"  # localStorage marker: default applied once


def _sel(inner: str) -> str:
    return ", ".join(f".st-key-{k} {inner}" for k in SELECTOR_KEYS)


CSS = f"""
<style>
.stApp {{
{_variables()}
    --cy-shadow-c: light-dark(rgba(31,42,31,.07), rgba(0,0,0,.45));
    --cy-shadow: 0 1px 2px var(--cy-shadow-c), 0 8px 24px var(--cy-shadow-c);
    --sp-1: 4px; --sp-2: 8px; --sp-3: 12px; --sp-4: 16px; --sp-5: 24px; --sp-6: 32px;
    --font-num: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
    --font-head: 'Manrope', 'Inter', system-ui, sans-serif;
}}
/* --- Chrome: no footer; the header keeps the page navigation and the theme menu. The sidebar
   is never hidden: on wide screens Streamlit does not draw one (the navigation is on top), and
   on narrow screens it moves the navigation into it, behind the "»" button. --- */
footer, [data-testid="stDecoration"] {{ display: none !important; }}
[data-testid="stMainBlockContainer"] {{
    padding: 4.25rem var(--sp-6) var(--sp-4) var(--sp-6) !important; max-width: 1600px;
}}
[data-testid="stMainBlockContainer"] > div > [data-testid="stVerticalBlock"] {{ gap: var(--sp-3); }}

/* --- App header (title + mode switch) --- */
.app-title {{ font-family: var(--font-head); font-size: 26px; font-weight: 800; color: var(--cy-text);
    margin: 0; line-height: 1.2; white-space: nowrap; }}
.app-title .leaf {{ color: var(--cy-green); }}
.app-subtitle {{ font-size: 15px; color: var(--cy-muted); margin: 2px 0 0 0; }}
.page-title {{ font-family: var(--font-head); font-size: 20px; font-weight: 700; color: var(--cy-text);
    margin: 0; }}

/* --- Cards --- */
[class*="st-key-{CARD_KEY_PREFIX}"] {{
    background: var(--cy-surface); border: 1px solid var(--cy-border); border-radius: 16px;
    box-shadow: var(--cy-shadow); padding: var(--sp-4) var(--sp-5) var(--sp-3) var(--sp-5);
    gap: var(--sp-2);
}}
.card-title {{ font-family: var(--font-head); font-size: 16px; font-weight: 700; color: var(--cy-text);
    margin: 0; line-height: 1.4; }}
.card-caption {{ font-size: 14px; color: var(--cy-muted); margin: 0; line-height: 1.45; }}
.card-title .inline-caption {{ font-family: Inter, sans-serif; font-size: 14px; font-weight: 400;
    color: var(--cy-muted); margin-left: 10px; }}
.eyebrow {{ font-size: 13px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;
    color: var(--cy-muted); margin: 0 0 2px 0; }}
.eyebrow.soil {{ color: var(--cy-soil-text); }} .eyebrow.sky {{ color: var(--cy-sky-text); }}
.eyebrow.green {{ color: var(--cy-green); }} .eyebrow.gold {{ color: var(--cy-gold-text); }}
.eyebrow .dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%;
    margin-right: 6px; vertical-align: 1px; background: currentColor; }}

/* --- Headline ("In simple words") and setup cards --- */
.headline, .setup {{
    background: var(--cy-surface); border: 1px solid var(--cy-border); border-radius: 16px;
    box-shadow: var(--cy-shadow); padding: var(--sp-3) var(--sp-5);
}}
.headline .h-text {{ font-size: 16px; color: var(--cy-text); margin: 0; line-height: 1.5; }}
.headline b, .plain-text b, .lesson b {{ color: var(--cy-green); font-weight: 700; }}
.plain-text {{ font-size: 15px; color: var(--cy-text); margin: 0; line-height: 1.5; }}
.setup {{ border-left: 6px solid var(--cy-gold); }}
.setup .s-title {{ font-family: var(--font-head); font-size: 17px; font-weight: 700;
    color: var(--cy-text); margin: 0 0 var(--sp-1) 0; }}
.setup .s-body {{ font-size: 15px; color: var(--cy-text); margin: 0 0 var(--sp-2) 0; line-height: 1.5; }}
.setup code, .cmd {{ font-family: var(--font-num); font-size: 14px; background: var(--cy-surface-2);
    color: var(--cy-text); border-radius: 6px; padding: 1px 6px; }}
.setup .cmd {{ display: inline-block; padding: var(--sp-2) var(--sp-3); margin-top: var(--sp-1); }}
.setup a {{ color: var(--cy-sky-text); }}

/* --- KPI grid: 4 per row, 2 below 1200 px, 1 below 600 px; values never wrap --- */
.kpis {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: var(--sp-3); }}
@media (max-width: 1199.98px) {{ .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
@media (max-width: 599.98px) {{ .kpis {{ grid-template-columns: 1fr; }} }}
.kpi {{
    background: var(--cy-surface); border: 1px solid var(--cy-border); border-radius: 16px;
    box-shadow: var(--cy-shadow); padding: var(--sp-3) var(--sp-5); min-width: 0;
}}
.kpi .k-label {{ font-size: 14px; font-weight: 600; color: var(--cy-muted); margin: 0; }}
.kpi .k-value {{ font-family: var(--font-num); font-size: 30px; font-weight: 600; color: var(--cy-text);
    margin: 2px 0 0 0; line-height: 1.2; white-space: nowrap; font-variant-numeric: tabular-nums; }}
.kpi .k-value span {{ font-family: Inter, sans-serif; font-size: 16px; font-weight: 600;
    color: var(--cy-muted); }}
.kpi .k-note {{ font-size: 14px; color: var(--cy-muted); margin: 2px 0 0 0; line-height: 1.4; }}
.kpi.accent {{ border-top: 4px solid var(--cy-green); }}
.kpi.accent .k-value {{ color: var(--cy-green); }}

/* --- Hero prediction (dark green gradient in both modes: white text passes AA) --- */
.st-key-hero-gauge, .hero {{
    background: linear-gradient(135deg, var(--cy-hero-a) 0%, var(--cy-hero-b) 100%);
    border-radius: 16px; box-shadow: var(--cy-shadow); color: {ON_HERO};
}}
.st-key-hero-gauge {{ padding: var(--sp-3) var(--sp-5); gap: var(--sp-1); }}
.hero {{ padding: var(--sp-4) var(--sp-5); height: 100%; box-sizing: border-box; }}
.st-key-hero-box, .st-key-hero-box [data-testid="stElementContainer"],
.st-key-hero-box [data-testid="stHtml"] {{ height: 100%; }}
.gauge-top, .hero .label {{ font-size: 15px; font-weight: 600; color: {ON_HERO}; opacity: .92; margin: 0; }}
.gauge-top .tip, .hero .tip {{ color: rgba(255,255,255,.85); }}
.gauge-value {{ text-align: center; color: {ON_HERO}; margin: 0; line-height: 1.1; }}
.gauge-value .num, .hero .value {{ font-family: var(--font-num); font-size: 44px; font-weight: 600;
    white-space: nowrap; }}
.gauge-value .unit, .hero .value span {{ font-size: 18px; font-weight: 600; opacity: .9; }}
.range-line {{ text-align: center; color: {ON_HERO}; font-size: 15px; margin: 2px 0 0 0; }}
.range-line b {{ font-family: var(--font-num); font-weight: 600; white-space: nowrap; }}
.gauge-rating {{ text-align: center; color: {ON_HERO}; font-size: 15px; margin: 2px 0 0 0; }}
.gauge-rating b {{ background: rgba(255,255,255,.95); color: #1B5E20; border-radius: 999px;
    padding: 2px 12px; font-weight: 700; }}
.hero .value {{ margin: 6px 0 2px 0; line-height: 1.1; color: {ON_HERO}; }}
.hero .vs, .hero .line {{ font-size: 15px; margin: 4px 0; color: {ON_HERO}; }}
.hero hr {{ border: none; border-top: 1px solid rgba(255,255,255,.3); margin: 12px 0; }}

/* --- Inputs --- */
[class*="st-key-{CARD_KEY_PREFIX}inputs"] [data-testid="stVerticalBlock"] {{ gap: .1rem; }}
[data-testid="stWidgetLabel"] p {{ font-size: 14px; word-break: normal; overflow-wrap: normal; hyphens: manual; }}
[data-testid="stSlider"] {{ padding-bottom: 0; }}
[data-testid="stSlider"] * {{ font-size: 14px; }}
[data-testid="stSliderThumbValue"] {{ font-family: var(--font-num); }}
.st-key-preset [data-testid="stButtonGroup"] {{ justify-content: flex-end; }}
.st-key-reset button {{ min-height: 32px; padding-top: 2px; padding-bottom: 2px; }}

/* --- Segmented selectors and pills: active option in green --- */
{_sel("button")} {{ background: transparent; }}
{_sel('button[aria-checked="true"]')} {{ border-color: var(--cy-green); }}
{_sel('button[aria-checked="true"] p')} {{ color: var(--cy-green); font-weight: 700; }}
{_sel("button p")} {{ font-size: 14px; }}
/* Primary buttons: dark text in dark mode (white on #66BB6A fails contrast) */
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primary"] p {{
    color: var(--cy-on-primary) !important; }}

/* --- Tables (st.table) --- */
[data-testid="stTable"] table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
[data-testid="stTable"] th {{ background: var(--cy-surface-2); color: var(--cy-text); font-weight: 600;
    padding: 5px 12px; border-bottom: 1px solid var(--cy-border); text-align: left; }}
[data-testid="stTable"] td {{ color: var(--cy-text); padding: 4px 12px; border-bottom: 1px solid var(--cy-border);
    font-family: var(--font-num); font-size: 13.5px; }}
[data-testid="stTable"] td:first-child {{ font-family: Inter, sans-serif; font-size: 14px; }}
[data-testid="stTable"] tr:last-child td {{ border-bottom: none; }}
[data-testid="stTable"] th:not(:first-child), [data-testid="stTable"] td:not(:first-child) {{ text-align: right; }}
[data-testid="stTable"] {{ border: 1px solid var(--cy-border); border-radius: 10px; overflow-x: auto; }}
.best {{ color: var(--cy-green); font-weight: 700; }}

/* --- ⓘ tooltips (CSS only) --- */
.tip {{ position: relative; display: inline-block; margin-left: 5px; color: var(--cy-muted);
    cursor: help; font-size: 14px; font-weight: 400; line-height: 1; outline: none; }}
.tip:hover, .tip:focus {{ color: var(--cy-green); }}
.tip:hover::after, .tip:focus::after {{
    content: attr(data-tip); position: absolute; top: 150%; left: -12px; z-index: 1000;
    width: max-content; max-width: 290px; padding: 8px 12px; border-radius: 10px;
    background: var(--cy-text); color: var(--cy-bg); font-family: Inter, sans-serif;
    font-size: 14px; font-weight: 400; line-height: 1.4; white-space: normal; text-align: left;
    text-transform: none; letter-spacing: 0;
}}
.tip.left:hover::after, .tip.left:focus::after {{ left: auto; right: -12px; }}
[data-testid="stHtml"] {{ overflow: visible; }}

/* --- Leaderboard --- */
.board {{ width: 100%; border-collapse: collapse; font-size: 15px; color: var(--cy-text); }}
.board td {{ padding: 5px 8px; border-bottom: 1px solid var(--cy-border); }}
.board tr:last-child td {{ border-bottom: none; }}
.board td.medal {{ width: 34px; font-size: 18px; text-align: center; }}
.board td.rank {{ width: 34px; text-align: center; color: var(--cy-muted); font-family: var(--font-num); }}
.board td.err {{ text-align: right; font-family: var(--font-num); font-weight: 600; white-space: nowrap; }}
.board tr.first td {{ color: var(--cy-green); font-weight: 600; }}
.board tr.muted td {{ color: var(--cy-muted); }}

/* --- Lessons, pipeline strip, glossary --- */
.lessons {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: var(--sp-3); }}
.lesson {{ background: var(--cy-surface); border: 1px solid var(--cy-border); border-radius: 16px;
    box-shadow: var(--cy-shadow); padding: var(--sp-3) var(--sp-5); }}
.lesson .l-num {{ font-family: var(--font-num); font-size: 24px; font-weight: 600; color: var(--cy-green);
    margin: 0; line-height: 1.2; white-space: nowrap; }}
.lesson .l-num span {{ font-family: Inter, sans-serif; font-size: 14px; font-weight: 600;
    color: var(--cy-muted); margin-right: 8px; }}
.lesson .l-text {{ font-size: 15px; color: var(--cy-text); margin: 4px 0 0 0; line-height: 1.5; }}
.flow {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: var(--sp-2); }}
.flow .step {{ background: var(--cy-surface); border: 1px solid var(--cy-border); border-radius: 14px;
    box-shadow: var(--cy-shadow); padding: var(--sp-3) var(--sp-4); position: relative; }}
.flow .s-no {{ display: inline-block; width: 26px; height: 26px; border-radius: 999px;
    background: var(--cy-green); color: var(--cy-on-primary); text-align: center; line-height: 26px;
    font-weight: 700; font-size: 14px; font-family: var(--font-num); }}
.flow .s-title {{ font-family: var(--font-head); font-size: 16px; font-weight: 700;
    color: var(--cy-text); margin: 6px 0 2px 0; }}
.flow .s-text {{ font-size: 14px; color: var(--cy-muted); margin: 0; line-height: 1.45; }}
.flow .s-num {{ font-family: var(--font-head); font-size: 17px; font-weight: 800; color: var(--cy-green);
    margin: 8px 0 0 0; }}
.glossary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: var(--sp-4); }}
.glossary p {{ font-size: 14px; color: var(--cy-text); margin: 0; line-height: 1.45; }}
.glossary p b {{ display: block; font-size: 15px; color: var(--cy-green); margin-bottom: 2px; }}
.sources a {{ color: var(--cy-sky-text); font-weight: 600; }}
.sources p {{ margin: 0 0 var(--sp-2) 0; font-size: 15px; color: var(--cy-text); line-height: 1.5; }}
.honest {{ font-size: 14px; color: var(--cy-muted); margin: 0; }}

/* --- Element toolbars are chrome (fullscreen buttons etc.) --- */
[data-testid="stElementToolbar"] {{ display: none; }}

/* --- Responsive: the Phase 1 breakpoints (1360 / 1200 / 900 px) --- */
@media (max-width: 1199.98px) {{
    [data-testid="stMainBlockContainer"] {{ padding-left: var(--sp-5) !important;
        padding-right: var(--sp-5) !important; }}
    .st-key-card-inputs [data-testid="stHorizontalBlock"]:has(.st-key-in_avg_temperature)
        > [data-testid="stColumn"] {{
        flex: 1 1 calc(50% - 2rem) !important; min-width: calc(50% - 2rem) !important; }}
    .st-key-card-inputs [data-testid="stHorizontalBlock"]:has(.st-key-preset)
        > [data-testid="stColumn"]:first-child {{ flex: 1 1 100% !important; min-width: 100% !important; }}
    .st-key-card-inputs [data-testid="stHorizontalBlock"]:has(.st-key-preset)
        > [data-testid="stColumn"]:last-child {{ flex: 0 0 auto !important; width: auto !important;
        min-width: 180px !important; }}
}}
@media (max-width: 899.98px) {{
    [data-testid="stMainBlockContainer"] {{ padding-left: var(--sp-4) !important;
        padding-right: var(--sp-4) !important; }}
    /* side-by-side card pairs stack */
    [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [class*="st-key-{CARD_KEY_PREFIX}"]),
    [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [class*="st-key-hero-"]) {{
        flex-wrap: wrap; }}
    [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [class*="st-key-{CARD_KEY_PREFIX}"])
        > [data-testid="stColumn"],
    [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [class*="st-key-hero-"])
        > [data-testid="stColumn"] {{ flex: 1 1 100% !important; min-width: 100% !important; }}
    .app-title {{ font-size: 22px; }}
    .kpi .k-value {{ font-size: 26px; }}
}}
</style>
"""

# Default to light the first time a browser opens the app (projector-safe). Streamlit
# keeps the viewer's choice in localStorage; with both [theme.light] and [theme.dark]
# defined it otherwise follows the OS. The marker makes this a one-time default, so a
# viewer who later picks Dark or System in the menu keeps that choice.
DEFAULT_LIGHT_JS = f"""
<script>
(() => {{
  const flag = "{DEFAULT_LIGHT_FLAG}";
  try {{
    if (window.parent.localStorage.getItem(flag)) return;
    window.parent.localStorage.setItem(flag, "1");
    const key = Object.keys(window.parent.localStorage).find(k => k.startsWith("stActiveTheme-"))
      || "stActiveTheme-/-v2";
    if (window.parent.localStorage.getItem(key) !== JSON.stringify("Light")) {{
      window.parent.localStorage.setItem(key, JSON.stringify("Light"));
      window.parent.location.reload();
    }}
  }} catch (e) {{}}
}})();
</script>
"""


def inject_theme() -> None:
    """Add the dashboard CSS (and the one-time light default) to the page."""
    st.html(CSS)
    st.html(DEFAULT_LIGHT_JS, unsafe_allow_javascript=True)


def tip(text: str | None, left: bool = False) -> str:
    """An ⓘ icon that shows `text` on hover or keyboard focus ('' when there is no text)."""
    if not text:
        return ""
    side = " left" if left else ""
    return f'<span class="tip{side}" tabindex="0" data-tip="{escape(text)}">ⓘ</span>'
