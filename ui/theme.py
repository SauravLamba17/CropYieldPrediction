"""'Cyber Sand' palette and the small amount of CSS that config.toml cannot express.

Colours, font and radius live in .streamlit/config.toml. This module adds only:
white cards with a soft shadow, pill-style tabs, compact spacing, and hidden
Streamlit chrome. Cards are targeted through `key=` classes (".st-key-card-...").
"""

from html import escape

import streamlit as st

CANVAS = "#F7F4EB"
CARD = "#FFFFFF"
NAVY = "#1E1B4B"
INDIGO = "#6366F1"
PURPLE = "#A855F7"


def navy(alpha: float) -> str:
    """Navy at reduced opacity (secondary text, borders, gridlines)."""
    return f"rgba(30, 27, 75, {alpha})"


def indigo(alpha: float) -> str:
    """Indigo at reduced opacity (soft backgrounds)."""
    return f"rgba(99, 102, 241, {alpha})"


def purple(alpha: float) -> str:
    """Purple at reduced opacity (soft backgrounds)."""
    return f"rgba(168, 85, 247, {alpha})"


CARD_KEY_PREFIX = "card-"  # every white card is a st.container(key="card-...")
SELECTOR_KEYS = ["india_table", "drivers_view", "explain_view", "preset", "mode"]
MODE_TOGGLE_WIDTH = 196  # px, the Simple / Technical control (the tab pills sit to its left)


def _sel(inner: str) -> str:
    """One CSS selector for `inner` inside every segmented selector / pill row."""
    return ", ".join(f".st-key-{k} {inner}" for k in SELECTOR_KEYS)


CSS = f"""
<style>
/* --- Streamlit chrome: header bar, footer, sidebar, top padding --- */
header[data-testid="stHeader"], footer, [data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"], [data-testid="stDecoration"] {{ display: none !important; }}
[data-testid="stMainBlockContainer"] {{
    padding: 0.6rem 2rem 0.25rem 2rem !important; max-width: 1640px;
}}
[data-testid="stMainBlockContainer"] > div > [data-testid="stVerticalBlock"] {{ gap: 0.65rem; }}

/* --- Cards --- */
[class*="st-key-{CARD_KEY_PREFIX}"] {{
    background: {CARD}; border: 1px solid {navy(0.08)}; border-radius: 16px;
    box-shadow: 0 1px 2px {navy(0.04)}, 0 8px 24px {navy(0.05)};
    padding: 14px 20px 12px 20px; gap: 0.4rem;
}}
.card-title {{ font-size: 16px; font-weight: 600; color: {NAVY}; margin: 0; line-height: 1.4; }}
.card-caption {{ font-size: 14px; color: {navy(0.62)}; margin: 0; line-height: 1.45; }}
.eyebrow {{ font-size: 14px; font-weight: 600; color: {navy(0.62)}; margin: 0 0 2px 0; }}

/* --- Header --- */
.app-title {{ font-size: 26px; font-weight: 700; color: {NAVY}; margin: 0; line-height: 1.2; }}
.app-subtitle {{ font-size: 15px; color: {navy(0.65)}; margin: 2px 0 0 0; }}

/* --- Tabs as pills (inactive: navy on transparent; active: purple, white text) --- */
/* The pill row sits in the header row (right-aligned) instead of taking its own row:
   the header is 57 px tall and the block gap 10.4 px, so -58 px centres the 38 px pills. */
[data-testid="stTabs"] {{ position: relative; }}
[data-testid="stTabs"] [role="tablist"] {{
    gap: 6px; border: none; position: absolute; top: -58px; right: 0; z-index: 5;
}}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"],
[data-testid="stTabs"] .react-aria-SelectionIndicator {{ display: none; }}
[data-testid="stTab"] {{
    border-radius: 999px; padding: 6px 16px; height: auto; background: transparent;
    border: 1px solid {navy(0.10)};
}}
[data-testid="stTab"] p {{ font-size: 15px; font-weight: 600; color: {NAVY}; }}
[data-testid="stTab"]:hover {{ background: {purple(0.08)}; }}
[data-testid="stTab"][aria-selected="true"] {{ background: {PURPLE}; border-color: {PURPLE}; }}
[data-testid="stTab"][aria-selected="true"] p {{ color: #FFFFFF; }}
[data-testid="stTabs"] [data-baseweb="tab-panel"] {{ padding-top: 0; }}
[data-testid="stTabs"] {{ margin-top: -0.5rem; }}

/* --- KPI cards (st.metric with border) --- */
[data-testid="stMetric"] {{
    background: {CARD}; border: 1px solid {navy(0.08)} !important; border-radius: 16px;
    box-shadow: 0 1px 2px {navy(0.04)}, 0 8px 24px {navy(0.05)}; padding: 14px 20px !important;
}}
[data-testid="stMetricLabel"] p {{ font-size: 14px; font-weight: 600; color: {navy(0.65)}; }}
[data-testid="stMetricValue"] {{ font-size: 30px; font-weight: 700; color: {NAVY}; }}

/* --- Compact inputs --- */
[class*="st-key-{CARD_KEY_PREFIX}inputs"] [data-testid="stVerticalBlock"] {{ gap: 0.1rem; }}
[data-testid="stWidgetLabel"] p {{ font-size: 14px; }}
[data-testid="stSlider"] {{ padding-bottom: 0; }}
.card-title .inline-caption {{ font-size: 14px; font-weight: 400; color: {navy(0.62)}; margin-left: 10px; }}

/* --- Hero prediction card (fills its column so it matches the chart card height) --- */
.st-key-hero-box, .st-key-hero-box [data-testid="stElementContainer"],
.st-key-hero-box [data-testid="stHtml"] {{ height: 100%; }}
.hero {{
    background: linear-gradient(135deg, {INDIGO} 0%, {PURPLE} 100%); color: #FFFFFF;
    border-radius: 16px; padding: 18px 24px; height: 100%; box-sizing: border-box;
    box-shadow: 0 8px 28px {indigo(0.30)};
}}
.hero .label {{ font-size: 15px; font-weight: 600; opacity: 0.9; margin: 0; }}
.hero .value {{ font-size: 52px; font-weight: 700; line-height: 1.1; margin: 6px 0 2px 0; }}
.hero .value span {{ font-size: 22px; font-weight: 600; opacity: 0.9; }}
.hero .vs {{ font-size: 15px; opacity: 0.92; margin: 0 0 14px 0; }}
.hero .line {{ font-size: 15px; margin: 4px 0; }}
.hero .line b {{ font-weight: 600; }}
.hero hr {{ border: none; border-top: 1px solid rgba(255, 255, 255, 0.28); margin: 12px 0; }}

/* --- KPI cards (custom HTML so no red/green deltas break the palette) --- */
.kpi {{
    min-height: 106px; box-sizing: border-box;
    background: {CARD}; border: 1px solid {navy(0.08)}; border-radius: 16px;
    box-shadow: 0 1px 2px {navy(0.04)}, 0 8px 24px {navy(0.05)}; padding: 14px 20px;
}}
.kpi .k-label {{ font-size: 14px; font-weight: 600; color: {navy(0.65)}; margin: 0; }}
.kpi .k-value {{ font-size: 30px; font-weight: 700; color: {NAVY}; margin: 2px 0 0 0; line-height: 1.2; }}
.kpi .k-value span {{ font-size: 16px; font-weight: 600; color: {navy(0.6)}; }}
.kpi .k-note {{ font-size: 14px; color: {navy(0.6)}; margin: 2px 0 0 0; }}
.kpi.accent .k-value {{ color: {INDIGO}; }}
.takeaway {{ font-size: 15px; color: {NAVY}; margin: 0; }}
.takeaway b {{ color: {NAVY}; font-weight: 600; }}
.takeaway b.accent {{ color: {INDIGO}; }}

/* --- Buttons: slim pill, 14 px label --- */
.st-key-reset button {{ min-height: 32px; padding-top: 2px; padding-bottom: 2px; }}
.st-key-reset button p {{ font-size: 14px; }}

/* --- Slider value readouts and min/max labels: keep the 14 px floor --- */
[data-testid="stSlider"] * {{ font-size: 14px; }}

/* --- Element hover toolbars are chrome: hidden, except to enlarge the beeswarm figure --- */
[data-testid="stElementToolbar"] {{ display: none; }}
.st-key-card-features [data-testid="stElementToolbar"] {{ display: flex; }}
[data-testid="stElementToolbarButtonContainer"] {{ background: {CARD}; }}

/* --- Tables (st.table): white, compact, navy hierarchy --- */
[data-testid="stTable"] table {{ width: 100%; border-collapse: collapse; background: {CARD}; font-size: 14px; }}
[data-testid="stTable"] th {{
    background: {indigo(0.08)}; color: {navy(0.72)}; font-weight: 600; font-size: 14px;
    padding: 5px 12px; border-bottom: 1px solid {navy(0.10)}; text-align: left;
}}
[data-testid="stTable"] td {{
    color: {NAVY}; padding: 4px 12px; border-bottom: 1px solid {navy(0.06)}; font-size: 14px;
}}
[data-testid="stTable"] tr:last-child td {{ border-bottom: none; }}
/* first column is always the row label; every other column is numeric */
[data-testid="stTable"] th:not(:first-child), [data-testid="stTable"] td:not(:first-child) {{
    text-align: right;
}}
[data-testid="stTable"] {{ border: 1px solid {navy(0.08)}; border-radius: 10px; overflow: hidden; }}

/* --- Segmented selectors and pills: active option in the secondary accent, like the tabs --- */
{_sel("button")} {{ background: transparent; }}
{_sel('button[aria-checked="true"]')} {{
    background: {purple(0.12)}; border-color: {PURPLE}; color: {PURPLE};
}}
{_sel('button[aria-checked="true"] p')} {{ color: {PURPLE}; font-weight: 600; }}
/* labels: keep the 14 px floor (Streamlit defaults to ~13 px) */
{_sel("button p")} {{ font-size: 14px; }}

/* --- Header: Simple / Technical toggle at the far right of the title row, tab pills left of it --- */
[data-testid="stMainBlockContainer"] {{ position: relative; }}
.st-key-mode {{ position: absolute; top: calc(0.6rem + 9px); right: 2rem; width: auto; z-index: 6; }}
[data-testid="stTabs"] [role="tablist"] {{ right: {MODE_TOGGLE_WIDTH + 14}px; }}

/* --- "In simple words" headline card --- */
.headline {{
    background: {CARD}; border: 1px solid {navy(0.08)}; border-radius: 16px;
    box-shadow: 0 1px 2px {navy(0.04)}, 0 8px 24px {navy(0.05)}; padding: 8px 20px;
}}
.headline .eyebrow {{ margin: 0; }}
.headline .h-text {{ font-size: 16px; color: {NAVY}; margin: 0; line-height: 1.45; }}
.headline b, .plain-text b, .lesson b {{ color: {INDIGO}; font-weight: 700; }}
.plain-text {{ font-size: 15px; color: {NAVY}; margin: 0; line-height: 1.45; }}

/* --- ⓘ tooltips (CSS only; text stays at 14 px) --- */
.tip {{
    position: relative; display: inline-block; margin-left: 5px; color: {navy(0.45)};
    cursor: help; font-size: 14px; font-weight: 400; line-height: 1; outline: none;
}}
.tip:hover, .tip:focus {{ color: {INDIGO}; }}
.tip:hover::after, .tip:focus::after {{
    content: attr(data-tip); position: absolute; top: 150%; left: -12px; z-index: 1000;
    width: max-content; max-width: 290px; padding: 8px 12px; border-radius: 10px; background: {NAVY}; color: #FFFFFF;
    font-size: 14px; font-weight: 400; line-height: 1.4; white-space: normal; text-align: left;
    box-shadow: 0 8px 24px {navy(0.25)};
}}
.tip.left:hover::after, .tip.left:focus::after {{ left: auto; right: -12px; }}
[data-testid="stHtml"] {{ overflow: visible; }}

/* --- Scenario presets (pills in the inputs card header) --- */
.st-key-preset [data-testid="stButtonGroup"] {{ justify-content: flex-end; }}

/* --- Simple-mode hero: gauge on the indigo-purple gradient --- */
.st-key-hero-gauge {{
    background: linear-gradient(135deg, {INDIGO} 0%, {PURPLE} 100%); border-radius: 16px;
    padding: 12px 22px; box-shadow: 0 8px 28px {indigo(0.30)}; gap: 0.2rem;
}}
.gauge-top {{ font-size: 15px; font-weight: 600; color: #FFFFFF; opacity: 0.92; margin: 0; }}
.gauge-top .tip {{ color: rgba(255, 255, 255, 0.85); }}
.gauge-value {{ text-align: center; color: #FFFFFF; margin: 0; line-height: 1.1; }}
.gauge-value .num {{ font-size: 40px; font-weight: 700; }}
.gauge-value .unit {{ font-size: 18px; font-weight: 600; opacity: 0.9; }}
.gauge-rating {{ text-align: center; color: #FFFFFF; font-size: 15px; margin: 2px 0 0 0; }}
.gauge-rating b {{
    background: rgba(255, 255, 255, 0.95); color: {INDIGO}; border-radius: 999px;
    padding: 2px 12px; font-weight: 700;
}}

/* --- Leaderboard --- */
.board {{ width: 100%; border-collapse: collapse; font-size: 15px; color: {NAVY}; }}
.board td {{ padding: 5px 8px; border-bottom: 1px solid {navy(0.06)}; }}
.board tr:last-child td {{ border-bottom: none; }}
.board td.medal {{ width: 34px; font-size: 18px; text-align: center; }}
.board td.rank {{ width: 34px; text-align: center; color: {navy(0.55)}; }}
.board td.err {{ text-align: right; font-weight: 600; white-space: nowrap; }}
.board tr.first td.err {{ color: {INDIGO}; }}
.board tr.muted td {{ color: {navy(0.6)}; }}

/* --- "3 things we learned" cards (equal heights across the row) --- */
.st-key-lessons [data-testid="stColumn"] > div, .st-key-lessons [data-testid="stElementContainer"],
.st-key-lessons [data-testid="stHtml"] {{ height: 100%; }}
.lesson {{
    background: {CARD}; border: 1px solid {navy(0.08)}; border-radius: 16px; height: 100%;
    box-sizing: border-box; box-shadow: 0 1px 2px {navy(0.04)}, 0 8px 24px {navy(0.05)};
    padding: 12px 20px;
}}
.lesson .l-num {{ font-size: 26px; font-weight: 700; color: {INDIGO}; margin: 0; line-height: 1.2; }}
.lesson .l-num span {{ font-size: 14px; font-weight: 600; color: {navy(0.6)}; margin-right: 8px; }}
.lesson .l-text {{ font-size: 15px; color: {NAVY}; margin: 4px 0 0 0; line-height: 1.45; }}

/* --- "How it works": step cards joined by arrows, glossary --- */
.flow {{ display: flex; align-items: stretch; gap: 10px; }}
.flow .step {{
    flex: 1; background: {CARD}; border: 1px solid {navy(0.08)}; border-radius: 16px;
    box-shadow: 0 1px 2px {navy(0.04)}, 0 8px 24px {navy(0.05)}; padding: 14px 18px;
}}
.flow .arrow {{ align-self: center; font-size: 26px; color: {PURPLE}; font-weight: 700; }}
.flow .s-no {{
    display: inline-block; width: 28px; height: 28px; border-radius: 999px; background: {INDIGO};
    color: #FFFFFF; text-align: center; line-height: 28px; font-weight: 700; font-size: 15px;
}}
.flow .s-title {{ font-size: 17px; font-weight: 700; color: {NAVY}; margin: 8px 0 4px 0; }}
.flow .s-text {{ font-size: 15px; color: {NAVY}; margin: 0; line-height: 1.45; }}
.flow .s-num {{ font-size: 24px; font-weight: 700; color: {INDIGO}; margin: 10px 0 0 0; }}
.flow .s-num span {{ display: block; font-size: 14px; font-weight: 600; color: {navy(0.6)}; }}
.glossary {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 18px; }}
.glossary p {{ font-size: 14px; color: {NAVY}; margin: 0; line-height: 1.45; }}
.glossary p b {{ display: block; font-size: 15px; color: {INDIGO}; margin-bottom: 2px; }}
.honest {{ font-size: 15px; color: {navy(0.72)}; margin: 0; }}
</style>
"""


def inject_theme() -> None:
    """Add the dashboard CSS to the page (call once, right after st.set_page_config)."""
    st.html(CSS)


def tip(text: str | None, left: bool = False) -> str:
    """An ⓘ icon that shows `text` on hover or keyboard focus ('' when there is no text).

    left=True opens the bubble towards the left (for icons near the right edge).
    """
    if not text:
        return ""
    side = " left" if left else ""
    return f'<span class="tip{side}" tabindex="0" data-tip="{escape(text)}">ⓘ</span>'


def headline_card(text: str) -> None:
    """The "In simple words" card that opens every tab (key numbers in <b> turn indigo)."""
    st.html(
        '<div class="headline"><p class="eyebrow">In simple words</p>'
        f'<p class="h-text">{text}</p></div>'
    )


def kpi_card(
    label: str,
    value: str,
    unit: str = "",
    note: str = "",
    accent: bool = False,
    help_text: str | None = None,
    help_left: bool = False,
) -> None:
    """A white KPI card: label (optional ⓘ), large value (optional unit), one-line note."""
    unit_html = f" <span>{unit}</span>" if unit else ""
    note_html = f'<p class="k-note">{note}</p>' if note else ""
    st.html(
        f'<div class="kpi{" accent" if accent else ""}">'
        f'<p class="k-label">{label}{tip(help_text, help_left)}</p>'
        f'<p class="k-value">{value}{unit_html}</p>{note_html}</div>'
    )


def card_title(
    title: str, caption: str | None = None, inline: bool = False, help_text: str | None = None
) -> None:
    """A card heading (optional ⓘ) with an optional caption, below it or on the same line."""
    title_html = f"{title}{tip(help_text)}"
    if inline and caption:
        st.html(
            f'<p class="card-title">{title_html}<span class="inline-caption">{caption}</span></p>'
        )
        return
    st.html(
        f'<p class="card-title">{title_html}</p>'
        + (f'<p class="card-caption">{caption}</p>' if caption else "")
    )
