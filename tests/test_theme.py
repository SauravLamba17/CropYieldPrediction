"""Theme tokens: both modes complete, WCAG AA contrast, and config.toml in sync."""

import re

import tomllib

from src import config
from ui import theme

CONFIG_TEXT = (config.ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
CONFIG = tomllib.loads(CONFIG_TEXT)


def test_both_modes_define_the_same_tokens():
    assert set(theme.TOKENS["light"]) == set(theme.TOKENS["dark"])
    for name in [*theme.SERIES, "bg", "surface", "text", "muted", "border"]:
        for mode in ("light", "dark"):
            assert theme.TOKENS[mode][name].startswith("#"), (mode, name)


def test_brand_colours_are_the_agreed_ones():
    light, dark = theme.TOKENS["light"], theme.TOKENS["dark"]
    brand = (light["bg"], light["green"], light["gold"], light["soil"], light["sky"], light["text"])
    assert brand == ("#F6F4EC", "#2E7D32", "#E0A526", "#8D6E63", "#3B82C4", "#1F2A1F")
    assert (dark["bg"], dark["green"], dark["gold"]) == ("#121611", "#66BB6A", "#F2C14E")
    assert light["shap_pos"] == "#1B7F5B"


def test_every_pair_passes_wcag_aa():
    failures = [r for r in theme.contrast_report() if r["ratio"] < r["need"]]
    assert not failures, failures


def test_contrast_formula():
    assert round(theme.contrast("#000000", "#FFFFFF"), 1) == 21.0
    assert round(theme.contrast("#777777", "#FFFFFF"), 2) == 4.48


def test_config_toml_matches_tokens():
    for mode in ("light", "dark"):
        section, t = CONFIG["theme"][mode], theme.TOKENS[mode]
        assert section["chartCategoricalColors"] == theme.series_colours(mode)
        assert section["chartSequentialColors"] == theme.SEQUENTIAL[mode]
        assert section["primaryColor"] == t["green"]
        assert section["backgroundColor"] == t["bg"]
        assert section["textColor"] == t["text"]


def test_fonts_are_bundled_not_downloaded():
    faces = CONFIG["theme"]["fontFaces"]
    assert {f["family"] for f in faces} == {"Inter", "Manrope", "IBM Plex Mono"}
    for face in faces:
        assert face["url"].startswith("app/static/")  # served by the app itself, no CDN
        assert (config.ROOT / face["url"].removeprefix("app/")).exists(), face["url"]
    assert "googleapis" not in CONFIG_TEXT
    assert CONFIG["server"]["enableStaticServing"] is True


def test_css_follows_the_mode_without_python():
    assert "light-dark(#F6F4EC, #121611)" in theme.CSS


def test_sidebar_is_never_hidden():
    """Below Streamlit's breakpoint the top navigation moves into the sidebar ("»"): hiding the
    sidebar left narrow screens with no navigation (Phase 7). Wide screens have no sidebar."""
    assert "stSidebar" not in theme.CSS
    assert "stExpandSidebarButton" not in theme.CSS


def test_light_default_covers_every_page_path():
    """The default writes Light for every page's theme key, not only the first page opened."""
    files = re.findall(r'st\.Page\("app_pages/(\w+)\.py"', (config.ROOT / "app.py").read_text("utf-8"))
    default = re.search(r'st\.Page\("app_pages/(\w+)\.py"[^)]*default=True', (config.ROOT / "app.py")
                        .read_text("utf-8")).group(1)
    assert set(theme.PAGE_PATHS) == {"" if f == default else f for f in files}
    js = theme.DEFAULT_LIGHT_JS
    assert str(list(theme.PAGE_PATHS)) in js
    assert "for (const k of keys)" in js and '"stActiveTheme-" + base + p + "-v2"' in js
    assert "explicit(" in js  # a viewer's own Light/Dark choice is kept
    assert theme.DEFAULT_LIGHT_FLAG in js
