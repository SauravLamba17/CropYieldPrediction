"""The UI check's truncation rule catches an ellipsis label inside a wrapper DIV (Phase 7 bug).

Runs tests/ui_check.py's geometry script in headless Chrome/Edge on a tiny page shaped like a
Streamlit button; skips when no browser is installed (e.g. on CI).
"""

import json
import urllib.parse

import pytest
import ui_check

PAGE = """<!doctype html><body style="margin:0">
<button style="width:150px"><div id="cut" style="overflow:hidden;text-overflow:ellipsis;
white-space:nowrap;width:77px"><p style="margin:0">Download CSV</p></div></button>
<button style="width:300px"><div style="overflow:hidden;text-overflow:ellipsis;
white-space:nowrap"><p style="margin:0">Download PDF</p></div></button></body>"""


@pytest.fixture(scope="module")
def browser():
    exe = ui_check.find_browser()
    if not exe:
        pytest.skip("no Chrome or Edge installed")
    b = ui_check.Browser(exe)
    yield b
    b.close()


def test_truncated_wrapper_div_is_flagged(browser):
    browser.size(800)
    browser.send("Page.navigate", url="data:text/html," + urllib.parse.quote(PAGE))
    for _ in range(50):
        if browser.js("document.readyState") == "complete" and browser.js("!!document.body"):
            break
    problems = browser.js(ui_check.GEOMETRY_JS)
    assert problems == ['truncated: DIV "Download CSV"'], json.dumps(problems)
