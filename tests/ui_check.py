"""Headless-browser UI check of a running app: every page, both modes, both themes, any widths.

    python tests/ui_check.py --base http://localhost:8611 --out screenshots/phase8/script

Per page it flags: an error box, sideways scrolling, content off the right edge, clipped or
ellipsis-truncated text (any element, wrapper DIVs included), an undrawn chart, the wrong
theme, and navigation that cannot be reached (top links from 768 px, the "»" sidebar below):
5 pages in Simple mode, all 7 with "Show details".
It also opens /whatif and /india with empty storage on a dark-mode system: both must be light.
Drives Chrome or Edge over the DevTools protocol with `websockets` (installed with Streamlit).
Exit code 1 if anything is flagged. Not part of pytest's run (needs a server and a browser);
tests/test_ui_check.py checks the truncation rule itself.
"""

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from websockets.sync.client import connect

PAGES = ["", "predict", "whatif", "models", "explain", "explorer", "india"]
SIMPLE_NAV = 5  # Simple mode lists 5 pages; "Show details" lists all 7
BACKGROUND = {"light": "rgb(246, 244, 236)", "dark": "rgb(18, 22, 17)"}

# Returns a list of problem strings for the current page (empty = clean).
GEOMETRY_JS = r"""
(() => {
  const problems = [], vw = window.innerWidth;
  const visible = e => { const r = e.getBoundingClientRect(), c = getComputedStyle(e);
    return r.width > 0 && r.height > 0 && c.visibility !== "hidden" && c.display !== "none"; };
  const name = e => (e.dataset.testid || e.tagName) + ' "' + e.textContent.trim().slice(0, 40) + '"';
  const scrolls = e => { for (let p = e.parentElement; p; p = p.parentElement) {
    const o = getComputedStyle(p).overflowX; if (o === "auto" || o === "scroll") return true; }
    return false; };
  if (document.documentElement.scrollWidth > vw + 1) problems.push("sideways scroll");
  for (const e of document.querySelectorAll('[data-testid="stException"], .stException'))
    problems.push("error box: " + name(e));
  const sidebar = document.querySelector('[data-testid="stSidebar"]');
  for (const e of document.querySelectorAll("body *")) {
    if (!visible(e) || (sidebar && sidebar.contains(e)) || e.closest("svg")) continue;
    const c = getComputedStyle(e), r = e.getBoundingClientRect();
    const ownText = [...e.childNodes].some(n => n.nodeType === 3 && n.textContent.trim());
    const text = e.textContent.trim();
    if (r.right > vw + 1 && ownText && !scrolls(e)) problems.push("off the right edge: " + name(e));
    // truncation: an ellipsis box (often a wrapper DIV around the label) whose text does not fit
    if (text && c.textOverflow === "ellipsis" && e.scrollWidth > e.clientWidth + 1)
      problems.push("truncated: " + name(e));
    else if (ownText && c.overflowX === "hidden" && e.scrollWidth > e.clientWidth + 1)
      problems.push("clipped: " + name(e));
  }
  for (const ch of document.querySelectorAll('[data-testid="stVegaLiteChart"]')) {
    const g = ch.querySelector("svg, canvas");
    if (!g || g.getBoundingClientRect().width < 10) problems.push("chart not drawn");
  }
  return [...new Set(problems)];
})()
"""

# Ready = the app's own content has drawn: its title is visible, its background colour is
# applied and the page script has finished. On Cloud, Streamlit reports a page loaded ~1-1.5 s
# before anything is drawn; on a cold start the title shows while Predict is still loading.
READY_JS = r"""
(() => {
  const app = document.querySelector('[data-testid="stApp"]');
  if (!app || !BACKGROUNDS.includes(getComputedStyle(app).backgroundColor)) return false;
  if (app.getAttribute("data-test-script-state") === "running") return false;  // page script still going
  const title = document.querySelector(".app-title");
  if (!title || title.getBoundingClientRect().height === 0) return false;
  if (document.querySelector('[data-testid="stStatusWidget"] [data-testid="stSpinner"]')) return false;
  if (document.querySelector('[data-testid="stSkeleton"], [data-testid="stSpinner"]')) return false;
  if (!document.querySelector('[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"]'))
    return false;
  return [...document.querySelectorAll('[data-testid="stVegaLiteChart"]')]
    .every(c => c.querySelector("svg, canvas"));
})()
""".replace("BACKGROUNDS", json.dumps(list(BACKGROUND.values())))

NAV_JS = r"""
(async () => {
  const vw = window.innerWidth, inView = e => { const r = e.getBoundingClientRect();
    return r.width > 0 && r.height > 0 && r.left >= 0 && r.right <= vw + 1; };
  if (vw >= 768)
    return [...document.querySelectorAll('[data-testid="stTopNavLink"]')].filter(inView).length;
  const links = () => [...document.querySelectorAll('[data-testid="stSidebar"] a')].filter(inView);
  if (links().length) return links().length;   // the sidebar is already open
  const b = document.querySelector('[data-testid="stExpandSidebarButton"]');
  if (!b || !inView(b)) return "no open-navigation button";
  (b.querySelector("button") || b).click();
  await new Promise(r => setTimeout(r, 900));
  return links().length;
})()
"""

THEME_JS = r"""
(() => { const s = window.localStorage;
  s.setItem("cyp-default-light-v2", "1");
  s.setItem("stActiveTheme-" + location.pathname + "-v2", JSON.stringify("%s")); })();
"""


def find_browser() -> str:
    for path in [os.environ.get("CHROME", ""), shutil.which("chrome") or "", shutil.which("google-chrome") or "",
                 r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                 r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]:
        if path and Path(path).exists():
            return path
    return ""


class Browser:
    """A headless Chrome/Edge with one page, driven over the DevTools protocol."""

    def __init__(self, exe: str):
        self.profile = tempfile.mkdtemp(prefix="ui-check-")
        self.proc = subprocess.Popen(
            [exe, "--headless=new", "--remote-debugging-port=0", f"--user-data-dir={self.profile}",
             "--no-first-run", "--hide-scrollbars", "--disable-extensions", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        port_file = Path(self.profile) / "DevToolsActivePort"
        for _ in range(100):
            if port_file.exists() and port_file.read_text().strip():
                break
            time.sleep(0.1)
        port = port_file.read_text().split()[0]
        targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list"))
        url = next(t["webSocketDebuggerUrl"] for t in targets if t["type"] == "page")
        self.ws = connect(url, max_size=None, open_timeout=10)
        self.next_id = 0
        self.send("Page.enable")
        self.send("Runtime.enable")

    def send(self, method: str, **params):
        self.next_id += 1
        self.ws.send(json.dumps({"id": self.next_id, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.next_id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def js(self, expr: str):
        res = self.send("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return res.get("result", {}).get("value")

    def size(self, width: int, height: int = 900):
        self.send("Emulation.setDeviceMetricsOverride", width=width, height=height,
                  deviceScaleFactor=1, mobile=False)

    def wait_ready(self, timeout: float = 45) -> bool:
        end, streak = time.time() + timeout, 0
        while time.time() < end:
            streak = streak + 1 if self.js(READY_JS) else 0
            if streak >= 3:
                return True
            time.sleep(0.3)
        return False

    def screenshot(self, path: Path):
        path.write_bytes(base64.b64decode(self.send("Page.captureScreenshot", format="png")["data"]))

    def close(self):
        try:
            self.ws.close()
        finally:
            self.proc.terminate()
            self.proc.wait(timeout=10)
            shutil.rmtree(self.profile, ignore_errors=True)


def check_page(b: Browser, theme: str) -> list[str]:
    problems = list(b.js(GEOMETRY_JS) or [])
    bg = b.js('getComputedStyle(document.querySelector(\'[data-testid="stApp"]\')).backgroundColor')
    if bg != BACKGROUND[theme]:
        problems.append(f"theme: background {bg}, expected {theme}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="http://localhost:8611")
    ap.add_argument("--out", default="screenshots/ui_check")
    ap.add_argument("--widths", type=int, nargs="+", default=[1366, 768, 600])
    ap.add_argument("--themes", nargs="+", default=["light", "dark"])
    args = ap.parse_args()
    exe = find_browser()
    if not exe:
        print("No Chrome or Edge found (set CHROME=path).")
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = args.base.rstrip("/") + "/"
    rows, b = [], Browser(exe)
    try:
        b.send("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "dark"}])
        for theme in args.themes:
            script = b.send("Page.addScriptToEvaluateOnNewDocument",
                            source=THEME_JS % theme.capitalize())["identifier"]
            for width in args.widths:
                b.size(width)
                for page in PAGES:
                    for mode in ("simple", "technical"):
                        t0 = time.time()
                        if mode == "simple":
                            b.send("Page.navigate", url=base + page)
                            time.sleep(0.5)
                        else:
                            b.js("[...document.querySelectorAll('.st-key-mode button')]"
                                 ".find(x => x.textContent.includes('Show details')).click()")
                            time.sleep(0.6)
                        ready = b.wait_ready()
                        problems = ([] if ready else ["page did not finish loading"]) + check_page(b, theme)
                        shot = out / f"{mode}_{theme}_{page or 'overview'}_{width}.png"
                        if mode == "simple":   # clean screenshot first: below 768 px the nav check opens the sidebar
                            b.screenshot(shot)
                        expected = SIMPLE_NAV if mode == "simple" else len(PAGES)
                        links = b.js(NAV_JS)
                        if links != expected:
                            problems.append(f"navigation: {links} links reachable, expected {expected}")
                        if mode == "technical":   # with the sidebar open below 768 px
                            b.screenshot(shot)
                        if mode == "technical" and width < 768:   # close the sidebar again
                            b.js("document.querySelector('[data-testid=\"stSidebarCollapseButton\"] "
                                 "button')?.click()")
                            time.sleep(0.5)
                        rows.append({"theme": theme, "width": width, "page": page or "overview",
                                     "mode": mode, "seconds": round(time.time() - t0, 2),
                                     "problems": problems})
                        print(f"{'PASS' if not problems else 'FAIL'}  {theme:5} {width:4} "
                              f"{page or 'overview':9} {mode:9} {'; '.join(problems)}", flush=True)
            b.send("Page.removeScriptToEvaluateOnNewDocument", identifier=script)
        # first visit on a dark-mode system, straight to a sub-page: must open light
        b.size(args.widths[0])
        for page in ("whatif", "india"):
            b.send("Page.navigate", url="about:blank")
            b.send("Storage.clearDataForOrigin", origin=base.rstrip("/"), storageTypes="all")
            b.send("Page.navigate", url=base + page)
            time.sleep(3)   # the default reloads the page once
            b.wait_ready()
            problems = check_page(b, "light")
            b.screenshot(out / f"default_light_direct_{page}_dark_os.png")
            rows.append({"theme": "default", "width": args.widths[0], "page": page, "mode": "fresh",
                         "seconds": None, "problems": problems})
            print(f"{'PASS' if not problems else 'FAIL'}  fresh storage, dark OS, /{page}: "
                  f"{'; '.join(problems) or 'light'}", flush=True)
    finally:
        b.close()
    (out / "rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    failed = [r for r in rows if r["problems"]]
    print(f"\n{len(rows) - len(failed)} of {len(rows)} checks pass.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
