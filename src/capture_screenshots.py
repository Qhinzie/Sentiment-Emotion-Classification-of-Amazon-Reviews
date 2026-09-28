#!/usr/bin/env python3
"""Step-8: capture the README screenshots with Playwright.

Launches the dashboard hermetically on a free port and saves section
screenshots to docs/screenshots/:
  01-dashboard-top.png         verdict + headline cards
  02-confusion-matrix.png      3×3 confusion matrix card
  03-where-each-class-went.png the 100% stacked "where true class went" chart
  04-emotion-comparison.png    emotion comparison section
  05-filtered-review-table.png review table with a live filter applied
  06-step2-vs-step6.png        the side-by-side comparison view

Usage:  .venv/bin/python src/capture_screenshots.py
"""
import os
import socket
import subprocess
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SHOTS = os.path.join(ROOT, "docs", "screenshots")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def wait_http(url, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def section_rect(page, start_text, end_text=None, pad_top=8, pad_bottom=10):
    """Viewport-rect (for clip screenshots) of a section: from the h3 whose
    text contains start_text to the h3 containing end_text (or the viewport
    bottom). Scrolls the start heading into view first."""
    page.locator(f"h3:has-text('{start_text}')").scroll_into_view_if_needed()
    time.sleep(0.5)
    return page.evaluate(
        """([s, e, pt, pb]) => {
            const h3s = Array.from(document.querySelectorAll('h3'));
            const a = h3s.find(h => h.textContent.includes(s));
            const b = e ? h3s.find(h => h.textContent.includes(e)) : null;
            if (!a) return null;
            const ar = a.getBoundingClientRect();
            const bottom = b ? b.getBoundingClientRect().top
                             : window.innerHeight - 60;
            return {x: 0, y: Math.max(0, ar.top - pt),
                    width: window.innerWidth,
                    height: Math.max(60, bottom - ar.top + pb)};
        }""", [start_text, end_text, pad_top, pad_bottom])


def main():
    port = free_port()
    env = dict(os.environ, GRADIO_SERVER_PORT=str(port),
               GRADIO_ANALYTICS_ENABLED="False")
    proc = subprocess.Popen([sys.executable, os.path.join(HERE, "dashboard.py")],
                            cwd=HERE, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}/"
    if not wait_http(url):
        print("FAIL: dashboard did not start")
        proc.terminate()
        return 1
    os.makedirs(SHOTS, exist_ok=True)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_selector('[data-chart="stars"]', timeout=60_000)
            time.sleep(2)
            page.wait_for_timeout(600)

            def shot(name, rect):
                page.screenshot(path=os.path.join(SHOTS, name), clip=rect)

            # 1) top: page top to end of section 02 (headline cards)
            r = page.evaluate("""() => {
                const h2 = Array.from(document.querySelectorAll('h3'))
                    .find(h => h.textContent.includes('How well did it do?'));
                if (!h2) return null;
                const br = h2.getBoundingClientRect();
                return {x: 0, y: 0, width: window.innerWidth,
                        height: br.top + 30};
            }""")
            shot("01-dashboard-top.png", r)

            # 2) confusion matrix (section 04)
            r = section_rect(page, "Correct vs. wrong", "Accuracy by star rating",
                             14, 30)
            shot("02-confusion-matrix.png", r)

            # 3) where-each-class-went chart card (section 08)
            page.locator('[data-chart="sankey"]').scroll_into_view_if_needed()
            time.sleep(0.6)
            r = page.evaluate("""() => {
                var el = document.querySelector('[data-chart="sankey"]');
                var b = el.getBoundingClientRect();
                return {x: b.x, y: b.y, width: b.width, height: b.height + 30};
            }""")
            shot("03-where-each-class-went.png", r)

            # 4) emotion comparison (section 07, incl. agreement + crosstab)
            r = section_rect(page, "LLM emotion vs NRC word-list emotion",
                             "All predictions", 14, 30)
            shot("04-emotion-comparison.png", r)

            # 5) filtered review table with live count
            page.locator("h3:has-text('All predictions')").scroll_into_view_if_needed()
            time.sleep(0.5)
            page.locator('input[value="mismatched"]').check()
            page.wait_for_function(
                "document.body.innerText.includes('Showing 42 of 150')",
                timeout=20_000)
            time.sleep(1)
            r = page.evaluate("""() => {
                const h = Array.from(document.querySelectorAll('h3'))
                    .find(x => x.textContent.includes('All predictions'));
                const hr = h.getBoundingClientRect();
                const tbl = document.querySelector('.virtual-table-viewport');
                let bottom = tbl ? tbl.getBoundingClientRect().bottom
                                 : hr.top + 620;
                return {x: 0, y: Math.max(0, hr.top - 12),
                        width: window.innerWidth,
                        height: Math.min(bottom - hr.top + 60, window.innerHeight - hr.top + 12)};
            }""")
            shot("05-filtered-review-table.png", r)

            # 6) step 2 vs step 6 comparison view (section 01b)
            page.locator("h3:has-text('Step 2')").scroll_into_view_if_needed()
            time.sleep(0.6)
            r = page.evaluate("""() => {
                const h = Array.from(document.querySelectorAll('h3'))
                    .find(x => x.textContent.includes('Step 2'));
                const hr = h.getBoundingClientRect();
                const tbl = document.querySelectorAll('.vs-table')[0];
                const tb = tbl.getBoundingClientRect();
                return {x: 0, y: Math.max(0, hr.top - 24),
                        width: window.innerWidth,
                        height: tb.bottom - hr.top + 70};
            }""")
            shot("06-step2-vs-step6.png", r)

            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    for f in sorted(os.listdir(SHOTS)):
        if f.startswith("0") or f in ("step7_desktop.png", "step7_narrow.png"):
            print(f"  saved docs/screenshots/{f} "
                  f"({os.path.getsize(os.path.join(SHOTS, f))} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
