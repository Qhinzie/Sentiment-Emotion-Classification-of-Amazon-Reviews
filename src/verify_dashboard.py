#!/usr/bin/env python3
"""Step-7 verification: Playwright drives the dashboard and checks every chart
and headline number against step6_metrics.json, step2_metrics.json (comparison
view) and step7_dataset_stats.json, measures min-size for nonzero bars, checks
label placement, and saves desktop + narrow screenshots.

Usage:  .venv/bin/python verify_dashboard.py
Exits 0 when every check passes (prints the pass/fail table).
"""
import csv
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from collections import Counter

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTPUTS = os.path.join(ROOT, "outputs")
SHOTS = os.path.join(ROOT, "docs", "screenshots")
MIN_PX = 3.0
URL = "http://127.0.0.1:{port}/"

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


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


def main():
    mj = json.load(open(os.path.join(OUTPUTS, "step6_metrics.json"), encoding="utf-8"))
    m2 = json.load(open(os.path.join(OUTPUTS, "step2_metrics.json"), encoding="utf-8"))
    ds = json.load(open(os.path.join(OUTPUTS, "step7_dataset_stats.json"), encoding="utf-8"))
    rows = list(csv.DictReader(open(os.path.join(OUTPUTS, "step6_predictions.csv"),
                                    encoding="utf-8")))
    for r in rows:
        r["rating_i"] = int(float(r["rating"]))

    n = len(rows)
    gt_cnt = Counter(r["ground_truth"] for r in rows)
    pr_cnt = Counter(r["prediction"] for r in rows)
    cm = Counter((r["ground_truth"], r["prediction"]) for r in rows)
    sample_stars = Counter(r["rating_i"] for r in rows)
    by_star = {}
    for s in range(1, 6):
        ok = sum(1 for r in rows if r["rating_i"] == s
                 and r["ground_truth"] == r["prediction"])
        by_star[s] = {"n": sum(1 for r in rows if r["rating_i"] == s),
                      "correct": ok, "accuracy": ok / max(1, sum(1 for r in rows if r["rating_i"] == s))}

    sent = mj["llm_sentiment"]
    baseline = sent["baseline_always_predict_majority"]

    # ---- launch the app on a free port ----
    port = free_port()
    env = dict(os.environ, GRADIO_SERVER_PORT=str(port),
               GRADIO_ANALYTICS_ENABLED="False")
    proc = subprocess.Popen([sys.executable, os.path.join(HERE, "dashboard.py")],
                            cwd=HERE, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = URL.format(port=port)
    if not wait_http(url):
        print("FAIL — dashboard did not come up")
        proc.terminate()
        return 1

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_selector('[data-chart="stars"]', timeout=60000)
            time.sleep(2)

            # ---------------- A) headline numbers (KPI cards) ----------------
            for kpi, key in (("accuracy", "accuracy"),
                             ("balanced_accuracy", "balanced_accuracy"),
                             ("macro_f1", "macro_f1")):
                v = page.eval_on_selector(
                    f'[data-kpi="{kpi}"]', "el => parseFloat(el.getAttribute('data-value'))")
                exp = sent[key]
                # attributes carry 6 decimals; compare within rounding
                check(f"KPI {kpi}", v is not None and abs(v - exp) < 1e-4,
                      f"{v} vs {exp:.6f}")

            body = page.inner_text("body")
            m = re.search(r"Right on (\d+) of (\d+)", body)
            exp_right = round(sent["accuracy"] * n)
            check("verdict", m and int(m.group(1)) == exp_right
                  and int(m.group(2)) == n,
                  f"{m.group(1)} of {m.group(2)} vs {exp_right} of {n}")
            check("verdict mentions all 3 classes",
                  all(c in body for c in ("negative", "neutral", "positive")))
            ag = mj["agreement"]
            check("agreement overall shown",
                  f"{ag['overall']:.1%}" in body, f"{ag['overall']:.1%}")
            check("agreement excl-none shown",
                  f"{ag['excluding_none']:.1%}" in body)
            check("baselines shown",
                  f"{sent['baseline_always_predict_majority']:.1%}" in body
                  and f"{m2['baseline_always_predict_positive']:.1%}" in body)

            # ---------------- B) charts vs json files ----------------
            # B1 star-rating distribution: full dataset panel (vs ds json)
            for s in range(1, 6):
                el = page.query_selector(
                    f'[data-panel="full"][data-star="{s}"]')
                vn = int(el.get_attribute("data-value"))
                vp = float(el.get_attribute("data-pct"))
                e = ds["stars_full"][str(s)]
                check(f"stars full {s}★", vn == e["n"] and abs(vp - e["pct"]) < 1e-5,
                      f"{vn} ({vp:.1%}) vs {e['n']} ({e['pct']:.1%})")
            # B2 star-rating distribution: sample panel (vs step6 csv)
            for s in range(1, 6):
                el = page.query_selector(
                    f'[data-panel="sample"][data-star="{s}"]')
                vn = int(el.get_attribute("data-value"))
                vp = float(el.get_attribute("data-pct"))
                check(f"stars sample {s}★",
                      vn == sample_stars[s] and abs(vp - sample_stars[s] / n) < 1e-5,
                      f"{vn} ({vp:.1%}) vs {sample_stars[s]} ({sample_stars[s] / n:.1%})")

            # B3 true vs predicted per class (vs step6 csv counts)
            for cls_ in gt_cnt:
                for kind, exp in (("true", gt_cnt[cls_]),
                                  ("pred", pr_cnt[cls_])):
                    el = page.query_selector(
                        f'[data-chart="tvp"][data-class="{cls_}"][data-kind="{kind}"]')
                    v = int(el.get_attribute("data-value"))
                    check(f"tvp {cls_} {kind}", v == exp, f"{v} vs {exp}")

            # B4 recall per class + chance reference line
            for cls_, act in gt_cnt.items():
                el = page.query_selector(
                    f'[data-chart="recall"][data-class="{cls_}"]')
                v = float(el.get_attribute("data-value"))
                exp = cm[(cls_, cls_)] / act
                check(f"recall {cls_}", abs(v - exp) < 1e-9, f"{v:.4f} vs {exp:.4f}")
            el = page.query_selector('[data-metric="baseline"]')
            v = float(el.get_attribute("data-value"))
            check("recall chance line", abs(v - baseline) < 1e-4,
                  f"{v:.4f} vs {baseline:.4f}")

            # B5 where each true class went (100% stacked, counts per segment)
            for a in ("positive", "neutral", "negative"):
                for pr_ in ("positive", "neutral", "negative"):
                    el = page.query_selector(
                        f'.st-seg[data-class="{a}"][data-pred="{pr_}"]')
                    v = int(el.get_attribute("data-value"))
                    exp = cm.get((a, pr_), 0)
                    check(f"sankey {a}->{pr_}", v == exp, f"{v} vs {exp}")
                # explicit zero labels present for the zero segments
                for pr_ in ("positive", "neutral", "negative"):
                    expected_zero = cm.get((a, pr_), 0) == 0
                    zero_el = page.query_selector(
                        f'.st-seg.st-zero[data-class="{a}"][data-pred="{pr_}"]')
                    cnt_zero = page.query_selector(
                        f'.st-cnt[data-class="{a}"][data-pred="{pr_}"][data-value="0"]')
                    check(f"sankey {a}->{pr_} zero handling",
                          (not expected_zero) or (zero_el is not None
                                                  and cnt_zero is not None))

            # B6 accuracy by star rating (vs step6 csv)
            for s in range(1, 6):
                el = page.query_selector(f'[data-chart="bystar"][data-star="{s}"]')
                v = float(el.get_attribute("data-value"))
                vn = int(el.get_attribute("data-n"))
                exp = by_star[s]
                check(f"bystar {s}★", abs(v - exp["accuracy"]) < 1e-4
                      and vn == exp["n"],
                      f"{v:.3f} (n={vn}) vs {exp['accuracy']:.3f} (n={exp['n']})")

            # B7 comparison view vs step2_metrics.json + step6 json
            vstbl = page.evaluate(
                "Array.from(document.querySelectorAll('.vs-table tr')).map("
                "tr => Array.from(tr.querySelectorAll('td,th')).map("
                "td => td.textContent.trim()))")
            col = {r[0]: r[1:] for r in vstbl[1:]}
            exp_vs = {
                "Accuracy": [f"{m2['accuracy']:.1%}", f"{sent['accuracy']:.1%}",
                             f"{mj['vs_step2']['step6_binary_collapsed']['accuracy']:.1%}"],
                "Chance baseline": [f"{m2['baseline_always_predict_positive']:.1%}",
                                    f"{baseline:.1%}",
                                    f"{max(mj['vs_step2']['step6_binary_collapsed']['n_by_class'].values()) / mj['vs_step2']['step6_binary_collapsed']['n']:.1%}"],
                "Macro-F1": [f"{m2['macro_f1']:.3f}", f"{sent['macro_f1']:.3f}", "—"],
                "Positive recall": [f"{m2['per_class']['positive']['recall']:.1%}",
                                    f"{sent['per_class']['positive']['recall']:.1%}",
                                    f"{mj['vs_step2']['step6_binary_collapsed']['per_class']['positive']['recall']:.1%}"],
                "Negative recall": [f"{m2['per_class']['negative']['recall']:.1%}",
                                    f"{sent['per_class']['negative']['recall']:.1%}",
                                    f"{mj['vs_step2']['step6_binary_collapsed']['per_class']['negative']['recall']:.1%}"],
                "Neutral recall": ["—", f"{sent['per_class']['neutral']['recall']:.1%}", "—"],
            }
            for name, exp in exp_vs.items():
                got = col.get(name)
                check(f"compare table {name}", got == exp, f"{got} vs {exp}")

            # ---------------- C) min visible size for nonzero values ----------------
            bad_min = page.evaluate("""() => {
              const bad = [];
              for (const el of document.querySelectorAll('.hb-fill, .st-seg')) {
                const v = parseFloat(el.getAttribute('data-value'));
                if (v === 0) continue;              // zeros are exempt (explicit '0')
                const r = el.getBoundingClientRect();
                if (r.width < 3 || r.height < 3) bad.push({
                  cls: el.className, v, w: r.width, h: r.height});
              }
              return bad; }""")
            check("min-size: every nonzero bar >= 3px", len(bad_min) == 0,
                  str(bad_min[:5]))

            # zero segments must be zero-width (no phantom inflation)
            zero_w = page.evaluate("""() => {
              const ws = [];
              for (const el of document.querySelectorAll('.st-seg.st-zero')) {
                const r = el.getBoundingClientRect();
                if (r.width > 0.5) ws.push(r.width);
              }
              return ws; }""")
            check("zero segments render zero-width", len(zero_w) == 0,
                  str(zero_w[:5]))

            # ---------------- D) labels never overlap bars ----------------
            overlap = page.evaluate("""() => {
              const bad = [];
              for (const row of document.querySelectorAll('.hbar')) {
                const fill = row.querySelector('.hb-fill');
                const note = row.querySelector('.hb-note b');
                if (!fill || !note) continue;
                const f = fill.getBoundingClientRect(), t = note.getBoundingClientRect();
                // clean rect intersection (tolerating 0.5px rounding)
                if (t.x < f.right - 0.5 && f.x < t.x + t.width - 0.5) {
                  bad.push({v: fill.getAttribute('data-value'), fx: f.x,
                            fw: f.width, tx: t.x, tw: t.width});
                }
              }
              return bad; }""")
            check("labels clear of bars", len(overlap) == 0, str(overlap[:5]))

            # ---------------- E) screenshots ----------------
            page.screenshot(path=os.path.join(SHOTS, "step7_desktop.png"),
                            full_page=True)
            check("screenshot desktop 1440px",
                  os.path.getsize(os.path.join(SHOTS, "step7_desktop.png")) > 20_000)

            page.set_viewport_size({"width": 400, "height": 900})
            time.sleep(1.5)
            page.screenshot(path=os.path.join(SHOTS, "step7_narrow.png"),
                            full_page=True)
            check("screenshot narrow 400px",
                  os.path.getsize(os.path.join(SHOTS, "step7_narrow.png")) > 10_000)

            # clipped labels at 400px: everything inside the viewport width
            clip = page.evaluate("""() => {
              const bad = [];
              const W = window.innerWidth;
              for (const sel of ['.hb-label', '.hb-note b', '.st-label', '.st-cnt']) {
                for (const el of document.querySelectorAll(sel)) {
                  const r = el.getBoundingClientRect();
                  if (r.width === 0) continue;
                  if (r.x < -0.5 || r.right > W + 0.5) bad.push(
                    {sel, x: Math.round(r.x), right: Math.round(r.right), W});
                }
              }
              return bad; }""")
            check("narrow: no clipped/overflowing labels", len(clip) == 0,
                  str(clip[:6]))

            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    fails = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(fails)}/{len(results)} checks passed")
    if fails:
        print(f"{len(fails)} FAILED: {[r[0] for r in fails]}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
