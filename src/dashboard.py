#!/usr/bin/env python3
"""Step 3: results dashboard for the Step-2 LLM sentiment classification run (Gradio).

A gr.Blocks app that READS step2_predictions.csv and step2_metrics.json at
startup and renders: a plain-English verdict, headline metric cards, per-class
recall/precision bars, a colored confusion matrix, accuracy by star rating,
misclassified reviews grouped by pattern, an interactive filterable/searchable
table, and a run-details footer. No numbers are hardcoded.

Class-agnostic: class labels are read from the data, so adding a NEUTRAL class
later requires no code changes (filter pills, bars, confusion axes adapt).

Offline: system font stack, no Google Fonts, no external assets; all colors
live in the theme + one CSS-variable block.

Run:  .venv/bin/python dashboard.py
Then open http://127.0.0.1:7860
"""

import csv
import datetime
import html
import json
import os
import socket

import gradio as gr
import pandas as pd
from gradio.themes import Color, sizes
from gradio.themes.utils import fonts

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)             # repo root
OUTPUTS = os.path.join(ROOT, "outputs")  # predictions + metrics artifacts
# Step 5 (emotions) / Step 6 (three-class) outputs take priority; falls back
# to the Step-2 files.
_STEP6 = os.path.exists(os.path.join(OUTPUTS, "step6_predictions.csv"))
_STEP5 = os.path.exists(os.path.join(OUTPUTS, "step5_predictions.csv"))
if _STEP6:
    PRED_CSV = os.path.join(OUTPUTS, "step6_predictions.csv")
    METRICS_JSON = os.path.join(OUTPUTS, "step6_metrics.json")
    RUN_LABEL = "Step 6"
elif _STEP5:
    PRED_CSV = os.path.join(OUTPUTS, "step5_predictions.csv")
    METRICS_JSON = os.path.join(OUTPUTS, "step5_metrics.json")
    RUN_LABEL = "Step 5"
else:
    PRED_CSV = os.path.join(OUTPUTS, "step2_predictions.csv")
    METRICS_JSON = os.path.join(OUTPUTS, "step2_metrics.json")
    RUN_LABEL = "Step 2"

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

esc = html.escape


# ---------------------------------------------------------------------------
# Data + stats (computed from files at startup; same logic as Step 2)
# ---------------------------------------------------------------------------
def is_positive(lbl):
    return str(lbl).strip().lower() in {"positive", "pos"}


def is_negative(lbl):
    return str(lbl).strip().lower() in {"negative", "neg"}


def class_var(lbl):
    if is_positive(lbl):
        return "var(--dash-class-a)"
    if is_negative(lbl):
        return "var(--dash-class-b)"
    if str(lbl).strip().lower() in {"neutral", "neu"}:
        return "var(--dash-class-n)"
    return "var(--dash-accent)"


def pct(x):
    return f"{100.0 * x:.1f}%" if x is not None else "n/a"


def plural(n):
    return "review" if n == 1 else "reviews"


def star_of(r):
    try:
        return int(float(r["rating"]))
    except (TypeError, ValueError):
        return None


def load_rows():
    with open(PRED_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_stats(rows):
    n = len(rows)
    cnt = {}
    for r in rows:
        v = r["ground_truth"]
        cnt[v] = cnt.get(v, 0) + 1
    labels = sorted(cnt, key=lambda l: (-cnt[l], l))

    cm = {}
    for r in rows:
        key = (r["ground_truth"], r["prediction"])
        cm[key] = cm.get(key, 0) + 1

    per = {}
    for a in labels:
        tp = cm.get((a, a), 0)
        act = sum(cm.get((a, p), 0) for p in labels)
        prd = sum(cm.get((g, a), 0) for g in labels)
        recall = tp / act if act else 0.0
        precision = tp / prd if prd else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per[a] = dict(actual=act, predicted=prd, correct=tp,
                      recall=recall, precision=precision, f1=f1)

    correct_n = sum(cm.get((a, a), 0) for a in labels)
    majority = labels[0] if labels else None
    by_rating = {}
    for star in range(1, 6):
        rs = [r for r in rows if star_of(r) == star]
        if rs:
            ok = sum(1 for r in rs if r["ground_truth"] == r["prediction"])
            by_rating[star] = dict(n=len(rs), correct=ok, accuracy=ok / len(rs))
        else:
            by_rating[star] = dict(n=0)

    errors = [r for r in rows if r["ground_truth"] != r["prediction"]]
    return dict(n=n, labels=labels, counts=cnt, confusion=cm, per_class=per,
                accuracy=correct_n / n if n else 0.0, majority=majority,
                baseline=cnt[majority] / n if majority else 0.0,
                balanced_accuracy=sum(per[l]["recall"] for l in per) / len(per) if per else 0.0,
                macro_f1=sum(per[l]["f1"] for l in per) / len(per) if per else 0.0,
                by_rating=by_rating, errors=errors)


def verify(stats, mj):
    """Startup cross-check of every computed number against the metrics JSON."""
    sent = mj.get("llm_sentiment", mj)  # step5 nests sentiment metrics
    issues = []
    for name, a, b in [("n", stats["n"], sent.get("n")),
                       ("accuracy", stats["accuracy"], sent.get("accuracy")),
                       ("baseline", stats["baseline"],
                        sent.get("baseline_always_predict_positive")
                        or sent.get("baseline_always_predict_majority")),
                       ("balanced accuracy", stats["balanced_accuracy"], sent.get("balanced_accuracy")),
                       ("macro-F1", stats["macro_f1"], sent.get("macro_f1"))]:
        ok = a is not None and b is not None and abs(a - b) < 1e-9
        print(f"  [{'OK ' if ok else 'FAIL'}] {name}: {a} vs {b}")
        if not ok:
            issues.append(name)
    for l in stats["labels"]:
        p, mp = stats["per_class"][l], sent.get("per_class", {}).get(l)
        if mp is None:
            continue
        for m in ("precision", "recall", "f1"):
            ok = abs(p[m] - mp.get(m, -1)) < 1e-9
            print(f"  [{'OK ' if ok else 'FAIL'}] {l}.{m}: {p[m]} vs {mp[m]}")
            if not ok:
                issues.append(f"{l}.{m}")
    for star, v in stats["by_rating"].items():
        mv = sent.get("accuracy_by_rating", {}).get(str(star))
        if mv and abs(v.get("accuracy", -1) - mv.get("accuracy", -2)) > 1e-9:
            print(f"  [FAIL] {star}-star accuracy: {v} vs {mv}")
            issues.append(f"rating {star}")

    # Emotion block (step5): dashboard recomputes from the CSV and cross-checks
    # the JSON numbers it renders (agreement, distributions, cross-tab, NRC).
    rows = load_rows()
    emo_checks = []
    if mj.get("agreement") is not None:
        EM8 = ("anger", "anticipation", "disgust", "fear", "joy",
               "sadness", "surprise", "trust")
        def llm_in(em, nrc):
            return bool(em and nrc and nrc != "none" and em in nrc.split("/"))
        n = len(rows)
        agree = [r for r in rows if llm_in(r.get("llm_emotion"), r.get("nrc_emotion"))]
        excl = [r for r in rows if r.get("llm_emotion") and r.get("nrc_emotion") != "none"]
        ag = mj["agreement"]
        nrc = mj.get("nrc", {})
        n_tie_csv = sum(1 for r in rows if (r.get("nrc_tie") or "").lower() == "true")
        n_none_csv = sum(1 for r in rows if r.get("nrc_emotion") == "none")
        n_unparsed_csv = sum(1 for r in rows if not r.get("llm_emotion"))
        xt_total = sum(v for row in mj.get("cross_tab", {}).get("matrix", {}).values()
                       for v in row.values())
        dis_total = sum(len(v) for v in mj.get("disagreements", {}).get("by_reason", {}).values())
        for name, a, b in [
                ("emotion agreement overall", len(agree) / n, ag.get("overall")),
                ("emotion agreement excl-none", (len([r for r in excl if llm_in(r.get("llm_emotion"), r.get("nrc_emotion"))]) / len(excl)) if excl else None, ag.get("excluding_none")),
                ("llm emotion unparsed", n_unparsed_csv, sent.get("n_emotion_unparsed")),
                ("nrc none", n_none_csv, nrc.get("n_none")),
                ("nrc ties", n_tie_csv, nrc.get("n_tie")),
                ("cross-tab total", xt_total,
                 sent.get("n", 100) - n_none_csv - n_tie_csv - n_unparsed_csv),
                ("disagreement total", sent.get("n", 100) - len(agree), dis_total)]:
            ok = a is not None and b is not None and abs(a - b) < 1e-9
            print(f"  [{'OK ' if ok else 'FAIL'}] {name}: {a} vs {b}")
            emo_checks.append((name, ok))
            if not ok:
                issues.append(name)
        llm_dist = {}
        for r in rows:
            llm_dist[r.get("llm_emotion") or "unparsed"] = llm_dist.get(r.get("llm_emotion") or "unparsed", 0) + 1
        for e, c in llm_dist.items():
            jc = mj.get("distributions", {}).get("llm", {}).get(e)
            ok = jc == c
            print(f"  [{'OK ' if ok else 'FAIL'}] llm distribution {e}: {c} vs {jc}")
            if not ok:
                issues.append(f"llm dist {e}")

    # Step-6 blocks: true-NEUTRAL destination, error groups, vs_step2 payload
    if mj.get("errors") is not None:
        n_err_csv = sum(1 for r in rows if r["ground_truth"] != r["prediction"])
        n_err_json = sum(len(v) for v in mj["errors"]["by_pattern"].values())
        ok = n_err_csv == n_err_json
        print(f"  [{'OK ' if ok else 'FAIL'}] error total: {n_err_csv} vs {n_err_json}")
        if not ok:
            issues.append("error total")
        nd_json = sent.get("true_neutral_destination")
        if nd_json is not None:
            nd_csv = {}
            for r in rows:
                if r["ground_truth"] == "neutral":
                    nd_csv[r["prediction"]] = nd_csv.get(r["prediction"], 0) + 1
            for p, c in nd_json.items():
                ok = nd_csv.get(p, 0) == c
                print(f"  [{'OK ' if ok else 'FAIL'}] neutral -> {p}: {nd_csv.get(p, 0)} vs {c}")
                if not ok:
                    issues.append(f"neutral -> {p}")
        vs = mj.get("vs_step2", {}).get("step2_imbalanced_binary", {})
        if vs:
            try:
                with open(os.path.join(OUTPUTS, "step2_metrics.json"), encoding="utf-8") as f:
                    m2 = json.load(f)
            except (OSError, json.JSONDecodeError):
                m2 = None
            if m2:
                for name, a, b in [("vs step2 accuracy", vs.get("accuracy"), m2.get("accuracy")),
                                   ("vs step2 macro_f1", vs.get("macro_f1"), m2.get("macro_f1"))]:
                    ok = a is not None and b is not None and abs(a - b) < 1e-9
                    print(f"  [{'OK ' if ok else 'FAIL'}] {name}: {a} vs {b}")
                    if not ok:
                        issues.append(name)
                for cls_ in ("positive", "negative"):
                    a = vs.get("per_class", {}).get(cls_, {}).get("recall")
                    b = m2.get("per_class", {}).get(cls_, {}).get("recall")
                    ok = a is not None and b is not None and abs(a - b) < 1e-9
                    print(f"  [{'OK ' if ok else 'FAIL'}] vs step2 {cls_} recall: {a} vs {b}")
                    if not ok:
                        issues.append(f"vs step2 {cls_} recall")
    if issues:
        print(f"  MISMATCHES vs {METRICS_JSON.rsplit('/', 1)[-1]}:", issues)
    else:
        print(f"  ALL METRICS MATCH {METRICS_JSON.rsplit('/', 1)[-1]}")
    return issues


def error_pattern(r):
    gt, pr = r["ground_truth"], r["prediction"]
    star = star_of(r)
    if is_positive(gt) and is_negative(pr):
        if star == 4:
            return "Satisfied overall, but the text focuses on a complaint (4-star gripes)"
        return f"Positive {star}★ review whose text reads negative"
    if is_negative(gt) and is_positive(pr):
        if star == 3:
            return "Mixed 3-star reviews read as positive"
        return "Negative review with positive-sounding wording (sarcasm or irony)"
    return f"Misread ({gt} vs {pr})"


# ---------------------------------------------------------------------------
# HTML fragments (styled by the CSS block; same palette everywhere)
# ---------------------------------------------------------------------------
def verdict(stats):
    n, acc = stats["n"], stats["accuracy"]
    right = round(acc * n)
    per, labels = stats["per_class"], stats["labels"]
    if not labels:
        return "No classifiable reviews."
    catches = "caught " + ", ".join(
        f"{per[l]['correct']} of {per[l]['actual']} {l} {plural(per[l]['actual'])}"
        for l in labels) + "."
    if len(labels) == 2:
        maj, weak = labels[0], min(labels, key=lambda l: per[l]["f1"])
        m, w = per[maj], per[weak]
        maj_phrase = f"it caught {m['correct']} of {m['actual']} {maj} {plural(m['actual'])}"
        if w["precision"] < w["recall"]:
            weak_phrase = f"among the {w['predicted']} it called {weak}, {w['correct']} were right"
        elif w["recall"] < w["precision"]:
            weak_phrase = f"it missed {w['actual'] - w['correct']} of {w['actual']} {weak} {plural(w['actual'])}"
        else:
            weak_phrase = f"it caught {w['correct']} of {w['actual']} {weak} {plural(w['actual'])}"
        return f"Right on **{right} of {n}** reviews overall: {maj_phrase}, and {weak_phrase}."
    return f"Right on **{right} of {n}** reviews overall — {catches}"


def kpi_cards(stats):
    per = stats["per_class"]
    maj = stats["majority"]
    cls_balance_val = " · ".join(f"{l} {per[l]['actual']}" for l in stats["labels"])
    legend = " &nbsp;·&nbsp; ".join(
        f'<span class="lg"><span class="dot" style="background:{class_var(l)}"></span>{esc(l)} {per[l]["actual"]}</span>'
        for l in stats["labels"])
    stack = "".join(
        f'<i style="width:{per[l]["actual"] / stats["n"] * 100:.1f}%;background:{class_var(l)}"></i>'
        for l in stats["labels"])
    return [
        f"""<div class="dash-card kpi" data-kpi="accuracy" data-value="{stats['accuracy']:.6f}">
          <div class="kpi-label">Accuracy</div>
          <div class="kpi-val">{pct(stats['accuracy'])}</div>
          <div class="kpi-sub">baseline — always predict the majority class ({esc(maj)}): <b>{pct(stats['baseline'])}</b></div>
          <div class="dualbar"><i class="d-base" style="width:{stats['baseline'] * 100:.1f}%"></i><i class="d-acc" style="width:{stats['accuracy'] * 100:.1f}%"></i></div>
          <div class="dual-leg">teal bar = accuracy · grey bar = baseline</div></div>""",
        f"""<div class="dash-card kpi" data-kpi="balanced_accuracy" data-value="{stats['balanced_accuracy']:.6f}">
          <div class="kpi-label">Balanced accuracy</div>
          <div class="kpi-val">{pct(stats['balanced_accuracy'])}</div>
          <div class="kpi-sub">mean recall across classes</div>
          <div class="mini-rule"><i style="width:{stats['balanced_accuracy'] * 100:.1f}%"></i></div></div>""",
        f"""<div class="dash-card kpi" data-kpi="macro_f1" data-value="{stats['macro_f1']:.6f}">
          <div class="kpi-label">Macro-F1</div>
          <div class="kpi-val">{pct(stats['macro_f1'])}</div>
          <div class="kpi-sub">unweighted mean of per-class F1</div>
          <div class="mini-rule"><i style="width:{stats['macro_f1'] * 100:.1f}%"></i></div></div>""",
        f"""<div class="dash-card kpi">
          <div class="kpi-label">Class balance (sample)</div>
          <div class="kpi-val small">{esc(cls_balance_val)}</div>
          <div class="stackbar">{stack}</div>
          <div class="dual-leg">{legend}</div></div>""",
    ]


def per_class(stats):
    cards = []
    for l in stats["labels"]:
        p = stats["per_class"][l]
        color = class_var(l)
        if p["predicted"] == p["correct"]:
            prec_words = f"of the {p['predicted']} it called {l}, all {p['predicted']} were right"
        else:
            prec_words = f"of the {p['predicted']} it called {l}, {p['correct']} were right"
        rec_words = f"caught {p['correct']} of {p['actual']} {l} {plural(p['actual'])}"

        def bar(label, val, words):
            return (f'<div class="bar-row"><span class="l">{label}</span>'
                    f'<span class="track"><i style="width:{val * 100:.1f}%;background:{color}"></i></span>'
                    f'<span class="note">{pct(val)}</span></div>'
                    f'<div class="word">{esc(words)}</div>')
        cards.append(f"""<div class="dash-card">
          <div class="pc-head"><span class="dot" style="background:{color}"></span>
            {esc(l)} <span class="muted">· {p['actual']} {plural(p['actual'])} in sample</span></div>
          {bar("Recall", p["recall"], rec_words)}
          {bar("Precision", p["precision"], prec_words)}
        </div>""")
    return "".join(cards)


def confusion(stats):
    labels = stats["labels"]
    k = len(labels)
    head = '<div class="cm-corner">predicted →</div>' + "".join(
        f'<div class="cm-col">{esc(l)}</div>' for l in labels)
    rows = [head]
    for a in labels:
        act = sum(stats["confusion"].get((a, p), 0) for p in labels)
        cells = [f'<div class="cm-rowlab">actual: {esc(a)}</div>']
        for p in labels:
            c = stats["confusion"].get((a, p), 0)
            share = c / act if act else 0.0
            if a == p:
                style = (f"background:color-mix(in srgb, var(--dash-good) "
                         f"{8 + 62 * share:.0f}%, transparent)")
            else:
                style = (f"background:color-mix(in srgb, var(--dash-bad) "
                         f"{8 + 62 * share:.0f}%, transparent)")
            if c == 0:
                style = "background:var(--dash-surface-2)"
            cells.append(f'<div class="cm-cell" style="{style}"><b>{c}</b>'
                         f'<span>{pct(share) if act else "—"}</span></div>')
        rows.append("".join(cells))
    caption = ("Rows are the true class, columns are the model’s prediction. Green cells on the "
               "diagonal are correct calls — the stronger the tint, the more reviews. Red cells off "
               "the diagonal are errors.")
    return (f'<div class="cm" style="grid-template-columns:auto repeat({k},1fr)">'
            f'{"".join(rows)}</div><p class="cm-cap">{caption}</p>')


def ratings(stats):
    out = []
    for star in range(1, 6):
        v = stats["by_rating"][star]
        if v["n"] == 0:
            out.append(f'<div class="rating-row"><span class="star">{star}★</span>'
                       f'<span class="n muted">no {star}-star reviews in this sample</span><span></span></div>')
            continue
        g, r = v["accuracy"] * 100, (1 - v["accuracy"]) * 100
        out.append(f"""<div class="rating-row"><span class="star">{star}★</span>
          <span class="n">{v['correct']} of {v['n']} ({pct(v['accuracy'])})</span>
          <span class="track2"><i class="g" style="width:{g:.1f}%"></i><i class="r" style="width:{r:.1f}%"></i></span></div>""")
    return "".join(out)


def error_groups(stats):
    groups = {}
    for r in stats["errors"]:
        groups.setdefault(error_pattern(r), []).append(r)
    if not groups:
        return '<p class="muted">No misclassified reviews.</p>'
    out = []
    for pattern, rs in groups.items():
        cards = []
        for r in rs:
            star = star_of(r) or "?"
            pred = r["prediction"] if r["prediction"] is not None else "—"
            cards.append(f"""<div class="dash-card err">
              <div class="err-meta"><span class="chip">{star}★</span><span class="badge">wrong</span></div>
              <h4>{esc(r['title']) or '(no title)'}</h4>
              <p class="err-text">{esc(r['text']) or '(no text)'}</p>
              <div class="err-flip"><span class="chip">true: {esc(r['ground_truth'])}</span>
                <span class="arrow">→</span>
                <span class="chip chip-pred">predicted: {esc(pred)}</span></div></div>""")
        out.append(f'<div class="err-group"><h3>{esc(pattern)} '
                   f'<span class="count">{len(rs)}</span></h3>{"".join(cards)}</div>')
    return "".join(out)


# ---------------------------------------------------------------------------
# Interactive table (filters + search -> filtered dataframe)
# ---------------------------------------------------------------------------
TRUNC = 160

def trunc(text):
    text = text or "(no text)"
    if len(text) <= TRUNC:
        return text
    return text[: TRUNC - 1].rstrip() + "…"


def build_table_rows(rows):
    out = []
    for i, r in enumerate(rows):
        star = star_of(r)
        ok = r["ground_truth"] == r["prediction"]
        pred = r["prediction"] if r["prediction"] is not None else "—"
        llm_em = r.get("llm_emotion") or "—"
        nrc_em = r.get("nrc_emotion") or "—"
        em_agree = (r.get("emotions_agree") or "").lower() == "true"
        out.append(dict(
            idx=i + 1, star=star, stars=f"{star}★" if star else "—",
            title=r["title"] or "(no title)", text=r["text"] or "(no text)",
            true=r["ground_truth"], predicted=pred,
            ok=ok, result="✓ Correct" if ok else "✗ Mismatched",
            llm_em=llm_em, nrc_em=nrc_em,
            em_ok=(em_agree if r.get("emotions_agree") is not None else None),
            em_result="✓ Agree" if em_agree else "✗ Disagree",
            nrc_words=r.get("nrc_matched_words") or "",
            hay=(f"{star}★ {r['title']} {r['text']} {llm_em} {nrc_em}").lower(),
        ))
    return out


TABLE_COLUMNS = ["#", "Stars", "Review title", "Review text", "True", "Predicted",
                 "LLM emotion", "NRC emotion", "Emotion agree", "Result"]


def _filter_rows(rows_all, ok_filter, cls_true, cls_pred, stars, query, emo_filter="any"):
    """Pure filtering — returns (rows as lists, number of correct rows)."""
    q = (query or "").strip().lower()
    out = []
    n_ok = 0
    for r in rows_all:
        if ok_filter == "correct" and not r["ok"]:
            continue
        if ok_filter == "mismatched" and r["ok"]:
            continue
        if cls_true != "any" and r["true"] != cls_true:
            continue
        if cls_pred != "any" and r["predicted"] != cls_pred:
            continue
        if stars != "any" and r["star"] != int(stars):
            continue
        if emo_filter == "agree" and not r["em_ok"]:
            continue
        if emo_filter == "disagree" and r["em_ok"]:
            continue
        if q and q not in r["hay"]:
            continue
        out.append([r["idx"], r["stars"], trunc(r["title"]), trunc(r["text"]),
                    r["true"], r["predicted"], r["llm_em"], r["nrc_em"],
                    r["em_result"], r["result"]])
        n_ok += r["ok"]
    return out, n_ok


def filter_table(rows_all, ok_filter, cls_true, cls_pred, stars, query, emo_filter="any"):
    """Apply every filter; returns (table update, count line / empty state)."""
    out, n_ok = _filter_rows(rows_all, ok_filter, cls_true, cls_pred, stars, query,
                             emo_filter)
    total = len(rows_all)
    if not out:
        return (gr.update(visible=False),
                "**No reviews match these filters** — try Reset filters.")
    df = pd.DataFrame(out, columns=TABLE_COLUMNS)
    line = (f"**Showing {len(out)} of {total} reviews** · "
            f"{n_ok} correct, {len(out) - n_ok} mismatched")
    return gr.update(visible=True, value=df), line


DETAIL_PLACEHOLDER = '<p class="hint">Click any cell of a review to see its full text here.</p>'


def review_detail_card(r):
    badge = ('<span class="badge badge-ok">correct</span>' if r["ok"]
             else '<span class="badge">mismatched</span>')
    em_badge = ('<span class="badge badge-ok">emotions agree</span>' if r["em_ok"]
                else '<span class="badge">emotions disagree</span>')
    words = (f'<div class="nrc-words">NRC matched words — {esc(r["nrc_words"])}</div>'
             if r.get("nrc_words") else "")
    return f"""<div class="dash-card detail">
      <div class="err-meta"><span class="chip">{esc(r['stars']) or '—'}</span>
        <span class="chip">true: {esc(r['true'])}</span><span class="arrow">→</span>
        <span class="chip chip-pred">predicted: {esc(r['predicted'])}</span>{badge}</div>
      <div class="err-meta" style="margin-top:6px"><span class="chip">LLM emotion: {esc(r['llm_em'])}</span>
        <span class="chip">NRC emotion: {esc(r['nrc_em'])}</span>{em_badge}</div>
      <h4>{esc(r['title'])}</h4>
      <p class="err-text">{esc(r['text'])}</p>{words}</div>"""


def make_detail_handler(by_idx):
    """Row-click -> full review card (uses evt.row_value: the whole clicked row)."""
    def on_select(evt: gr.SelectData):
        row = evt.row_value
        if not row:
            return DETAIL_PLACEHOLDER
        try:
            num = int(row[0])
        except (TypeError, ValueError, IndexError):
            return DETAIL_PLACEHOLDER
        r = by_idx.get(num)
        return review_detail_card(r) if r else DETAIL_PLACEHOLDER
    return on_select


# ---------------------------------------------------------------------------
# Step 5 — emotion comparison section (renders from step5_metrics.json only)
# ---------------------------------------------------------------------------
EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy",
            "sadness", "surprise", "trust"]


def emotion_agreement_cards(mj):
    """4 headline cards: agreement overall / excluding none / strict + NRC stats."""
    ag, nrc = mj["agreement"], mj["nrc"]
    mix = mj.get("distributions", {}).get("nrc", {})
    return [
        f"""<div class="dash-card kpi">
          <div class="kpi-label">Emotion agreement</div>
          <div class="kpi-val">{pct(ag['overall'])}</div>
          <div class="kpi-sub">LLM vs NRC — <b>{ag['n_agree']} of {ag['n']} reviews</b>.
          A tie counts as agreement when the LLM's emotion is among the tied ones.</div>
          <div class="mini-rule"><i style="width:{ag['overall'] * 100:.1f}%"></i></div></div>""",
        f"""<div class="dash-card kpi">
          <div class="kpi-label">…excluding NRC “none”</div>
          <div class="kpi-val">{pct(ag['excluding_none'])}</div>
          <div class="kpi-sub">only the <b>{ag['n_excluding_none']}</b> reviews where the
          word list actually found an emotion</div>
          <div class="mini-rule"><i style="width:{ag['excluding_none'] * 100:.1f}%"></i></div></div>""",
        f"""<div class="dash-card kpi">
          <div class="kpi-label">Strict exact match</div>
          <div class="kpi-val">{pct(ag['strict_exact_match'])}</div>
          <div class="kpi-sub">LLM emotion identical to the single NRC winner
          (ties count as disagreement here)</div>
          <div class="mini-rule"><i style="width:{ag['strict_exact_match'] * 100:.1f}%"></i></div></div>""",
        f"""<div class="dash-card kpi">
          <div class="kpi-label">Where they fail</div>
          <div class="kpi-val small">none {nrc['n_none']} · tie {nrc['n_tie']}</div>
          <div class="kpi-sub">NRC found no lexicon words in <b>{nrc['n_none']}</b> reviews
          and was tied at the top in <b>{nrc['n_tie']}</b> — intense words like
          “gift” carry several emotions. Unable to parse (LLM): 0.</div></div>""",
    ]


def emotion_crosstab(mj):
    """8x8 LLM-emotion (rows) x NRC-emotion (cols) grid with counts."""
    emos, matrix = mj["cross_tab"]["emotions"], mj["cross_tab"]["matrix"]
    skipped_tie, skipped_none = (mj["cross_tab"]["n_skipped_nrc_ties"],
                                 mj["cross_tab"]["n_skipped_nrc_none"])
    maxc = max((matrix[a][b] for a in emos for b in emos), default=1) or 1
    head = ('<div class="cm-corner">LLM →<br>NRC ↓</div>' +
            "".join(f'<div class="cm-col">{esc(e)}</div>' for e in emos))
    rows = [head]
    for a in emos:
        cells = [f'<div class="cm-rowlab">{esc(a)}</div>']
        for b in emos:
            c = matrix.get(a, {}).get(b, 0)
            if c:
                tint = 8 + 62 * (c / maxc)
                style = (f"background:color-mix(in srgb, var(--dash-accent) "
                         f"{tint:.0f}%, transparent)")
            else:
                style = "background:var(--dash-surface-2)"
            cells.append(f'<div class="cm-cell" style="{style}"><b>{c}</b>'
                         f'<span>{pct(c / len(matrix)) if False else ""}</span></div>')
        rows.append("".join(cells))
    grid_cols = f"auto repeat({len(emos)},1fr)"
    caption = (f"Rows are the LLM's emotion, columns the NRC word-list emotion. "
               f"{skipped_tie} NRC-tie reviews and {skipped_none} “none” reviews aren't "
               "in the grid (no single emotion to place).")
    return (f'<div class="cm emo-xtab" style="grid-template-columns:{grid_cols}">'
            f'{"".join(rows)}</div><p class="cm-cap">{caption}</p>')


def emotion_distributions(mj):
    """Paired bars per emotion: LLM (teal) vs NRC (orange), plus none/tie."""
    llm, nrc = mj["distributions"]["llm"], mj["distributions"]["nrc"]
    n = sum(llm.values()) or 1
    out = []
    for e in EMOTIONS:
        l, r = llm.get(e, 0), nrc.get(e, 0)
        out.append(f"""<div class="dist-row">
          <span class="dl">{esc(e)}</span>
          <span class="dt"><i class="d-llm" style="width:{l / n * 100:.1f}%"></i></span>
          <span class="dn">{l}</span>
          <span class="dt"><i class="d-nrc" style="width:{r / n * 100:.1f}%"></i></span>
          <span class="dn">{r}</span></div>""")
    for label, k in (("none", "none"), ("tie", "tie")):
        if k in nrc:
            r = nrc[k]
            out.append(f"""<div class="dist-row">
              <span class="dl">{label}</span><span class="dt"></span><span class="dn">0</span>
              <span class="dt"><i class="d-nrc" style="width:{r / n * 100:.1f}%"></i></span>
              <span class="dn">{r}</span></div>""")
    return ('<div class="dash-card"><div class="dist-head">'
            '<span class="lg"><span class="dot" style="background:var(--dash-accent)"></span>LLM emotion</span>'
            '<span class="lg" style="margin-left:14px"><span class="dot" style="background:var(--dash-class-b)"></span>NRC word list</span>'
            f'<span class="muted" style="margin-left:auto">counts out of {sum(llm.values())}</span></div>'
            + "".join(out) + "</div>")


def emotion_vs_tables(mj):
    """Emotion x sentiment/rating: per emotion, n, mean stars, % positive (GT)."""
    blocks = []
    for label, key in (("LLM emotions", "llm"), ("NRC emotions", "nrc")):
        rows = mj.get("emotion_vs_sentiment", {}).get(key, {})
        body = ""
        for e in EMOTIONS:
            v = rows.get(e)
            if not v:
                continue
            body += (f'<tr><td class="vl">{esc(e)}</td><td class="vn">{v["n"]}</td>'
                     f'<td class="vn">{v["mean_rating"]:.2f}★</td>'
                     f'<td class="vn">{pct(v["positive_share_gt"])}</td></tr>')
        blocks.append(
            f'<div class="dash-card"><div class="kpi-label">{esc(label)}</div>'
            '<table class="vs-table"><tr><th>emotion</th><th>n</th><th>mean ★</th>'
            f'<th>% positive</th></tr>{body}</table>'
            '<div class="hint">% positive = share of ground-truth positive reviews '
            f'({mj["n"]} total).</div></div>')
    return '<div style="display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(300px,1fr))">' + "".join(blocks) + "</div>"


def emotion_disagreements(mj):
    """Disagreement groups with examples + curated notes."""
    groups = mj.get("disagreements", {}).get("by_reason", {})
    if not groups:
        return '<p class="muted">No emotion disagreements — both methods always agree.</p>'
    out = []
    for reason, rs in groups.items():
        cards = []
        for r in rs[:3]:
            note = (f'<div class="cur-note">why: {esc(r["curated_note"])}</div>'
                    if r.get("curated_note") else "")
            words = (f'<div class="nrc-words">NRC matched — {esc(r["nrc_matched_words"])}</div>'
                     if r.get("nrc_matched_words") else '<div class="nrc-words">NRC matched — none</div>')
            cards.append(f"""<div class="dash-card err">
              <div class="err-meta"><span class="chip">{r['rating']}★</span>
                <span class="chip">LLM: {esc(r['llm_emotion'] or '—')}</span>
                <span class="arrow">→</span>
                <span class="chip chip-pred">NRC: {esc(r['nrc_emotion'])}</span>
                <span class="badge">disagree</span></div>
              <h4>{esc(r['title']) or '(no title)'}</h4>
              <p class="err-text">{esc(trunc(r['text']))}</p>
              {words}{note}</div>""")
        out.append(f'<div class="err-group"><h3>{esc(reason.replace("_", " "))} '
                   f'<span class="count">{len(rs)}</span></h3>{"".join(cards)}</div>')
    return "".join(out)


# ---------------------------------------------------------------------------
# Step 6 — comparison + 3-class error renderers
# ---------------------------------------------------------------------------
def comparison_view(mj, mj2=None):
    """Step 2 (imbalanced binary) vs Step 6 (balanced 3-class) side by side.
    Step-2 numbers come straight from step2_metrics.json (mj2, read at startup);
    Step-6 numbers come from step6_metrics.json and are cross-checked against
    the CSV + step2 file by verify() at startup."""
    s6 = mj.get("llm_sentiment", {})
    vs = mj.get("vs_step2", {})
    s6b = vs.get("step6_balanced_three_class", {})
    c6 = vs.get("step6_binary_collapsed", {})

    def rec(p):
        return f"{p:.1%}" if isinstance(p, (int, float)) else "—"

    def f1_(p):
        return f"{p:.3f}" if isinstance(p, (int, float)) else "—"

    if mj2:
        s2b = (f"{int(mj2.get('n', 0))} — imbalanced "
               f"{int(mj2.get('n_ground_truth_positive', 0))} pos / "
               f"{int(mj2.get('n_ground_truth_negative', 0))} neg (all rows)")
        s2_acc, s2_base, s2_mf1 = (mj2.get("accuracy"), mj2.get("baseline_always_predict_positive"),
                                   mj2.get("macro_f1"))
        s2_rec = {l: mj2.get("per_class", {}).get(l, {}).get("recall")
                  for l in ("positive", "negative")}
    else:
        s2b, s2_acc, s2_base, s2_mf1 = "—", None, None, None
        s2_rec = {}
    nb = c6.get("n_by_class", {})
    c6_base = (max(nb.values()) / sum(nb.values())) if nb and sum(nb.values()) else None
    rows = [
        ("Sample", s2b, f"{s6.get('n')} — balanced 50/50/50 (deduped pool)",
         f"{c6.get('n')} — POS/NEG rows only"),
        ("Classes", "2 — positive / negative", "3 — positive / neutral / negative",
         "2 (neutral predictions count as wrong)"),
        ("Accuracy", rec(s2_acc), rec(s6.get("accuracy")), rec(c6.get("accuracy"))),
        ("Chance baseline", rec(s2_base), rec(s6.get("baseline_always_predict_majority")),
         rec(c6_base)),
        ("Macro-F1", f1_(s2_mf1), f1_(s6b.get("macro_f1")), "—"),
    ]
    for cls_ in ("positive", "neutral", "negative"):
        r6 = s6b.get("per_class", {}).get(cls_, {}).get("recall")
        r2 = s2_rec.get(cls_) if cls_ != "neutral" else None
        r2c = c6.get("per_class", {}).get(cls_, {}).get("recall") if cls_ != "neutral" else None
        rows.append((f"{cls_.capitalize()} recall", rec(r2), rec(r6), rec(r2c)))
    body = "".join(
        f'<tr><td class="vl">{esc(n)}</td><td class="vn">{a}</td>'
        f'<td class="vn">{b}</td><td class="vn">{c}</td></tr>'
        for n, a, b, c in rows)
    note = esc(mj.get("vs_step2", {}).get("note", ""))
    return ('<div class="dash-card"><div class="kpi-label">Step 2 vs Step 6 — side by side</div>'
            '<table class="vs-table"><tr><th>metric</th><th>Step 2 (binary, imbalanced)</th>'
            '<th>Step 6 (3-class, balanced)</th><th>Step 6 collapsed to binary</th></tr>'
            f'{body}</table>'
            f'<div class="hint">{note}</div></div>')


def errors3(mj):
    """Step-6 error groups with curated why-notes (from step6_metrics.json)."""
    groups = mj.get("errors", {}).get("by_pattern", {})
    if not groups:
        return '<p class="muted">No misclassified reviews.</p>'
    out = []
    for pattern, rs in groups.items():
        cards = []
        for r in rs[:6]:
            note = (f'<div class="cur-note">why: {esc(r.get("note", ""))}</div>'
                    if r.get("note") else "")
            cards.append(f"""<div class="dash-card err">
              <div class="err-meta"><span class="chip">{r['rating']}★</span>
                <span class="chip chip-ok">true: {esc(r['ground_truth'])}</span>
                <span class="arrow">→</span>
                <span class="chip chip-pred">predicted: {esc(r['prediction'])}</span>
                <span class="badge">wrong</span></div>
              <h4>{esc(r['title']) or '(no title)'}</h4>
              <p class="err-text">{esc(trunc(r['text']))}</p>{note}</div>""")
        more = (f'<div class="hint">…and {len(rs) - 6} more in this group '
                f'({len(rs)} total)</div>' if len(rs) > 6 else "")
        out.append(f'<div class="err-group"><h3>{esc(pattern)} '
                   f'<span class="count">{len(rs)}</span></h3>{"".join(cards)}{more}</div>')
    return "".join(out)


def emotion_by_class(mj):
    """Emotion x true class: LLM and NRC mini-tables."""
    byc = mj.get("emotion_by_true_class", {})
    if not byc:
        return ""
    blocks = []
    for label, key in (("LLM emotion by true class", "llm"),
                       ("NRC emotion by true class", "nrc")):
        data = byc.get(key, {})
        body = ""
        for e, counts in sorted(data.items()):
            if not counts:
                continue
            cells = "".join(f'<td class="vn">{counts.get(c, 0)}</td>'
                            for c in ("positive", "neutral", "negative"))
            total = sum(counts.values())
            body += (f'<tr><td class="vl">{esc(e)}</td>{cells}'
                     f'<td class="vn">{total}</td></tr>')
        blocks.append(
            f'<div class="dash-card"><div class="kpi-label">{esc(label)}</div>'
            '<table class="vs-table"><tr><th>emotion</th><th>positive</th>'
            '<th>neutral</th><th>negative</th><th>n</th></tr>'
            f'{body}</table></div>')
    return '<div style="display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(320px,1fr))">' + "".join(blocks) + "</div>"


# ---------------------------------------------------------------------------
# Step 7 — descriptive & prediction charts (inline HTML bars, data attrs)
# ---------------------------------------------------------------------------
def hbar(label, val, note, color, attrs="", strong=False):
    """One horizontal bar: fixed label column, flexible track, fixed value
    column. The value lives OUTSIDE the bar (its own column), so a short bar
    never shrinks or overlaps its label. Nonzero fills get min-width:3px via
    CSS; zero values render the explicit '0' in the note column."""
    cls = "hb-note strong" if strong else "hb-note"
    return (f'<div class="hbar"><span class="hb-label">{esc(label)}</span>'
            f'<span class="hb-track"><i class="hb-fill" style="background:{color}" '
            f'{attrs}></i></span>'
            f'<span class="{cls}"><b data-value="{val}" data-label="{esc(label)}">'
            f"{esc(note)}</b></span></div>")


def stacked_row(true_label, pred_counts, total, color_fn):
    """100% stacked bar for one true class + an explicit count strip below
    (zeros shown as '0')."""
    order = ("positive", "neutral", "negative")
    segs, cells = "", ""
    for p in order:
        c = pred_counts.get(p, 0)
        w = (c / total * 100) if total else 0.0
        width = f"width:{w:.2f}%;" if c else "width:0;"
        inside = f">{c}</i>" if (c and w >= 15) else "></i>"
        cls = "st-seg" if c else "st-seg st-zero"
        segs += (f'<i class="{cls}" style="{width}background:{color_fn(p)}" '
                 f'data-class="{true_label}" data-pred="{p}" data-value="{c}"'
                 f'{inside}')
        cells += (f'<span class="st-cnt" data-class="{true_label}" '
                  f'data-pred="{p}" data-value="{c}">{esc(p)} {c}</span>'
                  + (" · " if p != "negative" else ""))
    return (f'<div class="stacked-row"><span class="st-label">'
            f'{esc(true_label)}</span><span class="st-track">{segs}</span></div>'
            f'<div class="st-counts">{cells}</div>')


def recall_takeaway(stats):
    per, labels = stats["per_class"], stats["labels"]
    weak = min(labels, key=lambda l: per[l]["recall"])
    r, base = per[weak]["recall"], stats["baseline"]
    rel = "above" if r >= base else "below"
    return (f"{weak.capitalize()} recall ({r:.0%}) is the weak spot — "
            f"{rel} the {base:.0%} chance line",
            "Recall = share of truly-X reviews the model caught. Solid bars = "
            "each class; dashed line = always-predict-the-majority baseline.")


def stars_takeaway(ds, sample_dist):
    f = ds.get("stars_full", {}); s = sample_dist
    f5, f3, s5, s3 = f["5"]["pct"], f["3"]["pct"], s["5"]["pct"], s["3"]["pct"]
    return (f"Rebalancing: 5★ drops from {f5:.0%} of the dataset to {s5:.0%} "
            "of the sample, 3★ rises from "
            f"{f3:.1%} to {s3:.0%}",
            "Counts are left of each bar; bar width = share of that panel's "
            "reviews. The dataset's star skew (right-heavy) vs the sample's "
            "forced thirds, side by side.")


def tvp_takeaway(stats):
    per, labels = stats["per_class"], stats["labels"]
    gaps = {l: per[l]["predicted"] - per[l]["actual"] for l in labels}
    w = max(gaps, key=lambda l: abs(gaps[l]))
    return (f"The largest true-vs-predicted gap is {w}: "
            f"{per[w]['actual']} true vs {per[w]['predicted']} predicted",
            "Solid (class color) = how many reviews are truly in the class; "
            "faded = how many the model predicted as that class.")


def sankey_takeaway(stats):
    per, labels = stats["per_class"], stats["labels"]
    best = None
    for a in labels:
        act = per[a]["actual"]
        if not act:
            continue
        dom, dc = max(((p, stats["confusion"].get((a, p), 0)) for p in labels),
                      key=lambda t: t[1])
        if dom != a and (best is None or dc / act > best[2]):
            best = (a, dom, dc / act)
    if best is None:
        return ("Most reviews stay in their true class",
                "Each bar = one true class; segments = where its reviews went.")
    a, dom, share = best
    return (f"{a.capitalize()} reviews collapse into {dom.upper()}: "
            f"{share:.0%} predicted {dom}",
            "Each bar is one true class, split by predicted class (100% "
            "stacked). Diagonal = correct. Counts below; zeros shown as 0.")


def bystar_takeaway(abr):
    ws = min(abr, key=lambda s: abr[s]["accuracy"])
    return (f"{ws}★ reviews are the hardest ({abr[ws]['accuracy']:.0%} "
            "correct)",
            "Accuracy per star rating on the balanced sample, with each "
            "star's n. 3★ is ground-truth NEUTRAL.")


def charts_section(rows, stats, mj, ds):
    per, labels = stats["per_class"], stats["labels"]

    # --- 1) star rating distribution: full deduped dataset vs sample ---
    sample_dist = {str(s): {"n": 0, "pct": 0.0} for s in range(1, 6)}
    n = len(rows)
    for r in rows:
        s = str(star_of(r))
        sample_dist[s]["n"] += 1
    for s in sample_dist:
        sample_dist[s]["pct"] = sample_dist[s]["n"] / n if n else 0.0

    def star_panel(prefix, data, ds_panel=None):
        body = ""
        for s in range(1, 6):
            d = data[str(s)]
            color = class_var("neutral") if s == 3 else "var(--dash-muted)"
            p = "var(--dash-accent)" if s == 5 else color
            note = f"{d['n']:,} ({d['pct']:.1%})"
            attrs = (f'data-panel="{ds_panel}" data-star="{s}" '
                     f'data-value="{d["n"]}" data-pct="{d["pct"]:.6f}"')
            body += hbar(f"{s}★", d["n"], note, p, attrs)
        return body

    full_attr = ""
    if ds and ds.get("stars_full"):
        t, c = stars_takeaway(ds, sample_dist)
        star_body = ('<div class="pair">'
                     '<div class="chart-panel"><div class="panel-head">Full '
                     f'deduped dataset <span class="muted">n = '
                     f'{ds["n_deduped"]:,}</span></div>'
                     + star_panel("full", ds["stars_full"], "full") + "</div>"
                     '<div class="chart-panel"><div class="panel-head">Balanced '
                     f'sample <span class="muted">n = {n}</span></div>'
                     + star_panel("sample", sample_dist, "sample") + "</div></div>")
        stars_card = chart_card("stars", t, c, star_body,
                                f'data-step="6"' if _STEP6 else "")
    else:
        star_body = ('<div class="chart-panel chart-panel-full"><div '
                     'class="panel-head">Sample <span class="muted">n = '
                     f'{n}</span></div>' + star_panel("sample", sample_dist,
                                                      "sample") + "</div>")
        stars_card = chart_card("stars", "Star-rating balance of the sample",
                                "One panel only: step7_dataset_stats.json "
                                "absent for the dataset side.", star_body, "")

    # --- 2) true vs predicted per class ---
    tvp = ""
    for l in labels:
        color = class_var(l)
        c = per[l]
        tvp += (f'<div class="tvp-class"><span class="tvp-lab" '
                f'style="color:{color}">{esc(l)}</span>'
                + hbar("true", c["actual"], f"{c['actual']} "
                       f"({c['actual'] / stats['n']:.1%})", color,
                       f'data-chart="tvp" data-class="{l}" data-kind="true" '
                       f'data-value="{c["actual"]}"', strong=True)
                + hbar("predicted", c["predicted"],
                       f"{c['predicted']} ({c['predicted'] / stats['n']:.1%})",
                       color, f'data-chart="tvp" data-class="{l}" '
                       f'data-kind="pred" data-value="{c["predicted"]}"')
                + '</div>')
    t, c = tvp_takeaway(stats)
    tvp_card = chart_card("tvp", t, c, tvp, "")

    # --- 3) per-class recall vs chance baseline ---
    rec = ""
    weak = min(labels, key=lambda l: per[l]["recall"])
    base = stats["baseline"]
    for l in labels:
        color = class_var(l)
        if l == weak:
            color = "var(--dash-bad)"
        note = f"{per[l]['recall']:.1%}"
        if l == weak:
            note += " · weakest"
        rec += hbar(l, per[l]["recall"], note, color,
                    f'data-chart="recall" data-class="{l}" '
                    f'data-metric="recall" data-value="{per[l]["recall"]:.6f}"')
    ref = ('<span class="hb-ref" style="left:{:.1%}" data-metric="baseline" '
           'data-value="{:.6f}"></span>'
           '<span class="hb-ref-lab" style="left:{:.1%}">chance '
           '{:.1%}</span>').format(base, base, base, base)
    rec_html = '<div class="hb-track hb-track-ref">' + ref + "</div>"
    t, c = recall_takeaway(stats)
    recall_card = chart_card("recall", t, c, rec + rec_html, "")

    # --- 4) where each true class went (100% stacked) ---
    st = ""
    for a in labels:
        preds = {p: stats["confusion"].get((a, p), 0) for p in labels}
        st += stacked_row(a, preds, per[a]["actual"], class_var)
    t, c = sankey_takeaway(stats)
    sankey_card = chart_card("sankey", t, c, st, "")

    # --- 5) accuracy by star rating ---
    abr = {}
    for s in range(1, 6):
        rows_s = [r for r in rows if star_of(r) == s]
        if not rows_s:
            abr[s] = {"accuracy": None, "n": 0}
            continue
        acc = sum(1 for r in rows_s
                  if r["ground_truth"] == r["prediction"]) / len(rows_s)
        abr[s] = {"accuracy": acc, "n": len(rows_s)}
    bs = ""
    ws = min((s for s in abr if abr[s]["accuracy"] is not None),
             key=lambda s: abr[s]["accuracy"])
    for s in range(1, 6):
        d = abr[s]
        if d["accuracy"] is None:
            continue
        color = "var(--dash-bad)" if s == ws else "var(--dash-accent)"
        note = f"{d['accuracy']:.1%} (n={d['n']})"
        bs += hbar(f"{s}★", d["accuracy"], note, color,
                   f'data-chart="bystar" data-star="{s}" '
                   f'data-metric="accuracy" data-value="{d["accuracy"]:.6f}" '
                   f'data-n="{d["n"]}"')
    t, c = bystar_takeaway(abr)
    bystar_card = chart_card("bystar", t, c, bs, "")

    return ('<div class="charts">'
            '<div class="chart-grid">'
            + "".join([stars_card, tvp_card, recall_card,
                       sankey_card, bystar_card]) + "</div></div>")


def chart_card(chart_id, title, caption, body, extra_attrs=""):
    return (f'<div class="chart-card" data-chart="{chart_id}" '
            f'{extra_attrs}><h4 class="chart-title">{esc(title)}</h4>'
            f'<p class="chart-caption">{esc(caption)}</p>{body}</div>')


# ---------------------------------------------------------------------------
# Theme + CSS (all colors in one place)
# ---------------------------------------------------------------------------
TEAL = Color(c50="#f0fdfa", c100="#ccfbf1", c200="#99f6e4", c300="#5eead4", c400="#2dd4bf",
             c500="#14b8a6", c600="#0d9488", c700="#0f766e", c800="#115e59", c900="#134e4a",
             c950="#042f2e")
ORANGE = Color(c50="#fff7ed", c100="#ffedd5", c200="#fed7aa", c300="#fdba74", c400="#fb923c",
               c500="#f97316", c600="#ea580c", c700="#c2410c", c800="#9a3412", c900="#7c2d12",
               c950="#431407")
SLATE = Color(c50="#f8fafc", c100="#f1f5f9", c200="#e2e8f0", c300="#cbd5e1", c400="#94a3b8",
              c500="#64748b", c600="#475569", c700="#334155", c800="#1e293b", c900="#0f172a",
              c950="#020617")

SYSTEM_FONT = ("ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto",
               "Helvetica Neue", "Arial", "sans-serif")
SYSTEM_MONO = ("ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace")

theme = (gr.themes.Base(primary_hue=TEAL, secondary_hue=ORANGE, neutral_hue=SLATE,
                        font=SYSTEM_FONT, font_mono=SYSTEM_MONO, radius_size=sizes.radius_lg)
         .set(body_background_fill="#f1f3f2", body_background_fill_dark="#0e1211",
              body_text_color="#1c2320", body_text_color_dark="#e7ebe9",
              block_background_fill="#ffffff", block_background_fill_dark="#171c1a",
              block_border_color="#dce3df", block_border_color_dark="#29312e",
              block_title_text_color="#1c2320", block_title_text_color_dark="#e7ebe9",
              block_label_text_color="#5c6863", block_label_text_color_dark="#98a39f",
              border_color_primary="#dce3df", border_color_primary_dark="#29312e"))

CSS = """
/* ---- palette: one place to recolor the whole app ---- */
.gradio-container{
  --dash-bg:#f1f3f2; --dash-surface:#ffffff; --dash-surface-2:#f5f7f6;
  --dash-ink:#1c2320; --dash-muted:#5c6863; --dash-border:#dce3df;
  --dash-accent:#0f766e; --dash-accent-soft:#e1f0ed;
  --dash-class-a:#0f766e; --dash-class-b:#b45309; --dash-class-b-soft:#f9eee2;
  --dash-class-n:#5b5bd6; --dash-class-n-soft:#eef0fd;
  --dash-good:#21803f; --dash-bad:#cf3f31;
  --dash-good-soft:#e4f2ea; --dash-bad-soft:#fae7e4;
}
.gradio-container.dark,
body.dark .gradio-container{
  --dash-bg:#0e1211; --dash-surface:#171c1a; --dash-surface-2:#1e2422;
  --dash-ink:#e7ebe9; --dash-muted:#98a39f; --dash-border:#29312e;
  --dash-accent:#2dd4bf; --dash-accent-soft:#0f2e2a;
  --dash-class-a:#2dd4bf; --dash-class-b:#f0a35c; --dash-class-b-soft:#2c1a0c;
  --dash-class-n:#a5b4fc; --dash-class-n-soft:#1d2140;
  --dash-good:#53c37e; --dash-bad:#ef6a5a;
  --dash-good-soft:#122e1f; --dash-bad-soft:#341a16;
}
/* same dark palette via media query — flips in lockstep with gradio's theme */
@media (prefers-color-scheme: dark){
  .gradio-container{
    --dash-bg:#0e1211; --dash-surface:#171c1a; --dash-surface-2:#1e2422;
    --dash-ink:#e7ebe9; --dash-muted:#98a39f; --dash-border:#29312e;
    --dash-accent:#2dd4bf; --dash-accent-soft:#0f2e2a;
    --dash-class-a:#2dd4bf; --dash-class-b:#f0a35c; --dash-class-b-soft:#2c1a0c;
    --dash-class-n:#a5b4fc; --dash-class-n-soft:#1d2140;
    --dash-good:#53c37e; --dash-bad:#ef6a5a;
    --dash-good-soft:#122e1f; --dash-bad-soft:#341a16;
  }
}

/* ---- app shell ---- */
.gradio-container{max-width:1180px!important;background:var(--dash-bg)!important;color:var(--dash-ink)}
.gradio-container .app{background:transparent!important;padding-top:8px}
.gradio-container .block{background:var(--dash-surface);border-color:var(--dash-border)}
h1.dash-title{margin:2px 0 4px;font-size:24px;font-weight:650;letter-spacing:-.015em}
.dash-sub{color:var(--dash-muted);font-size:13.5px;margin-bottom:6px}
.dash-section{margin-top:30px}
.dash-kicker{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:var(--dash-muted);margin:0 0 2px}
.dash-section h2{font-size:18px;font-weight:650;margin:0 0 10px;color:var(--dash-ink)}
.muted{color:var(--dash-muted)}

/* ---- cards ---- */
.dash-card{background:var(--dash-surface);border:1px solid var(--dash-border);border-radius:14px;
  padding:18px 20px;box-shadow:0 1px 2px rgba(14,18,17,.05);height:100%}
.kpi-label{font-size:11.5px;color:var(--dash-muted);text-transform:uppercase;letter-spacing:.07em}
.kpi-val{font-size:30px;font-weight:680;font-variant-numeric:tabular-nums;margin:7px 0 3px;letter-spacing:-.02em}
.kpi-val.small{font-size:19px;line-height:1.35}
.kpi-sub{font-size:12.5px;color:var(--dash-muted)}
.dualbar{height:9px;border-radius:99px;background:var(--dash-surface-2);border:1px solid var(--dash-border);
  position:relative;margin-top:12px;overflow:hidden}
.dualbar i{position:absolute;inset-block:0;left:0;display:block}
.dualbar .d-base{background:var(--dash-border)}
.dualbar .d-acc{background:var(--dash-accent)}
.dual-leg{font-size:11.5px;color:var(--dash-muted);margin-top:6px}
.mini-rule{height:9px;border-radius:99px;background:var(--dash-surface-2);border:1px solid var(--dash-border);
  overflow:hidden;margin-top:12px}
.mini-rule i{display:block;height:100%;background:var(--dash-accent)}
.stackbar{display:flex;height:9px;border-radius:99px;overflow:hidden;border:1px solid var(--dash-border);margin-top:12px}
.stackbar i{height:100%}
.lg{display:inline-flex;align-items:center;gap:6px}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block;flex:none}

/* ---- per-class bars ---- */
.pc-head{display:flex;align-items:center;gap:8px;font-weight:620;margin-bottom:4px}
.pc-head .muted{font-weight:400}
.bar-row{display:grid;grid-template-columns:86px 1fr auto;gap:12px;align-items:center;margin-top:14px;font-size:13px}
.bar-row .l{color:var(--dash-muted)}
.track{display:block;height:8px;border-radius:99px;background:var(--dash-surface-2);border:1px solid var(--dash-border);overflow:hidden}
.track i{display:block;height:100%;border-radius:99px}
.bar-row .note{font-variant-numeric:tabular-nums;color:var(--dash-muted)}
.word{font-size:12.5px;color:var(--dash-muted);margin-top:3px}

/* ---- confusion matrix ---- */
.cm{display:grid;gap:6px;max-width:560px;margin-top:4px}
.cm-corner,.cm-col,.cm-rowlab,.cm-cell{border-radius:8px}
.cm-corner,.cm-col{font-size:11.5px;color:var(--dash-muted);text-align:center;padding:4px 6px}
.cm-rowlab{font-size:11.5px;color:var(--dash-muted);display:flex;align-items:center;padding-right:8px}
.cm-cell{text-align:center;padding:11px 6px;background:var(--dash-surface-2);border:1px solid var(--dash-border);
  font-variant-numeric:tabular-nums}
.cm-cell b{font-size:16px;font-weight:660;display:block}
.cm-cell span{font-size:11px;color:var(--dash-muted)}
.cm-cap{font-size:12.5px;color:var(--dash-muted);margin-top:12px;max-width:66ch}

/* ---- emotion comparison ---- */
.emo-xtab{max-width:860px}
.emo-xtab .cm-cell{padding:8px 4px}
.emo-xtab .cm-cell b{font-size:14px}
.emo-xtab .cm-corner,.emo-xtab .cm-col{font-size:10.5px;line-height:1.25}
.dist-head{display:flex;align-items:center;gap:8px;margin-bottom:10px;font-size:12.5px}
.dist-row{display:grid;grid-template-columns:110px 1fr 28px 1fr 28px;gap:10px;align-items:center;margin-top:8px;font-size:12.5px}
.dist-row .dl{color:var(--dash-muted);text-transform:capitalize}
.dt{display:block;height:8px;border-radius:99px;background:var(--dash-surface-2);border:1px solid var(--dash-border);overflow:hidden}
.dt i{display:block;height:100%}
.d-llm{background:var(--dash-accent)}
.d-nrc{background:var(--dash-class-b)}
.dn{font-variant-numeric:tabular-nums;color:var(--dash-muted);text-align:right}
.vs-table{width:100%;border-collapse:collapse;margin-top:10px;font-size:13px}
.vs-table th{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--dash-muted);text-align:right;padding:4px 8px;border-bottom:1px solid var(--dash-border)}
.vs-table th:first-child{text-align:left}
.vs-table td{padding:6px 8px;border-bottom:1px solid var(--dash-border);text-align:right;font-variant-numeric:tabular-nums}
.vs-table .vl{text-align:left;text-transform:capitalize;color:var(--dash-ink)}
.nrc-words{font-family:var(--font-mono);font-size:11.5px;color:var(--dash-muted);background:var(--dash-surface-2);border:1px solid var(--dash-border);border-radius:8px;padding:6px 10px;margin:0 0 10px;overflow-wrap:anywhere}
.cur-note{font-size:12px;color:var(--dash-accent);font-style:italic}

/* ---- ratings ---- */
.rating-row{display:grid;grid-template-columns:44px 140px 1fr;gap:12px;align-items:center;margin-top:12px}
.rating-row .star{font-weight:620}
.rating-row .n{font-size:13px;color:var(--dash-muted);font-variant-numeric:tabular-nums;text-align:right}
.track2{display:flex;height:8px;border-radius:99px;overflow:hidden;border:1px solid var(--dash-border);background:var(--dash-surface-2)}
.track2 .g{background:var(--dash-good)}
.track2 .r{background:var(--dash-bad)}

/* ---- errors ---- */
.err-group{margin-top:18px}
.count{font-size:11px;font-weight:600;color:var(--dash-muted);border:1px solid var(--dash-border);
  border-radius:99px;padding:1px 8px;vertical-align:2px;margin-left:4px}
.err{margin-top:10px}
.err-meta{display:flex;gap:8px;align-items:center;margin-bottom:8px}
.badge{font-size:10.5px;font-weight:650;text-transform:uppercase;letter-spacing:.05em;
  background:var(--dash-bad-soft);color:var(--dash-bad);padding:2px 9px;border-radius:99px}

/* ---- step 7 charts ---- */
.chart-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px}
.chart-card{border:1px solid var(--dash-border);border-radius:14px;background:var(--dash-surface);padding:16px 16px 14px}
.chart-card .chart-title{font-size:14.5px;margin:0 0 4px;color:var(--dash-ink)}
.chart-card .chart-caption{margin:0 0 12px;font-size:12.5px;line-height:1.45;color:var(--dash-muted)}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.chart-panel{border:1px solid var(--dash-border);border-radius:10px;padding:10px 12px;background:var(--dash-surface-2)}
.panel-head{font-size:12px;font-weight:650;color:var(--dash-ink);margin-bottom:6px}
.hbar{display:flex;align-items:center;gap:10px;margin:6px 0}
.hb-label{width:74px;flex:0 0 74px;font-size:12px;color:var(--dash-muted);text-align:right}
.hb-track{flex:1;position:relative;height:16px;border-radius:4px;overflow:visible;background:var(--dash-surface-2)}
.hb-fill{display:block;height:100%;min-width:3px;border-radius:4px}
.hb-note{width:130px;flex:0 0 130px;text-align:left;font-size:12px;color:var(--dash-muted);font-variant-numeric:tabular-nums}
.hb-note.strong b{color:var(--dash-ink)}
.hb-ref{position:absolute;top:-4px;bottom:-4px;width:1.5px;background:var(--dash-muted);opacity:.7;pointer-events:none}
.hb-ref-lab{position:absolute;top:-17px;font-size:10px;color:var(--dash-muted);transform:translateX(-60%);white-space:nowrap;pointer-events:none;font-variant-numeric:tabular-nums}
.hb-track-ref{margin-top:20px;height:8px;border-radius:2px}
.tvp-class{margin:10px 0 14px}
.tvp-lab{display:block;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;margin-bottom:2px}
.stacked-row{display:flex;align-items:center;gap:10px;margin:8px 0 2px}
.st-label{width:74px;flex:0 0 74px;font-size:11.5px;text-align:right;font-weight:650;color:var(--dash-ink);text-transform:uppercase}
.st-track{flex:1;display:flex;height:22px;border-radius:5px;overflow:hidden;background:var(--dash-surface-2);border:1px solid var(--dash-border)}
.st-seg{display:block;height:100%;min-width:3px;color:#0d1110;font-size:11px;font-weight:650;text-align:center;line-height:22px;font-variant-numeric:tabular-nums}
.st-seg.st-zero{min-width:0;width:0}
.st-counts{font-size:11.5px;color:var(--dash-muted);font-variant-numeric:tabular-nums;margin:0 0 12px 84px}
.st-counts .st-cnt[data-value="0"]{color:var(--dash-muted);opacity:.75}

@media (max-width:820px){
  .pair{grid-template-columns:1fr}
  .chart-grid{grid-template-columns:1fr}
}
.badge-ok{background:var(--dash-good-soft);color:var(--dash-good);padding:2px 9px;
  border-radius:99px;font-size:10.5px;font-weight:650;text-transform:uppercase;letter-spacing:.05em}
.hint{color:var(--dash-muted);font-size:12.5px;margin:8px 0 2px}
.detail h4{margin:0 0 6px;font-size:15px;font-weight:620}
.detail .err-text{margin-bottom:0}
.reset-btn{background:var(--dash-surface-2);border:1px solid var(--dash-border);
  color:var(--dash-ink);align-self:flex-end;margin-top:18px}
.err h4{margin:0 0 6px;font-size:15px;font-weight:620}
.err-text{margin:0 0 12px;color:var(--dash-muted);font-size:13.5px}
.err-flip{display:flex;align-items:center;gap:8px;font-size:13px;flex-wrap:wrap}
.arrow{color:var(--dash-muted)}
.chip{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--dash-border);
  background:var(--dash-surface-2);border-radius:7px;padding:3px 9px;font-size:12.5px;color:var(--dash-ink)}
.chip-pred{background:var(--dash-accent-soft);border-color:transparent;color:var(--dash-accent);font-weight:600}

/* ---- footer ---- */
.dash-footer{color:var(--dash-muted);font-size:12.5px;border-top:1px solid var(--dash-border);
  padding-top:14px;margin-top:8px}
.dash-footer b{color:var(--dash-ink);font-weight:620}

@media (max-width:700px){
  .rating-row{grid-template-columns:40px 1fr;row-gap:2px}
  .rating-row .track2{grid-column:1 / -1}
  .bar-row{grid-template-columns:1fr;gap:4px}
  .bar-row .note{text-align:left}
}
"""

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def build_app(rows, stats, mj, mj2=None, ds=None):
    rd = mj.get("run_details", {})
    has_emotions = mj.get("agreement") is not None
    is_3class = any(str(l).strip().lower() in {"neutral", "neu"} for l in stats["labels"])
    table_rows = build_table_rows(rows)
    initial_out, initial_ok = _filter_rows(table_rows, "all", "any", "any", "any", "")
    initial_df = pd.DataFrame(initial_out, columns=TABLE_COLUMNS)
    initial_count = (f"**Showing {len(initial_out)} of {len(table_rows)} reviews** · "
                     f"{initial_ok} correct, {len(initial_out) - initial_ok} mismatched")
    kpis = kpi_cards(stats)
    by_idx = {r["idx"]: r for r in table_rows}

    with gr.Blocks(title=f"{RUN_LABEL} · Sentiment + Emotion Classifier Dashboard",
                   fill_width=True) as demo:
        gr.Markdown(f"# Sentiment + emotion classifier — {RUN_LABEL} results\n"
                    f"<div class='dash-sub'>{stats['n']} reviews"
                    + (" — balanced three-class sample (POSITIVE / NEUTRAL / NEGATIVE)"
                       if is_3class else " sampled from the Amazon 2023 Gift Cards dataset")
                    + " · LLM sees only title + text · emotions: LLM (one of 8) vs "
                    "NRC word-list (EmoLex v0.92)</div>")

        with gr.Row():
            gr.Markdown("### In one sentence")
        with gr.Row():
            gr.Markdown(verdict(stats))

        if is_3class and mj2 is not None:
            with gr.Row():
                gr.Markdown('<div class="dash-kicker">01b · Side by side</div>\n\n'
                            "### Step 2 (binary, imbalanced) vs Step 6 (3-class, balanced)")
            with gr.Row():
                gr.HTML(comparison_view(mj, mj2))

        with gr.Row():
            gr.Markdown('<div class="dash-kicker">02 · Headline numbers</div>\n\n### How well did it do?')
        with gr.Row():
            for card in kpis:
                with gr.Column(scale=1):
                    gr.HTML(card)

        with gr.Row():
            gr.Markdown('<div class="dash-kicker">03 · Per class</div>'
                        '### Recall &amp; precision, in plain words')
        with gr.Row():
            gr.HTML('<div class="pc-grid" style="display:grid;gap:14px;grid-template-columns:'
                    f"repeat({len(stats['labels'])},1fr)\">{per_class(stats)}</div>")

        with gr.Row():
            gr.Markdown('<div class="dash-kicker">04 · Confusion matrix</div>\n\n'
                        "### Correct vs. wrong, class by class")
            gr.HTML('<div style="display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap">'
                    f'<div>{confusion(stats)}</div>'
                    '<div class="dash-card" style="max-width:330px"><div class="kpi-label">How to read it</div>'
                    '<div style="font-size:13px;color:var(--dash-muted);margin-top:8px">Rows are the true class, '
                    'columns are the model’s prediction. The diagonal (green) holds correct calls — the stronger '
                    'the tint, the more reviews. Red cells off the diagonal are errors.</div></div></div>')

        with gr.Row():
            gr.Markdown('<div class="dash-kicker">05 · Star ratings</div>\n\n'
                        "### Accuracy by star rating")
        with gr.Row():
            gr.HTML('<div class="dash-card" style="padding:22px 26px">' + ratings(stats) + "</div>")

        with gr.Row():
            gr.Markdown('<div class="dash-kicker">06 · Misclassified reviews</div>\n\n'
                        "### Every error, grouped by pattern")
        with gr.Row():
            gr.HTML('<div>' + (errors3(mj) if is_3class else error_groups(stats)) + "</div>")

        if has_emotions:
            with gr.Row():
                gr.Markdown('<div class="dash-kicker">07 · Emotion comparison</div>\n\n'
                            "### LLM emotion vs NRC word-list emotion")
            with gr.Row():
                for card in emotion_agreement_cards(mj):
                    with gr.Column(scale=1):
                        gr.HTML(card)
            with gr.Row():
                with gr.Column(scale=3):
                    gr.HTML(emotion_crosstab(mj))
                with gr.Column(scale=2):
                    gr.HTML(emotion_distributions(mj))
            with gr.Row():
                byc = emotion_by_class(mj)
                if byc:
                    gr.HTML(byc)
            with gr.Row():
                gr.HTML(emotion_vs_tables(mj))
            with gr.Row():
                gr.Markdown('<div class="dash-kicker">07b · Disagreements</div>\n\n'
                            "### Where the two emotions differ, grouped by reason")
            with gr.Row():
                gr.HTML('<div>' + emotion_disagreements(mj) + "</div>")

        with gr.Row():
            gr.Markdown('<div class="dash-kicker">08 · Descriptive & prediction '
                        "charts</div>\n\n### Where the numbers live")
        with gr.Row():
            gr.HTML(charts_section(rows, stats, mj, ds))

        with gr.Row():
            gr.Markdown('<div class="dash-kicker">09 · Full detail</div>\n\n'
                        "### All predictions — filter, search, verify")
        with gr.Row():
            ok_filter = gr.Radio(["all", "correct", "mismatched"], value="all", label="Result")
            emo_filter = gr.Radio(["any", "agree", "disagree"], value="any",
                                  label="Emotion match")
            true_cls = gr.Dropdown(["any"] + stats["labels"], value="any", label="True class")
            pred_cls = gr.Dropdown(["any"] + sorted({r["predicted"] for r in table_rows}),
                                   value="any", label="Predicted class")
            star_filter = gr.Dropdown(["any", "1", "2", "3", "4", "5"], value="any",
                                      label="Stars")
            reset_btn = gr.Button("Reset filters", elem_classes=["reset-btn"])
        with gr.Row():
            search = gr.Textbox(label="Search reviews", placeholder="Search title or text…")
        count_md = gr.Markdown(initial_count)
        table = gr.Dataframe(value=initial_df, interactive=False, wrap=True,
                             show_row_numbers=False, max_height=560)
        detail = gr.HTML(DETAIL_PLACEHOLDER)

        FILTER_INPUTS = [gr.State(table_rows), ok_filter, true_cls, pred_cls,
                         star_filter, search, emo_filter]
        for widget in (ok_filter, true_cls, pred_cls, star_filter, emo_filter):
            widget.change(fn=filter_table, inputs=FILTER_INPUTS,
                          outputs=[table, count_md])
        # live search (per keystroke) AND commit (Enter/blur)
        for trigger in ("input", "change", "submit"):
            getattr(search, trigger)(fn=filter_table, inputs=FILTER_INPUTS,
                                     outputs=[table, count_md])
        reset_btn.click(fn=lambda: ("all", "any", "any", "any", "", "any",
                                    gr.update(visible=True, value=initial_df),
                                    initial_count),
                        outputs=[ok_filter, true_cls, pred_cls, star_filter, search,
                                 emo_filter, table, count_md])
        table.select(fn=make_detail_handler(by_idx), outputs=[detail])

        pv = f' · prompt version: <b>{esc(str(rd.get("prompt_version", "—")))}</b>'
        lex = (f' · NRC lexicon: <b>{esc(str(rd.get("nrc_lexicon", "—"))) or "—"}</b>'
               if rd.get("nrc_lexicon") else "")
        gr.Markdown(
            f'<div class="dash-footer">Run details — model: <b>{esc(str(rd.get("model", "—")))}</b>'
            f' · endpoint: <b>{esc(str(rd.get("endpoint", "—")))}</b> · seed: <b>{esc(str(rd.get("seed", "—")))}</b>'
            f' · sample: <b>{stats["n"]} reviews</b> (of {esc(str(rd.get("dataset_total_reviews", "—")))} in dataset)'
            f' · run date: <b>{esc(str(rd.get("run_date", "—")))}</b> · data: <b>{esc(str(rd.get("data_file", "—")))}</b>'
            f'{pv}{lex}'
            f'<br>Built by dashboard.py on {datetime.date.today().isoformat()} · fully offline</div>')
    return demo


def find_free_port(start=7860, tries=20):
    """Return the first free port at/after `start` (the default is often taken
    by another local app, so fall back rather than crash)."""
    for p in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return start


def main():
    rows = load_rows()
    stats = compute_stats(rows)
    try:
        with open(METRICS_JSON, encoding="utf-8") as f:
            mj = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"FATAL: cannot read {METRICS_JSON}: {e}")

    print("Verification — computed from step2_predictions.csv vs step2_metrics.json:")
    issues = verify(stats, mj)

    try:
        with open(os.path.join(OUTPUTS, "step2_metrics.json"), encoding="utf-8") as f:
            mj2 = json.load(f)
    except (OSError, json.JSONDecodeError):
        mj2 = None

    try:
        with open(os.path.join(OUTPUTS, "step7_dataset_stats.json"), encoding="utf-8") as f:
            ds = json.load(f)
    except (OSError, json.JSONDecodeError):
        ds = None

    demo = build_app(rows, stats, mj, mj2, ds)
    port = os.environ.get("GRADIO_SERVER_PORT")
    port = int(port) if port and port.isdigit() else find_free_port()
    print(f"\nLaunching Gradio app on http://127.0.0.1:{port} ...")
    demo.queue().launch(server_name="127.0.0.1", server_port=port, show_error=True,
                        theme=theme, css=CSS)
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
