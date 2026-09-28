#!/usr/bin/env python3
"""Every number quoted in README.md must be traceable to outputs/* (JSON files
or values derivable from the CSVs), or be a documented structural constant.

  .venv/bin/python src/check_readme_numbers.py
Exits 0 when every number is accounted for; prints an audit table.
"""
import csv
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTPUTS = os.path.join(ROOT, "outputs")
README = os.path.join(ROOT, "README.md")

# Structural constants that legitimately appear in the README without being
# data values (each with a reason). Anything else must match the outputs.
ALLOW = {
    "2023": "dataset year in the title/URL context",
    "2024": "paper year in citation",
    "2403.03952": "arXiv ID in citation",
    "29": "journal volume 29(3)",
    "3": "journal issue 29(3)",
    "436": "journal page range",
    "465": "journal page range",
    "0.92": "lexicon version (code-span stripped anyway)",
    "42": "fixed random seed",
    "100": "Step-2 sample size",
    "150": "Step-6 sample size",
    "50": "balanced per-class count",
    "1440": "screenshot width",
    "400": "screenshot width (narrow)",
    "2013": "publication year in citation",
    "63": "number of verify_dashboard.py checks",
    "10": "approx. setup download minutes",
    "120": "approx. MB downloaded by setup.sh",
    "1": "one model/one category in Limitations (prose)",
    "8": "eight NRC emotions (spec)",
}


def json_numbers():
    """All ints + floats (rounded 1/3/4 decimals) inside outputs/*.json."""
    ints, floats = set(), set()
    for f in glob.glob(os.path.join(OUTPUTS, "*.json")):
        def walk(o):
            if isinstance(o, bool):
                return
            if isinstance(o, int):
                ints.add(o)
            elif isinstance(o, float):
                floats.add(o)
            elif isinstance(o, dict):
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                ints.add(len(o))
                for v in o:
                    walk(v)
        with open(f, encoding="utf-8") as fh:
            walk(json.load(fh))
    return ints, floats


def csv_derived():
    """Counts + percentages computed from outputs/*.csv (same formulas as the
    pipelines/dashboard)."""

    def load(fn):
        with open(os.path.join(OUTPUTS, fn), encoding="utf-8") as f:
            return list(csv.DictReader(f))

    ints, pct = set(), set()
    for fn in ("step2_predictions.csv", "step5_predictions.csv",
               "step6_predictions.csv"):
        rows = load(fn)
        n = len(rows)
        ints.add(n)
        for r in rows:
            r["rating_i"] = int(float(r["rating"]))
        from collections import Counter
        gt = Counter(r["ground_truth"] for r in rows)
        pr = Counter(r["prediction"] for r in rows)
        for k, v in gt.items():
            ints.add(v)
            pct.add(round(v / n * 100, 1))
        for k, v in pr.items():
            ints.add(v)
            pct.add(round(v / n * 100, 1))
        cm = Counter((r["ground_truth"], r["prediction"]) for r in rows)
        for (a, b), v in cm.items():
            ints.add(v)
        for s in range(1, 6):
            rs = [r for r in rows if r["rating_i"] == s]
            ints.add(len(rs))
            if len(rs):
                ok = sum(1 for r in rs
                         if r["ground_truth"] == r["prediction"])
                ints.add(ok)
                pct.add(round(ok / len(rs) * 100, 1))
            pct.add(round(len(rs) / n * 100, 1))
        # error count + correct count
        err = sum(1 for r in rows if r["ground_truth"] != r["prediction"])
        ints.add(err)
        ints.add(n - err)
        for c in gt.values():
            pct.add(round(c / n * 100, 1))
    # balanced-sample star destination percentages (step6 only)
    step6 = load("step6_predictions.csv")
    neu = [r for r in step6 if r["ground_truth"] == "neutral"]
    for p in ("negative", "neutral", "positive"):
        v = sum(1 for r in neu if r["prediction"] == p)
        ints.add(v)
        pct.add(round(v / len(neu) * 100, 1))
    return ints, pct


def readme_tokens():
    with open(README, encoding="utf-8") as f:
        text = f.read()
    # strip code spans, URLs and markdown link targets
    text = re.sub(r"`[^`]*`", "", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[[^\]]*\]\([^)]*\)", lambda m: m.group(0).rsplit("](", 1)[0] + "]",
                  text)
    text = re.sub(r"https?://[^\s)\]]+", "", text)
    toks = []
    for m in re.finditer(r"(?<![\w.])(\d[\d,\.]*)(%|★|px)?(?![\w.])", text):
        num = m.group(1).replace(",", "").rstrip(".")
        toks.append((num, bool(m.group(2)) and m.group(2) == "%"))
    return toks, text


def main():
    jints, jfloats = json_numbers()
    cints, cpct = csv_derived()
    ints = jints | cints
    pcts = cpct | {round(f * 100, 1) for f in jfloats}

    def fmt_float_rounds(f):
        return {round(f, 1), round(f, 3), round(f, 4),
                round(round(f, 3), 3), round(f, 2)}

    float_ok = set()
    for f in jfloats:
        float_ok |= fmt_float_rounds(f)

    toks, text = readme_tokens()
    issues = []
    seen = {}
    for num, is_pct in toks:
        key = f"{num}{'%' if is_pct else ''}"
        seen[key] = seen.get(key, 0) + 1
        if "." in num:
            v = float(num)
            ok = (v in float_ok) or (v in pcts) or (is_pct and round(v, 1) in pcts)
        else:
            v = int(num)
            ok = v in ints
            if is_pct:
                ok = ok or (round(float(num), 1) in pcts)
        if not ok and num not in ALLOW:
            idx = text.find(num)
            ctx = re.sub(r"\s+", " ", text[max(0, idx - 40):idx + 30])
            issues.append((key, ctx))

    print("README number audit (README.md vs outputs/*):")
    total = sum(seen.values())
    for key, n in sorted(seen.items(), key=lambda kv: -kv[1] * len(kv[0])):
        print(f"  {key:>10}  x{n}")
    if issues:
        print(f"\n{len(issues)} UNVERIFIED number(s):")
        for key, ctx in issues:
            print(f"  ! {key}   ...{ctx}...")
        return 1
    print(f"\nAll {total} quoted numbers trace back to outputs/ or the "
          "documented allowlist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
