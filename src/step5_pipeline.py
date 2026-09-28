#!/usr/bin/env python3
"""Step 5: two independent primary-emotion predictions, compared.

A) LLM emotion  -- same 100-row seed-42 sample as Step 2 (rows are read from
   step2_predictions.csv so they are IDENTICAL by construction), re-classified
   with the extended prompt {"sentiment": ..., "emotion": ...} where emotion
   must be exactly one of the 8 NRC emotions. The cache key carries a prompt
   version (pv2) so old sentiment-only entries are never reused. The rating
   is never in the prompt.

B) NRC word-list emotion  -- the official NRC Emotion Lexicon (EmoLex) v0.92,
   word-level file (Mohammad & Turney 2013; NRC Canada, non-commercial use).
   Title + text are lowercased and split on simple word tokens; each token's
   8-emotion associations are counted (NRC's positive/negative tags ignored);
   the highest count wins. No matches -> "none"; ties are kept in the
   nrc_emotion column joined with "/" and flagged nrc_tie=True.

Outputs:
    step5_predictions.csv  (step2 columns + llm_emotion, nrc_emotion,
                            nrc_matched_words, nrc_tie, emotions_agree)
    step5_metrics.json     (sentiment metrics re-check + agreement rates,
                            cross-tab, distributions, emotion x sentiment /
                            rating, disagreements grouped by reason)

Usage:
    .venv/bin/python step5_pipeline.py --workers 12
"""

import argparse
import csv
import datetime
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import sentiment_classifier as sc
from step2_pipeline import metrics

HERE = sc.HERE
STEP2_CSV = os.path.join(sc.OUTPUTS, "step2_predictions.csv")
PRED_CSV = os.path.join(sc.OUTPUTS, "step5_predictions.csv")
METRICS_JSON = os.path.join(sc.OUTPUTS, "step5_metrics.json")
NRC_FILE = os.path.join(sc.DATA_DIR, "nrc_lexicon", "NRC-Emotion-Lexicon", "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")

TOKEN_RE = re.compile(r"[a-z]+")
NEGATION_RE = re.compile(r"\b(not|no|never|nothing|n't|nt)\b")
TWEETY_LEN = 60  # "terse" hint: title+text shorter than this


# ---------------------------------------------------------------------------
# NRC lexicon + word-list emotion
# ---------------------------------------------------------------------------
def load_lexicon():
    """word -> set(emotions) from EmoLex v0.92 (8 emotions only; the
    positive/negative rows of the file are ignored, per assignment)."""
    lex = {}
    with open(NRC_FILE, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 3:
                continue
            word, cat, assoc = parts[0], parts[1], parts[2]
            if cat not in sc.EMOTIONS or assoc != "1":
                continue
            lex.setdefault(word, set()).add(cat)
    return lex


def nrc_emotion(title, text, lex):
    """Count 8-emotion hits; return (value, tie, matched_words).

    value: "none", one emotion, or tied emotions joined with "/".
    matched_words: compact "emotion:word1,word2; ..." for explainability.
    """
    counts = {e: 0 for e in sc.EMOTIONS}
    matched = {e: [] for e in sc.EMOTIONS}
    negation_hit = bool(NEGATION_RE.search(f"{title} {text}".lower()))
    for tok in TOKEN_RE.findall(f"{title} {text}".lower()):
        for emo in lex.get(tok, ()):
            counts[emo] += 1
            if len(matched[emo]) < 8 and tok not in matched[emo]:
                matched[emo].append(tok)
    best = max(counts.values())
    if best == 0:
        return "none", False, "", negation_hit, negation_hit
    top = [e for e in sc.EMOTIONS if counts[e] == best]
    tie = len(top) > 1
    value = "/".join(top) if tie else top[0]
    words = "; ".join(f"{e}:{','.join(matched[e])}" for e in sc.EMOTIONS
                      if matched[e])
    return value, tie, words, negation_hit, False


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------
def llm_in(emotion, nrc_value):
    """True if the LLM emotion is (one of) the NRC-selected emotion(s)."""
    if not emotion or not nrc_value or nrc_value == "none":
        return False
    return emotion in nrc_value.split("/")


def emotion_distribution(rows, key):
    """Counter per emotion value ('none' and 'tie'-joined included)."""
    return Counter(r[key] or "unparsed" for r in rows)


def cross_tab(rows):
    """LLM emotion (rows) x NRC emotion (cols) for non-tie, non-none rows."""
    emos = list(sc.EMOTIONS)
    m = {a: {b: 0 for b in emos} for a in emos}
    n_skipped_tie = n_skipped_none = 0
    for r in rows:
        llm, nrc = r["llm_emotion"], r["nrc_emotion"]
        if not llm or llm == "unparsed":
            continue
        if nrc == "none":
            n_skipped_none += 1
            continue
        if "/" in nrc:  # tie: no single NRC emotion to place
            n_skipped_tie += 1
            continue
        m[llm][nrc] += 1
    return m, n_skipped_tie, n_skipped_none


def emotion_vs(rows, key):
    """Per emotion: n, mean star rating, share of positive ground truth."""
    out = {}
    for e in sc.EMOTIONS:
        rs = [r for r in rows if r[key] == e]
        if not rs:
            continue
        mean_rating = sum(r["rating"] for r in rs) / len(rs)
        pos = sum(1 for r in rs if r["ground_truth"] == "positive") / len(rs)
        out[e] = {"n": len(rs), "mean_rating": mean_rating,
                  "positive_share_gt": pos}
    return out


def guess_reason(r):
    """First-pass disagreement reason; superseded by CURATED overrides below."""
    neg, terse = r["_negation_hint"], r["_terse_hint"]
    nrc = r["nrc_emotion"]
    if not r["llm_emotion"]:
        return "llm_emotion_unparsed"
    if nrc == "none":
        return "no_matched_words"
    if "/" in nrc:
        return "tie"
    if neg:
        return "negation"
    if terse:
        return "terse"
    return "context_mismatch"


# Manual curation of the heuristic labels, reviewed row by row (rating, title).
# Each entry: (reason, note). Kept in code so the run is reproducible.
CURATED = {
    (5, "Got it when it said"): ("context_mismatch",  # positive "no problems")
        "'no problems' is positive but contains a negation word; hope pulls anticipation"),
    (1, "Gift card compromised!!!"): ("context_mismatch",
        "positive words (gift, daughter, Christmas) in a scam/complaint review"),
    (1, "not happy my 2% credit didn't automatically apply"): ("negation",
        "'not happy' counted as joy/anticipation; 'credit' counts trust"),
    (1, "Invalid"): ("negation",
        "'didn't work' / 'didn't take' ignored; gift/lesson/trust words win"),
    (4, "Watch what you are buying"): ("negation",
        "'not what I expected' — expectation words feed anticipation"),
    (4, "Nice gift card"): ("context_mismatch",
        "'nice/merry' joy words vs misleading-cutout complaint"),
    (1, "Nominations"): ("context_mismatch",
        "EmoLex oddity: 'serial' -> anticipation in an angry incoherent review"),
    (3, "Supposed to be ready to use."): ("context_mismatch",
        "'assumed/ready/gift' anticipation words vs not-ready complaint"),
}


def curate(r):
    """(rating, title) curated label, else the heuristic guess."""
    return CURATED.get((r["rating"], r["title"]),
                       (guess_reason(r), None))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    sc.load_cache()
    print(f"Endpoint: {sc.BASE_URL} | model: {sc.MODEL} | prompt v{sc.PROMPT_VERSION} "
          f"| cache: {len(sc._cache)} entries")

    # 1) The exact Step-2 sample (same rows, same seed by construction)
    with open(STEP2_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} rows from {STEP2_CSV} (the Step-2 seed-42 sample)")
    for r in rows:
        r["rating"] = int(float(r["rating"]))
        r["correct"] = r["correct"] == "True"

    # 2) LLM emotion pass (new prompt, versioned cache -> live calls)
    start = time.time()
    results = [None] * len(rows)
    n_cache = 0
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(sc.classify_row_ext, r): i
                   for i, r in enumerate(rows)}
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            if results[i]["from_cache"]:
                n_cache += 1
            if done % 25 == 0 or done == len(results):
                print(f"  {done}/{len(results)} "
                      f"({done / (time.time() - start):.1f}/s, {n_cache} cached)")
    sc.save_cache(force=True)
    print(f"LLM pass done in {time.time() - start:.0f}s "
          f"({n_cache}/{len(results)} from cache)")

    # 3) NRC word-list pass (no model calls)
    lex = load_lexicon()
    print(f"NRC lexicon loaded: {len(lex)} words")
    for r in results:
        value, tie, words, neg, _ = nrc_emotion(r["title"], r["text"], lex)
        r["nrc_emotion"] = value
        r["nrc_tie"] = tie
        r["nrc_matched_words"] = words
        r["_negation_hint"] = neg
        r["_terse_hint"] = len(r["title"]) + len(r["text"]) < TWEETY_LEN
        r["llm_emotion"] = r.get("emotion")
        r["emotions_agree"] = bool(
            r["llm_emotion"] and llm_in(r["llm_emotion"], r["nrc_emotion"]))

    # 4) Sentiment re-check (same rows, new prompt)
    sent = metrics([{k: r[k] for k in ("rating", "ground_truth", "prediction")}
                    for r in results])
    n_emo_unparsed = sum(1 for r in results if not r["llm_emotion"])
    n_sent_unparsed = sum(1 for r in results if not r["prediction"])

    # 5) Agreement rates
    n = len(results)
    agree = [r for r in results if r["emotions_agree"]]
    excl = [r for r in results
            if r["llm_emotion"] and r["nrc_emotion"] != "none"]
    excl_agree = [r for r in excl if r["emotions_agree"]]
    strict = [r for r in results if r["llm_emotion"]
              and r["nrc_emotion"] not in ("none",)
              and "/" not in r["nrc_emotion"]
              and r["llm_emotion"] == r["nrc_emotion"]]
    agreement = {
        "n": n,
        "n_agree": len(agree),
        "n_disagree": n - len(agree),
        "overall": len(agree) / n,
        "overall_tie_counts_as_agreement": len(agree) / n,
        "excluding_none": (len(excl_agree) / len(excl)) if excl else None,
        "strict_exact_match": len(strict) / n,
        "n_excluding_none": len(excl),
        "n_strict": len(strict),
    }

    # 6) Cross-tab, distributions, emotion x sentiment / rating
    xtab, n_skip_tie, n_skip_none = cross_tab(results)
    llm_dist = emotion_distribution(results, "llm_emotion")
    nrc_dist = {e: 0 for e in sc.EMOTIONS}
    nrc_dist.update({e: 0 for e in ("none", "tie")})
    for r in results:
        v = r["nrc_emotion"]
        if v == "none":
            nrc_dist["none"] += 1
        elif "/" in v:
            nrc_dist["tie"] += 1
        else:
            nrc_dist[v] += 1
    llm_vs = emotion_vs(results, "llm_emotion")
    nrc_vs = emotion_vs(results, "nrc_emotion")

    # 7) Disagreements grouped by reason (curated list of triggers)
    dis = [r for r in results if not r["emotions_agree"]]
    by_reason = {}
    for r in dis:
        reason, note = curate(r)
        by_reason.setdefault(reason, []).append({
            **{k: r[k] for k in ("rating", "title", "text", "ground_truth",
                                 "prediction", "llm_emotion", "nrc_emotion",
                                 "nrc_matched_words", "nrc_tie")},
            "curated_note": note or "",
        })

    # 8) Outputs
    with open(PRED_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["asin", "rating", "title", "text", "ground_truth",
                    "prediction", "correct", "llm_emotion", "nrc_emotion",
                    "nrc_matched_words", "nrc_tie", "emotions_agree"])
        for r in results:
            w.writerow([r["asin"], r["rating"], r["title"], r["text"],
                        r["ground_truth"], r["prediction"], r["correct"],
                        r["llm_emotion"], r["nrc_emotion"],
                        r["nrc_matched_words"], r["nrc_tie"],
                        r["emotions_agree"]])

    run_details = {
        "prompt_version": sc.PROMPT_VERSION,
        "model": sc.MODEL,
        "endpoint": sc.BASE_URL,
        "seed": 42,
        "sample_size": n,
        "sample_source": STEP2_CSV.rsplit("/", 1)[-1],
        "nrc_lexicon": "NRC Emotion Lexicon v0.92 "
                       "(https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)",
        "dataset_total_reviews": 152410,
        "data_file": sc.DATA_FILE.rsplit("/", 1)[-1],
        "run_date": datetime.date.today().isoformat(),
    }
    mj = {
        "n": n,
        "run_details": run_details,
        "llm_sentiment": {
            **sent,
            "n_emotion_unparsed": n_emo_unparsed,
            "n_sentiment_unparsed": n_sent_unparsed,
        },
        "nrc": {
            "n_none": nrc_dist["none"],
            "n_tie": nrc_dist["tie"],
            "tie_examples": [{"idx": None, "title": r["title"][:60],
                              "nrc": r["nrc_emotion"]}
                             for r in results if r["nrc_tie"]][:5],
        },
        "agreement": agreement,
        "cross_tab": {
            "emotions": list(sc.EMOTIONS),
            "matrix": xtab,
            "n_skipped_nrc_ties": n_skip_tie,
            "n_skipped_nrc_none": n_skip_none,
        },
        "distributions": {
            "llm": dict(sorted(llm_dist.items())),
            "nrc": dict(sorted(nrc_dist.items())),
        },
        "emotion_vs_sentiment": {"llm": llm_vs, "nrc": nrc_vs},
        "disagreements": {
            "n": len(dis),
            "by_reason": {r: rows_ for r, rows_ in sorted(by_reason.items())},
        },
    }
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(mj, f, indent=2)

    # 9) Console report
    print(f"\nWrote {PRED_CSV} ({len(results)} rows) and {METRICS_JSON}")
    print("\n=== STEP 5 REPORT ===")
    a = agreement
    print(f"Agreement: {a['n_agree']}/{a['n']} = {a['overall']:.1%} overall | "
          f"excluding 'none': {a['excluding_none']:.1%} (n={a['n_excluding_none']}) | "
          f"strict exact: {a['strict_exact_match']:.1%}")
    print(f"Emotions unparsed (LLM): {n_emo_unparsed} | "
          f"NRC none: {nrc_dist['none']} | NRC ties: {nrc_dist['tie']}")
    print("LLM distribution:", dict(llm_dist))
    print("NRC distribution:", nrc_dist)
    print(f"Sentiment accuracy moved to: {sent['accuracy']:.4f} "
          f"(Step 2 was 0.9800)")
    print(f"\nDisagreements: {len(dis)} (sarcasm: 0 found)")
    for reason, rs in sorted(by_reason.items()):
        print(f"  {reason}: {len(rs)}")
        for x in rs[:2]:
            note = f" — {x['curated_note']}" if x["curated_note"] else ""
            print(f"      e.g. [{x['rating']}★] {x['title'][:45]!r} "
                  f"LLM={x['llm_emotion']} NRC={x['nrc_emotion']}{note}")


if __name__ == "__main__":
    main()
