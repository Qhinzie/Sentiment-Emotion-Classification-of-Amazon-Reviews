#!/usr/bin/env python3
"""Step 6: three-class (positive/neutral/negative) scoring, balanced sampling.

Ground truth (Step-6 spec): rating 4-5 -> POSITIVE, 3 -> NEUTRAL, 1-2 -> NEGATIVE.
Sampling: from the full DEDUPED file (identical title+text collapsed, first kept),
a balanced draw of 50 reviews per class (150 total) with seed 42. The sampled
review IDs are saved to step6_sample.csv so the set is reproducible.

Prompt: three classes + the Step-5 emotion, JSON {"sentiment": ..., "emotion": ...}
with an explicit NEUTRAL definition and boundaries. The rating is never shown.
Cache key carries prompt version pv3 so old two-class answers are never reused.

Outputs:
    step6_sample.csv      (the exact 150 sampled reviews / IDs)
    step6_predictions.csv (sample + ground_truth, prediction, correct,
                           llm_emotion, nrc_emotion, nrc_matched_words,
                           nrc_tie, emotions_agree)
    step6_errors.csv      (misclassified only)
    step6_metrics.json    (3x3 confusion, per-class P/R/F1, macro-F1, accuracy,
                           33% baseline, true-NEUTRAL destination, accuracy by
                           star, emotions + agreement + emotion by true class,
                           Step-2 vs Step-6 comparison incl. binary collapse)

Usage:
    .venv/bin/python step6_pipeline.py --workers 12
"""

import argparse
import csv
import datetime
import json
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import sentiment_classifier as sc
from step5_pipeline import load_lexicon, nrc_emotion, llm_in, emotion_distribution

HERE = sc.HERE
STEP2_METRICS = os.path.join(sc.OUTPUTS, "step2_metrics.json")
SAMPLE_CSV = os.path.join(sc.OUTPUTS, "step6_sample.csv")
PRED_CSV = os.path.join(sc.OUTPUTS, "step6_predictions.csv")
ERR_CSV = os.path.join(sc.OUTPUTS, "step6_errors.csv")
METRICS_JSON = os.path.join(sc.OUTPUTS, "step6_metrics.json")

LABELS = ("negative", "neutral", "positive")

# Curated notes for the most instructive errors (keyed by exact title). The
# auto note explains the pattern; these explain the specific text. Reviewed
# manually against the full review text.
NOTES6 = {
    "Box was damaged upon arrival": ("The card works, but the damaged premium box "
        "dominates the text — reads as a complaint despite 3 stars."),
    "Watch your packages!": "Delivery-failure story dominates; the 3-star rating is the only neutral signal.",
    "Works online, but not in the store": "Genuinely mixed (works online, fails in store) — the failure won.",
    "I was disappointed it took so long to ": "Slow-delivery complaint.",
    "Too slow for 2016  :/": "'Would be 5 stars if delivery were fast' — praise framed through a complaint.",
    "Carrabba's": "Shipping praised, restaurant meal panned — the complaint dominates.",
    "Flimsy": "Genuinely mixed ('not well put together, but for the price…') — leaned negative.",
    "Misleading Display": "Lukewarm ('It's OK if you know how to get around') — leaned negative.",
    "Look cool but function is another ques": "Title is literally mixed; the blurry-vision complaint drove the call.",
    "Gift card box": "'Otherwise great gift giving' — mild praise with a gripe; chose positive.",
    "Ecard best": "Comparative praise ('Ecard is better…') read as positive.",
    "Ange": "Text is clearly positive; the 3-star rating is the only neutral signal.",
    "Three Stars": "Typo-laden short praise ('Perry good') read as positive.",
    "Ahh": "Barely a review ('Ahh') — treated the nod as positive.",
    "Son in law gift": "Purely factual sentence; the positive signal exists only in the rating.",
    "Everything ok": "Lukewarm phrase read as neutral despite 5 stars.",
    "Nothing": "Terse 'Loved it' contradicts the 2-star rating; the model followed the words.",
}


# ---------------------------------------------------------------------------
# Metrics (class-agnostic, 3 classes)
# ---------------------------------------------------------------------------
def metrics3(results, run_details=None):
    n = len(results)
    labels = sorted({r["ground_truth"] for r in results})
    cnt = Counter(r["ground_truth"] for r in results)
    cm = {}
    for r in results:
        cm[(r["ground_truth"], r["prediction"])] = cm.get((r["ground_truth"], r["prediction"]), 0) + 1

    per = {}
    for a in labels:
        tp = cm.get((a, a), 0)
        act = sum(cm.get((a, p), 0) for p in labels + [None])
        prd = sum(cm.get((g, a), 0) for g in labels)
        prec = tp / prd if prd else None
        rec = tp / act if act else None
        f1 = (2 * prec * rec / (prec + rec)) if (prec is not None and rec is not None and prec + rec) else None
        per[a] = dict(actual=act, predicted=prd, correct=tp, precision=prec, recall=rec, f1=f1)

    correct_n = sum(cm.get((a, a), 0) for a in labels)
    majority = max(labels, key=lambda l: (cnt[l], l)) if labels else None
    by_rating = {}
    for star in range(1, 6):
        rs = [r for r in results if r["rating"] == star]
        if rs:
            ok = sum(1 for r in rs if r["prediction"] == r["ground_truth"])
            by_rating[str(star)] = dict(n=len(rs), correct=ok, accuracy=ok / len(rs))
        else:
            by_rating[str(star)] = dict(n=0)
    # where true NEUTRAL went
    neutral_dest = Counter(r["prediction"] for r in results if r["ground_truth"] == "neutral")
    m = {
        "n": n,
        "class_balance": {l: cnt[l] / n for l in labels},
        "n_by_class": dict(cnt),
        "baseline_always_predict_majority": (cnt[majority] / n) if majority else None,
        "accuracy": correct_n / n if n else None,
        "confusion_matrix": {
            "labels": labels,
            "matrix": {g: {p: cm.get((g, p), 0) for p in labels} for g in labels},
        },
        "per_class": {l: {k: per[l][k] for k in ("precision", "recall", "f1")}
                      for l in labels},
        "balanced_accuracy": (sum(per[l]["recall"] for l in labels) / len(labels)
                              if per and all(per[l]["recall"] is not None for l in labels) else None),
        "macro_f1": (sum(per[l]["f1"] for l in labels) / len(labels)
                     if per and all(per[l]["f1"] is not None for l in labels) else None),
        "accuracy_by_rating": by_rating,
        "true_neutral_destination": {p: neutral_dest.get(p, 0)
                                     for p in ("positive", "neutral", "negative")},
        "n_unparsed": sum(1 for r in results if r["prediction"] is None),
    }
    if run_details:
        m["run_details"] = run_details
    return m


def error_pattern3(r):
    gt, pr = r["ground_truth"], r["prediction"]
    if gt == "neutral":
        return f"3-star NEUTRAL pulled to {pr.upper()}"
    if gt == "positive":
        return "POSITIVE review read as " + pr.upper()
    return "NEGATIVE review read as " + pr.upper()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-class", type=int, default=50)
    args = ap.parse_args()

    sc.load_cache()
    print(f"Endpoint: {sc.BASE_URL} | model: {sc.MODEL} | prompt v{sc.PROMPT_VERSION} "
          f"| cache: {len(sc._cache)} entries")

    # 1) Full deduped pool (identical title+text collapsed, first kept)
    seen, pool = set(), []
    for r in sc.load_reviews():
        k = (r["title"].lower(), r["text"].lower())
        if k in seen:
            continue
        seen.add(k)
        r["ground_truth"] = sc.ground_truth(r["rating"])
        pool.append(r)
    pools = {l: [r for r in pool if r["ground_truth"] == l] for l in LABELS}
    print(f"Deduped pool: {len(pool)} reviews | "
          + " | ".join(f"{l}: {len(pools[l])}" for l in LABELS))

    # 2) Balanced draw, seed fixed; save the IDs
    rng = random.Random(args.seed)
    sample = []
    for l in LABELS:
        sample += rng.sample(pools[l], min(args.per_class, len(pools[l])))
    rng.shuffle(sample)
    print(f"Balanced sample: {len(sample)} (50 per class, seed {args.seed})")
    with open(SAMPLE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["asin", "rating", "title", "text", "ground_truth"])
        for r in sample:
            w.writerow([r["asin"], r["rating"], r["title"], r["text"], r["ground_truth"]])

    # 3) LLM pass (pv3 cache key -> live calls)
    start = time.time()
    results = [None] * len(sample)
    n_cache = done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool_exec:
        futures = {pool_exec.submit(sc.classify_row_ext, r): i for i, r in enumerate(sample)}
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            n_cache += results[i]["from_cache"]
            if done % 25 == 0 or done == len(results):
                print(f"  {done}/{len(results)} ({done / (time.time() - start):.1f}/s)")
    sc.save_cache(force=True)
    print(f"LLM pass done in {time.time() - start:.0f}s ({n_cache} from cache)")

    for r in results:
        r["correct"] = (r["prediction"] == r["ground_truth"])

    # 4) NRC pass
    from step5_pipeline import guess_reason
    lex = load_lexicon()
    for r in results:
        value, tie, words, neg_hint, _ = nrc_emotion(r["title"], r["text"], lex)
        r["nrc_emotion"] = value
        r["nrc_tie"] = tie
        r["nrc_matched_words"] = words
        r["llm_emotion"] = r.get("emotion")
        r["emotions_agree"] = bool(r["llm_emotion"] and llm_in(r["llm_emotion"], value))
        r["_negation_hint"] = neg_hint
        r["_terse_hint"] = len(r["title"]) + len(r["text"]) < 60
    # emotion-disagreement reasons (auto labels; step5 curated them — the
    # step-6 focus is classification errors, so these stay uncurated)
    dis_reasons = {}
    for r in results:
        if r["emotions_agree"]:
            continue
        reason = guess_reason(r)
        dis_reasons.setdefault(reason, []).append({
            **{k: r[k] for k in ("rating", "title", "text", "llm_emotion",
                                 "nrc_emotion", "nrc_matched_words", "nrc_tie")},
            "curated_note": "",
        })

    # 5) Metrics
    run_details = {
        "prompt_version": sc.PROMPT_VERSION,
        "model": sc.MODEL,
        "endpoint": sc.BASE_URL,
        "seed": args.seed,
        "sample_size": len(results),
        "sample_design": f"balanced {args.per_class} per class (deduped pool)",
        "dataset_total_reviews": 152410,
        "deduped_pool_reviews": len(pool),
        "class_pool_sizes": {l: len(pools[l]) for l in LABELS},
        "nrc_lexicon": "NRC Emotion Lexicon v0.92 (saifmohammad.com)",
        "data_file": sc.DATA_FILE.rsplit("/", 1)[-1],
        "run_date": datetime.date.today().isoformat(),
    }
    sent = metrics3(results, run_details)
    sent["n_emotion_unparsed"] = sum(1 for r in results if not r["llm_emotion"])

    # emotions: agreement + distributions + by true class
    n = len(results)
    agree = [r for r in results if r["emotions_agree"]]
    excl = [r for r in results if r["llm_emotion"] and r["nrc_emotion"] != "none"]
    excl_agree = [r for r in excl if r["emotions_agree"]]
    strict = [r for r in results if r["llm_emotion"]
              and r["nrc_emotion"] not in ("none",) and "/" not in r["nrc_emotion"]
              and r["llm_emotion"] == r["nrc_emotion"]]
    agreement = {
        "n": n, "n_agree": len(agree), "n_disagree": n - len(agree),
        "overall": len(agree) / n,
        "overall_tie_counts_as_agreement": len(agree) / n,
        "excluding_none": (len(excl_agree) / len(excl)) if excl else None,
        "strict_exact_match": len(strict) / n,
        "n_excluding_none": len(excl), "n_strict": len(strict),
    }
    emos = list(sc.EMOTIONS)
    xtab = {a: {b: 0 for b in emos} for a in emos}
    n_skip_tie = n_skip_none = 0
    for r in results:
        llm, nrc = r["llm_emotion"], r["nrc_emotion"]
        if not llm:
            continue
        if nrc == "none":
            n_skip_none += 1
            continue
        if "/" in nrc:
            n_skip_tie += 1
            continue
        xtab[llm][nrc] += 1
    llm_dist = emotion_distribution(results, "llm_emotion")
    nrc_dist = {e: 0 for e in emos + ["none", "tie"]}
    for r in results:
        v = r["nrc_emotion"]
        nrc_dist["tie" if "/" in v else v] += 1
    by_class = {"llm": {e: Counter() for e in emos + ["unparsed"]},
                "nrc": {e: Counter() for e in emos + ["none", "tie"]}}
    for r in results:
        by_class["llm"][r["llm_emotion"] or "unparsed"][r["ground_truth"]] += 1
        v = r["nrc_emotion"]
        by_class["nrc"]["tie" if "/" in v else v][r["ground_truth"]] += 1

    # 6) Step-2 comparison + binary collapse of Step 6
    try:
        with open(STEP2_METRICS, encoding="utf-8") as f:
            m2 = json.load(f)
    except (OSError, json.JSONDecodeError):
        m2 = None
    bin_rows = [r for r in results if r["ground_truth"] in ("positive", "negative")]
    n_neg_pred_neutral = sum(1 for r in bin_rows if r["prediction"] == "neutral")
    tp = sum(1 for r in bin_rows if r["ground_truth"] == "positive" and r["prediction"] == "positive")
    fn_ = sum(1 for r in bin_rows if r["ground_truth"] == "positive" and r["prediction"] != "positive")
    tn = sum(1 for r in bin_rows if r["ground_truth"] == "negative" and r["prediction"] == "negative")
    fp_ = sum(1 for r in bin_rows if r["ground_truth"] == "negative" and r["prediction"] != "negative")
    acc_bin = (tp + tn) / len(bin_rows)
    rec_pos = tp / (tp + fn_) if (tp + fn_) else None
    rec_neg = tn / (tn + fp_) if (tn + fp_) else None
    prc_pos = tp / (tp + fp_) if (tp + fp_) else None
    prc_neg = tn / (tn + fn_) if (tn + fn_) else None
    def f1m(p, r):
        return 2 * p * r / (p + r) if (p is not None and r is not None and p + r) else None
    vs_step2 = {
        "step2_imbalanced_binary": {
            "n": m2.get("n") if m2 else None,
            "n_by_class": {"positive": m2.get("n_ground_truth_positive") if m2 else None,
                           "negative": m2.get("n_ground_truth_negative") if m2 else None},
            "accuracy": m2.get("accuracy") if m2 else None,
            "balanced_accuracy": m2.get("balanced_accuracy") if m2 else None,
            "macro_f1": m2.get("macro_f1") if m2 else None,
            "per_class": m2.get("per_class") if m2 else None,
        },
        "step6_balanced_three_class": {
            "n": n, "accuracy": sent["accuracy"],
            "balanced_accuracy": sent["balanced_accuracy"], "macro_f1": sent["macro_f1"],
            "per_class": sent["per_class"],
        },
        "step6_binary_collapsed": {
            "n": len(bin_rows),
            "n_by_class": {l: sum(1 for r in bin_rows if r["ground_truth"] == l)
                           for l in ("positive", "negative")},
            "n_predictions_neutral": n_neg_pred_neutral,
            "accuracy": acc_bin,
            "confusion": {"true_positive": tp, "false_positive": fp_,
                          "false_negative": fn_, "true_negative": tn},
            "per_class": {
                "positive": {"precision": prc_pos, "recall": rec_pos, "f1": f1m(prc_pos, rec_pos)},
                "negative": {"precision": prc_neg, "recall": rec_neg, "f1": f1m(prc_neg, rec_neg)},
            },
        },
        "note": ("Step 2: binary prompt, imbalanced sample (88 pos / 12 neg). Step 6: "
                 "three-class prompt, balanced sample; the binary collapse keeps only "
                 "POSITIVE/NEGATIVE ground-truth rows and counts a 'neutral' prediction "
                 "as wrong. Recall is the fair comparison across prevalence; raw "
                 "accuracy is prevalence-sensitive."),
    }
    if m2:
        p2 = m2.get("per_class", {})
        c6 = vs_step2["step6_binary_collapsed"]["per_class"]
        vs_step2["comparison"] = {
            "positive_recall": {"step2": p2.get("positive", {}).get("recall"),
                                "step6_collapsed": c6["positive"]["recall"]},
            "negative_recall": {"step2": p2.get("negative", {}).get("recall"),
                                "step6_collapsed": c6["negative"]["recall"]},
            "positive_f1": {"step2": p2.get("positive", {}).get("f1"),
                            "step6_collapsed": c6["positive"]["f1"]},
            "negative_f1": {"step2": p2.get("negative", {}).get("f1"),
                            "step6_collapsed": c6["negative"]["f1"]},
            "accuracy": {"step2_imbalanced": m2.get("accuracy"),
                         "step6_balanced_three_class": sent["accuracy"],
                         "step6_binary_collapsed": acc_bin},
        }

    # 7) Errors grouped by pattern
    AUTO_NOTE = {
        ("neutral", "negative"): "Text reads predominantly as a complaint despite the 3-star rating.",
        ("neutral", "positive"): "Text reads favorably despite the 3-star rating.",
        ("positive", "neutral"): "No explicit opinion in the text (factual or terse); the positive signal is only the rating.",
        ("negative", "positive"): "Terse text reads positive; the negative signal is only the rating.",
        ("negative", "neutral"): "Complaint softened into a neutral read.",
        ("positive", "negative"): "Complaint-heavy text outshone the positive rating.",
    }
    errors = [r for r in results if not r["correct"]]
    err_groups = {}
    for r in errors:
        note = NOTES6.get(r["title"], AUTO_NOTE.get((r["ground_truth"], r["prediction"]), ""))
        err_groups.setdefault(error_pattern3(r), []).append({
            **{k: r[k] for k in ("asin", "rating", "title", "text", "ground_truth",
                                 "prediction", "llm_emotion", "nrc_emotion",
                                 "nrc_matched_words")},
            "note": note,
        })

    # 8) Outputs
    with open(PRED_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["asin", "rating", "title", "text", "ground_truth", "prediction",
                    "correct", "llm_emotion", "nrc_emotion", "nrc_matched_words",
                    "nrc_tie", "emotions_agree"])
        for r in results:
            w.writerow([r["asin"], r["rating"], r["title"], r["text"], r["ground_truth"],
                        r["prediction"], r["correct"], r["llm_emotion"],
                        r["nrc_emotion"], r["nrc_matched_words"], r["nrc_tie"],
                        r["emotions_agree"]])
    with open(ERR_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["asin", "rating", "title", "text", "ground_truth", "prediction"])
        for r in errors:
            w.writerow([r["asin"], r["rating"], r["title"], r["text"],
                        r["ground_truth"], r["prediction"]])

    mj = {
        "n": n,
        "run_details": run_details,
        "llm_sentiment": sent,
        "nrc": {"n_none": nrc_dist["none"], "n_tie": nrc_dist["tie"]},
        "agreement": agreement,
        "cross_tab": {"emotions": emos, "matrix": xtab,
                      "n_skipped_nrc_ties": n_skip_tie, "n_skipped_nrc_none": n_skip_none},
        "distributions": {"llm": dict(sorted(llm_dist.items())),
                          "nrc": dict(sorted(nrc_dist.items()))},
        "emotion_vs_sentiment": {},
        "emotion_by_true_class": {
            "llm": {e: dict(c) for e, c in by_class["llm"].items() if c},
            "nrc": {e: dict(c) for e, c in by_class["nrc"].items() if c},
        },
        "disagreements": {"n": n - len(agree), "by_reason": dis_reasons},
        "errors": {"n": len(errors), "by_pattern": err_groups},
        "vs_step2": vs_step2,
    }
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(mj, f, indent=2)

    # 9) Console report
    print(f"\nWrote {SAMPLE_CSV}, {PRED_CSV}, {ERR_CSV}, {METRICS_JSON}")
    print("\n=== STEP 6 REPORT ===")
    print(f"Accuracy: {sent['accuracy']:.4f} | 33% baseline: "
          f"{sent['baseline_always_predict_majority']:.4f} | macro-F1: {sent['macro_f1']:.4f}")
    cm = sent["confusion_matrix"]["matrix"]
    print("3x3 confusion (rows=true, cols=pred):")
    print("          " + "  ".join(f"{p:>9}" for p in LABELS))
    for g in LABELS:
        print(f"{g:>9}" + "".join(f"{cm[g][p]:>10}" for p in LABELS))
    for l in LABELS:
        p = sent["per_class"][l]
        print(f"{l:>8}: P={p['precision']:.3f} R={p['recall']:.3f} F1={p['f1']:.3f}")
    print("True NEUTRAL went:", sent["true_neutral_destination"])
    print("Accuracy by star:", {k: v["accuracy"] for k, v in sent["accuracy_by_rating"].items()})
    print(f"Emotion agreement: {agreement['overall']:.1%} overall | "
          f"{agreement['excluding_none']:.1%} excl-none (n={agreement['n_excluding_none']})")
    if m2:
        c = vs_step2["comparison"]
        print(f"POS recall: step2 {c['positive_recall']['step2']:.3f} -> step6 {c['positive_recall']['step6_collapsed']:.3f}")
        print(f"NEG recall: step2 {c['negative_recall']['step2']:.3f} -> step6 {c['negative_recall']['step6_collapsed']:.3f}")
    print(f"\nErrors: {len(errors)}")
    for pat, rs in err_groups.items():
        print(f"  {pat}: {len(rs)}")
        for x in rs[:2]:
            print(f"      [{x['rating']}★] {x['title'][:45]!r} -> {x['prediction']}")


if __name__ == "__main__":
    main()
