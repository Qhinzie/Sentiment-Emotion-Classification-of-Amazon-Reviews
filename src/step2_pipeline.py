#!/usr/bin/env python3
"""Step 2: binary LLM sentiment classification on a fixed random sample.

Samples N=100 reviews (random but seeded, default seed 42) from ALL Gift_Cards
reviews -- 3-star reviews are NOT excluded -- and classifies each as
POSITIVE / NEGATIVE from title + text only via the LLM course endpoint
(OpenAI-compatible). The star rating is used ONLY as ground truth
(rating >= 4 -> positive, else negative) and never shown to the LLM.

Reuses the engine from sentiment_classifier.py (same prompt, cache, workers).

Outputs:
    step2_predictions.csv  (asin, rating, title, text, ground_truth, prediction, correct)
    step2_errors.csv       (rating, title, text, ground_truth, prediction)
    step2_metrics.json     (class balance, accuracy, always-positive baseline,
                            confusion matrix, precision/recall/F1 per class,
                            balanced accuracy, macro-F1, accuracy by rating)

Usage:
    .venv/bin/python step2_pipeline.py --workers 12
    .venv/bin/python step2_pipeline.py --rows 200 --seed 7
"""

import argparse
import csv
import datetime
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import sentiment_classifier as sc

HERE = sc.HERE
PRED_CSV = os.path.join(sc.OUTPUTS, "step2_predictions.csv")
ERR_CSV = os.path.join(sc.OUTPUTS, "step2_errors.csv")
METRICS_JSON = os.path.join(sc.OUTPUTS, "step2_metrics.json")


def metrics(results, run_details=None):
    """All requested metrics from classified rows (dicts with rating/gt/pred)."""
    n = len(results)
    gt_pos = sum(1 for r in results if r["ground_truth"] == "positive")
    gt_neg = n - gt_pos

    tp = sum(1 for r in results if r["ground_truth"] == "positive" and r["prediction"] == "positive")
    fp = sum(1 for r in results if r["ground_truth"] == "negative" and r["prediction"] == "positive")
    tn = sum(1 for r in results if r["ground_truth"] == "negative" and r["prediction"] == "negative")
    fn = n - tp - fp - tn

    def safe(dividend, divisor):
        return dividend / divisor if divisor else None

    prec_pos, rec_pos = safe(tp, tp + fp), safe(tp, tp + fn)
    prec_neg, rec_neg = safe(tn, tn + fn), safe(tn, tn + fp)
    f1 = lambda p, r: safe(2 * p * r, p + r) if p is not None and r is not None else None
    f1_pos, f1_neg = f1(prec_pos, rec_pos), f1(prec_neg, rec_neg)

    m = {
        "n": n,
        "n_ground_truth_positive": gt_pos,
        "n_ground_truth_negative": gt_neg,
        "class_balance": {"positive": gt_pos / n, "negative": gt_neg / n},
        "baseline_always_predict_positive": gt_pos / n,
        "accuracy": safe(tp + tn, n),
        "confusion_matrix": {
            "true_positive": tp, "false_positive": fp,
            "false_negative": fn, "true_negative": tn,
        },
        "per_class": {
            "positive": {"precision": prec_pos, "recall": rec_pos, "f1": f1_pos},
            "negative": {"precision": prec_neg, "recall": rec_neg, "f1": f1_neg},
        },
        "balanced_accuracy": safe(rec_pos + rec_neg, 2),
        "macro_f1": safe(f1_pos + f1_neg, 2) if f1_pos is not None and f1_neg is not None else None,
        "accuracy_by_rating": {},
        "n_unparsed": sum(1 for r in results if r["prediction"] is None),
    }
    for rating in (1, 2, 3, 4, 5):
        rows = [r for r in results if r["rating"] == rating]
        if rows:
            correct = sum(1 for r in rows if r["prediction"] == r["ground_truth"])
            m["accuracy_by_rating"][str(rating)] = {
                "n": len(rows), "correct": correct, "accuracy": correct / len(rows),
            }
    if run_details:
        m["run_details"] = run_details
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=100, help="sample size (default 100)")
    ap.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    sc.load_cache()
    print(f"Endpoint: {sc.BASE_URL} | model: {sc.MODEL} | cache: {len(sc._cache)} entries")

    # 1) Draw the fixed sample from ALL reviews (3-star included, no dedupe)
    reviews = list(sc.load_reviews())
    rng = random.Random(args.seed)
    sample = rng.sample(reviews, min(args.rows, len(reviews)))
    print(f"Sampled {len(sample)} of {len(reviews)} reviews (seed {args.seed}, "
          f"3-star reviews included)")

    # 2) Classify in parallel, preserving input order
    start = time.time()
    results = [None] * len(sample)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(sc.classify_row, rec): i for i, rec in enumerate(sample)}
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            if done % 25 == 0 or done == len(sample):
                print(f"  {done}/{len(sample)} ({done / (time.time() - start):.1f} reviews/s)")
    sc.save_cache(force=True)

    for r in results:
        r["ground_truth"] = sc.ground_truth(r["rating"], binary=True)
        r["correct"] = (r["prediction"] == r["ground_truth"])

    # 3) step2_predictions.csv
    with open(PRED_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["asin", "rating", "title", "text", "ground_truth", "prediction", "correct"])
        for r in results:
            w.writerow([r["asin"], r["rating"], r["title"], r["text"],
                        r["ground_truth"], r["prediction"], r["correct"]])

    # 4) step2_errors.csv (misclassified only, sorted by rating)
    errors = [r for r in results if not r["correct"]]
    errors.sort(key=lambda r: (r["rating"], r["title"]))
    with open(ERR_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rating", "title", "text", "ground_truth", "prediction"])
        for r in errors:
            w.writerow([r["rating"], r["title"], r["text"], r["ground_truth"], r["prediction"]])

    # 5) step2_metrics.json
    run_details = {
        "model": sc.MODEL,
        "endpoint": sc.BASE_URL,
        "seed": args.seed,
        "sample_size": len(results),
        "dataset_total_reviews": len(reviews),
        "data_file": sc.DATA_FILE.rsplit("/", 1)[-1],
        "run_date": datetime.date.today().isoformat(),
    }
    m = metrics(results, run_details)
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)

    # 6) Console report
    print(f"\nWrote {PRED_CSV} ({len(results)} rows), {ERR_CSV} ({len(errors)} errors), "
          f"{METRICS_JSON}")
    print("\n=== STEP 2 REPORT ===")
    print(f"Sample: n={m['n']} | ground truth positive={m['n_ground_truth_positive']} "
          f"({m['class_balance']['positive']:.1%}), "
          f"negative={m['n_ground_truth_negative']} ({m['class_balance']['negative']:.1%})")
    print(f"Accuracy: {m['accuracy']:.4f} | always-positive baseline: "
          f"{m['baseline_always_predict_positive']:.4f}")
    cm = m["confusion_matrix"]
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(f"                predicted pos   predicted neg")
    print(f"actual pos        {cm['true_positive']:>5}          {cm['false_negative']:>5}")
    print(f"actual neg        {cm['false_positive']:>5}          {cm['true_negative']:>5}")
    for cls in ("positive", "negative"):
        p = m["per_class"][cls]
        print(f"{cls:>8}: precision={p['precision']:.4f} recall={p['recall']:.4f} "
              f"f1={p['f1']:.4f}")
    print(f"balanced_accuracy={m['balanced_accuracy']:.4f} macro_f1={m['macro_f1']:.4f}")
    print("Accuracy by star rating:")
    for rating, v in m["accuracy_by_rating"].items():
        print(f"  {rating} star: {v['correct']}/{v['n']} = {v['accuracy']:.2%}")


if __name__ == "__main__":
    main()
