#!/usr/bin/env python3
"""Step 7: dataset-level star-rating distribution for the dashboard charts.

Mirrors step6_pipeline.py's dedupe exactly (identical title+text collapsed,
first kept) so the pool size and class counts match the Step-6 run
(deduped_pool_reviews: 133,009). Writes step7_dataset_stats.json with the
full-pool star distribution (counts + %) and the balanced-sample star
distribution (from step6_predictions.csv).
"""
import csv
import json
import os

import sentiment_classifier as sc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTPUTS = os.path.join(ROOT, "outputs")


def star_dist(pool):
    out = {}
    for s in range(1, 6):
        n = sum(1 for r in pool if r["rating"] == s)
        out[str(s)] = {"n": n, "pct": n / len(pool) if pool else 0.0}
    return out


def main():
    seen, pool = set(), []
    raw = 0
    for r in sc.load_reviews():
        raw += 1
        k = (r["title"].lower(), r["text"].lower())
        if k in seen:
            continue
        seen.add(k)
        r["ground_truth"] = sc.ground_truth(r["rating"])
        pool.append(r)
    full = star_dist(pool)
    cls = {l: [r for r in pool if r["ground_truth"] == l]
           for l in ("negative", "neutral", "positive")}

    sample = list(csv.DictReader(open(os.path.join(OUTPUTS, "step6_predictions.csv"),
                                      encoding="utf-8")))
    for r in sample:
        r["rating"] = int(float(r["rating"]))
    sample_dist = star_dist(sample)

    # lexicon coverage: words with >=1 of the 8 emotions (used by the README's
    # EmoLex-coverage numbers; mirrors step5_pipeline's loading rules)
    lex_file = os.path.join(DATA_DIR := os.path.join(ROOT := os.path.dirname(HERE), "data"),
                            "nrc_lexicon", "NRC-Emotion-Lexicon",
                            "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")
    lex_with = set()
    lex_words = set()
    try:
        with open(lex_file, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                word, category, flag = parts[0], parts[1], parts[2]
                lex_words.add(word)
                if category in sc.EMOTIONS and flag == "1":
                    lex_with.add(word)
    except OSError:
        lex_words, lex_with = set(), set()

    ds = {
        "n_dataset_raw": raw,
        "n_deduped": len(pool),
        "class_pool_sizes": {l: len(cls[l]) for l in cls},
        "stars_full": full,
        "stars_sample": sample_dist,
        "lexicon_words_total": len(lex_words),
        "lexicon_words_with_emotion": len(lex_with),
    }
    with open(os.path.join(OUTPUTS, "step7_dataset_stats.json"), "w",
              encoding="utf-8") as f:
        json.dump(ds, f, indent=2)
    print(f"Deduped pool: {len(pool)} | " + " | ".join(f"{l}: {len(cls[l])}"
          for l in cls))
    for s in range(1, 6):
        a, b = full[str(s)], sample_dist[str(s)]
        print(f"{s}★  full {a['n']:>7,} ({a['pct']:.1%})   "
              f"sample {b['n']:>3} ({b['pct']:.1%})")


if __name__ == "__main__":
    main()
