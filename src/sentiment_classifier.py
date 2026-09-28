#!/usr/bin/env python3
"""Binary sentiment classifier for Amazon 2023 reviews (Gift Cards category).

Classifies each review as POSITIVE or NEGATIVE based on its title + text,
using an LLM served through an OpenAI-compatible endpoint (course endpoint:
http://dobolyi.com:9000/v1, model DeepSeek-V4-Flash-0731).

Ground truth for evaluation is derived from the star rating (binary mapping):
    rating >= 4  -> POSITIVE
    rating <= 3  -> NEGATIVE   (1-, 2- and 3-star reviews are all negative)

The rating is ONLY used for evaluation -- the prompt sent to the LLM contains
title and text, never the rating.

Also imported by step2_pipeline.py (fixed 100-review sample + full metrics).

Usage:
    .venv/bin/python sentiment_classifier.py --rows 500            # random sample (seed 42)
    .venv/bin/python sentiment_classifier.py --rows 500 --seed 7   # different sample
    .venv/bin/python sentiment_classifier.py --all --workers 16    # full file (~152k reviews)
    .venv/bin/python sentiment_classifier.py --rows 500 --no-dedupe  # sample raw rows

Output: predictions.csv with columns
    asin, rating, title, text, ground_truth, prediction, correct

Endpoint / model can be overridden via env vars LLM_BASE_URL, LLM_MODEL,
or --base-url / --model flags. The API key is read ONLY from the LLM_API_KEY
environment variable — never hardcoded, never defaulted.
"""

import argparse
import csv
import gzip
import hashlib
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)              # repo root
OUTPUTS = os.path.join(ROOT, "outputs")   # committed prediction/metrics artifacts
DATA_DIR = os.path.join(ROOT, "data")     # downloaded data + lexicon (gitignored)
DATA_FILE = os.path.join(DATA_DIR, "Gift_Cards.jsonl.gz")
CACHE_FILE = os.path.join(ROOT, ".llm_classifier_cache.json")
CSV_FILE = os.path.join(ROOT, "predictions.csv")

BASE_URL = os.environ.get("LLM_BASE_URL", "http://dobolyi.com:9000/v1")
API_KEY = os.environ.get("LLM_API_KEY") or ""
MODEL = os.environ.get("LLM_MODEL", "DeepSeek-V4-Flash-0731")


def _api_key() -> str:
    """The API key comes ONLY from the LLM_API_KEY environment variable.
    No hardcoded value and no default anywhere."""
    key = os.environ.get("LLM_API_KEY")
    if not key:
        raise RuntimeError(
            "LLM_API_KEY is not set. Refusing to fall back to any default: "
            "export LLM_API_KEY=... (see .env.example / README) before running."
        )
    return key

MAX_TOKENS = 1024   # reasoning model spends tokens on hidden thinking first;
                    # small budgets (e.g. 8) return empty `content`.
RETRIES = 3
CACHE_SAVE_EVERY = 25   # persist cache to disk every N new labels

# The 8 NRC emotions used for the primary-emotion prediction (Step 5).
EMOTIONS = ("anger", "anticipation", "disgust", "fear", "joy",
            "sadness", "surprise", "trust")
# Bump when the prompt changes so old cache entries are never reused.
# pv2 = binary + emotion (Step 5); pv3 = three-class + emotion (Step 6).
PROMPT_VERSION = "pv3"


def build_prompt(title: str, text: str) -> str:
    """The ONLY thing sent to the LLM. The rating is never interpolated here.

    Three classes (Step 6): POSITIVE / NEUTRAL / NEGATIVE with an explicit
    definition of NEUTRAL and its boundaries, plus the Step-5 emotion.
    """
    return (
        'Classify the sentiment of this Amazon product review as "positive", "neutral" '
        'or "negative", and its primary emotion. The emotion MUST be exactly one of: '
        "anger, anticipation, disgust, fear, joy, sadness, surprise, trust.\n\n"
        "How to decide:\n"
        '- "positive": clearly favorable, enthusiastic, or satisfied. Mostly praise with '
        "a minor gripe is still positive.\n"
        '- "negative": clearly unfavorable, disappointed, or angry. Mostly complaint with '
        "a small positive is still negative.\n"
        '- "neutral": genuinely mixed (real pros and cons with neither dominating), '
        'lukewarm or indifferent ("it\'s okay", "does the job"), or purely factual '
        "with no opinion.\n\n"
        f"Title: {title}\n"
        f"Text: {text}\n\n"
        'Return ONLY a JSON object like {"sentiment": "positive", "emotion": "joy"} '
        '(or "neutral", or "negative", each with one of the 8 emotions listed above).'
    )


# ---------------------------------------------------------------------------
# Cache: "<model>|<sha256(title\\ntext)>" -> "positive"|"negative"
# ---------------------------------------------------------------------------
_cache: dict = {}
_cache_lock = threading.Lock()
_cache_dirty = 0


def _cache_key(title: str, text: str) -> str:
    digest = hashlib.sha256(f"{title}\n{text}".encode("utf-8")).hexdigest()
    # prompt version prefixes the key so old sentiment-only entries stay frozen
    return f"{PROMPT_VERSION}|{MODEL}|{digest}"


def load_cache():
    global _cache
    try:
        with open(CACHE_FILE) as f:
            _cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        _cache = {}


def save_cache(force: bool = False):
    """Merge-write the cache atomically (thread-safe)."""
    global _cache_dirty
    if not force and _cache_dirty < CACHE_SAVE_EVERY:
        return
    with _cache_lock:
        tmp = CACHE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(_cache, f)
        os.replace(tmp, CACHE_FILE)
        _cache_dirty = 0


def cache_hit(key: str):
    with _cache_lock:
        return _cache.get(key)


def cache_store(key: str, label: str):
    global _cache_dirty
    with _cache_lock:
        _cache[key] = label
        _cache_dirty += 1


# ---------------------------------------------------------------------------
# LLM call (OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------
_thread_local = threading.local()


def _client():
    if not hasattr(_thread_local, "client"):
        from openai import OpenAI
        _thread_local.client = OpenAI(base_url=BASE_URL, api_key=_api_key(), timeout=120)
    return _thread_local.client


def parse_label(content: str | None) -> str | None:
    """Strict-JSON first, regex fallback. Returns 'positive'|'neutral'|'negative'|None."""
    if not content:
        return None
    content = content.strip()
    label = None
    try:
        label = json.loads(content).get("sentiment")
    except (json.JSONDecodeError, AttributeError):
        label = None
    if label not in ("positive", "neutral", "negative"):
        m = re.search(r"\b(positive|neutral|negative)\b", content)
        label = m.group(1) if m else None
    return label


def parse_emotion(content: str | None) -> str | None:
    """Strict-JSON first, regex fallback. Returns one of EMOTIONS or None.

    Anything else (e.g. the model inventing "excitement") is unparseable.
    """
    if not content:
        return None
    content = content.strip()
    emo = None
    try:
        emo = json.loads(content).get("emotion")
    except (json.JSONDecodeError, AttributeError):
        emo = None
    if emo not in EMOTIONS:
        m = re.search(r"\b(" + "|".join(EMOTIONS) + r")\b", content)
        emo = m.group(1) if m else None
    return emo


def call_llm_ext(title: str, text: str) -> tuple[str | None, str | None]:
    """One completion with retries on transient errors.

    Returns (sentiment, emotion); each part may be None if unparseable.
    Retries only while BOTH are unparseable (deterministic model output won't
    change on retry — an invalid emotion is counted, not fought).
    """
    prompt = build_prompt(title, text)
    last_err = None
    for attempt in range(RETRIES):
        try:
            resp = _client().chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=MAX_TOKENS,
            )
            msg = resp.choices[0].message
            content = getattr(msg, "content", None)
            label = parse_label(content)
            emotion = parse_emotion(content)
            if label is None or emotion is None:
                reasoning = getattr(msg, "reasoning", None)
                if label is None:
                    label = parse_label(reasoning)
                if emotion is None:
                    emotion = parse_emotion(reasoning)
            if label is not None or emotion is not None:
                return label, emotion
            last_err = f"unparseable response: {str(content)[:120]!r}"
        except Exception as e:  # transient network/server errors
            last_err = str(e)
            time.sleep(2 * (attempt + 1))
    print(f"  [warn] row failed after {RETRIES} tries: {last_err}")
    return None, None


def call_llm(title: str, text: str) -> str | None:
    """Sentiment-only wrapper (kept for step2/legacy callers)."""
    label, _ = call_llm_ext(title, text)
    return label


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_reviews():
    """Stream .jsonl.gz; yield dicts with at least one of title/text."""
    with gzip.open(DATA_FILE, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            title = (r.get("title") or "").strip()
            text = (r.get("text") or "").strip()
            if not title and not text:
                continue
            yield {
                "asin": r.get("asin", ""),
                "rating": r.get("rating"),
                "title": title,
                "text": text,
            }


def ground_truth(rating, binary: bool = False) -> str | None:
    """Rating -> ground-truth class.

    Step-6 spec (default): rating 4-5 -> POSITIVE, 3 -> NEUTRAL, 1-2 -> NEGATIVE.
    Steps 2-5 used a binary mapping (>=4 POSITIVE else NEGATIVE); pass
    binary=True for that (kept for step2_pipeline.py so it stays frozen).
    """
    if rating is None:
        return None
    if binary:
        return "positive" if rating >= 4 else "negative"
    if rating >= 4:
        return "positive"
    if rating == 3:
        return "neutral"
    return "negative"


def classify_row(rec):
    """Cached LLM classification for one review record (sentiment only).

    Cache values may be plain strings (legacy keys) or dicts (pv2 keys,
    {"s": sentiment, "e": emotion}); both are unwrapped here.
    """
    key = _cache_key(rec["title"], rec["text"])
    hit = cache_hit(key)
    if hit is not None:
        label = hit.get("s") if isinstance(hit, dict) else hit
        return {**rec, "prediction": label, "from_cache": True}

    label, _ = call_llm_ext(rec["title"], rec["text"])
    if label is not None:
        cache_store(key, {"s": label, "e": None})
        save_cache()
    return {**rec, "prediction": label, "from_cache": False}


def classify_row_ext(rec):
    """Cached classification returning BOTH sentiment and emotion (Step 5)."""
    key = _cache_key(rec["title"], rec["text"])
    hit = cache_hit(key)
    if hit is not None and isinstance(hit, dict):
        return {**rec, "prediction": hit.get("s"), "emotion": hit.get("e"),
                "from_cache": True}

    label, emotion = call_llm_ext(rec["title"], rec["text"])
    if label is not None or emotion is not None:
        cache_store(key, {"s": label, "e": emotion})
        save_cache()
    return {**rec, "prediction": label, "emotion": emotion, "from_cache": False}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=500,
                    help="random sample size (default 500)")
    ap.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    ap.add_argument("--workers", type=int, default=10,
                    help="parallel worker threads (default 10)")
    ap.add_argument("--no-dedupe", action="store_true",
                    help="sample raw rows instead of deduping identical title+text")
    ap.add_argument("--all", action="store_true",
                    help="classify the entire deduplicated file (ignores --rows)")
    args = ap.parse_args()

    load_cache()
    print(f"Endpoint: {BASE_URL} | model: {MODEL} | cache: {len(_cache)} entries")

    # 1) Load all reviews (3-star included), optionally dedupe identical title+text
    reviews = list(load_reviews())
    print(f"Reviews with title/text: {len(reviews)}")

    if args.no_dedupe:
        pool = reviews
        print("Dedupe disabled -- sampling from all raw rows")
    else:
        seen, unique = set(), []
        for r in reviews:
            k = (r["title"], r["text"])
            if k not in seen:
                seen.add(k)
                unique.append(r)
        pool = unique
        print(f"After dedupe of identical title+text: {len(pool)}")

    # 2) Sample or take all
    if args.all:
        sample = pool
    else:
        rng = random.Random(args.seed)
        n = min(args.rows, len(pool))
        sample = rng.sample(pool, n)
    print(f"Classifying {len(sample)} reviews with {args.workers} workers...")

    # 3) Parallel LLM classification, preserving input order
    start = time.time()
    results = [None] * len(sample)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(classify_row, rec): i for i, rec in enumerate(sample)}
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            if done % 50 == 0 or done == len(sample):
                elapsed = time.time() - start
                rate = done / elapsed if elapsed else 0
                print(f"  {done}/{len(sample)} ({rate:.1f} reviews/s)")
    save_cache(force=True)

    # 4) Evaluation + CSV
    n_correct = n_total = 0
    n_positive = n_negative = 0
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["asin", "rating", "title", "text", "ground_truth", "prediction", "correct"])
        for r in results:
            gt = ground_truth(r["rating"])
            pred = r["prediction"]
            correct = (gt == pred)
            writer.writerow([r["asin"], r["rating"], r["title"], r["text"], gt, pred, correct])
            if pred is not None:
                if pred == "positive":
                    n_positive += 1
                else:
                    n_negative += 1
            n_total += 1
            if correct:
                n_correct += 1

    print(f"\nWrote {CSV_FILE}")
    print(f"Total classified: {n_total} | positive: {n_positive} | negative: {n_negative} | "
          f"unparseable: {n_total - n_positive - n_negative}")
    print(f"Accuracy vs. rating ground truth: {n_correct}/{n_total} = "
          f"{n_correct / n_total:.4f}" if n_total else "n/a")
    print(f"Elapsed: {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
