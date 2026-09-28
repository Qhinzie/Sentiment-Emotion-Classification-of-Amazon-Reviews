# KNOWN ISSUES — Sentiment & Emotion Classification of Amazon Reviews

Every issue below was actually encountered during this project (Steps 1–8). Each entry follows the
same structure, and every claim is backed by a file, output, or verification result in this repo.
"Detected" marks how the issue surfaced: **verification scripts** (the Playwright suite
`src/verify_dashboard.py`, the startup checks in `src/dashboard.py`, or `src/check_readme_numbers.py`)
vs. **manual** (observed while working, later locked in by a script where possible).

Quick status map: 17 fixed / worked around · 2 open limitations (both data findings, not bugs).

## Summary

| # | Issue | Category | Status |
|---|-------|----------|--------|
| 1 | Dataset URL truncated (404) | agent-process | fixed |
| 2 | Engine script deleted; stale mock-model cache | agent-process | fixed |
| 3 | Cache key ignored prompt version | code | fixed |
| 4 | 3-star reviews marginalised in binary design | design decision | resolved in Step 6 |
| 5 | Imbalanced sample inflated accuracy | evaluation | fixed |
| 6 | Gradio loads Google Fonts (offline break) | UI | fixed |
| 7 | Tiny nonzero bars / label placement | charts | fixed |
| 8 | `###` section headings rendered literally | UI | fixed |
| 9 | String `"False"` is truthy → by-star chart 100% everywhere | charts | fixed |
| 10 | Rating-vs-text ground-truth mismatch (3★) | data | open limitation |
| 11 | NRC no-match / tie explosion | data | worked around |
| 12 | Macro-F1 used harmonic mean | code | fixed |
| 13 | verify_dashboard first run: 17 failures (harness bugs) | agent-process | fixed |
| 14 | Gradio combobox ignores synthetic clicks | agent-process | worked around |
| 15 | Duplicate (cached-frame) screenshots | agent-process | fixed |
| 16 | Port 7860 occupied by unrelated app | environment | worked around |
| 17 | Hardcoded API key default | code / security | fixed |
| 18 | Auto-generated disagreement reasons dishonest | code | worked around |
| 19 | `int("5.0")` crash on CSV ratings | code | fixed |
| 20 | EmoLex URL case-sensitivity | setup | fixed |

---

### 1. Dataset URL truncated in the assignment message (404)
- Step: Setup
- Category: agent-process
- Symptom: the data link pasted into the assignment 404'd; the download could not start.
- Cause: the URL was truncated in the chat message; the real files live under
  `mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/`.
- Fix / workaround: reconstructed the full path and verified it by downloading the file; the
  working URL is frozen in `setup.sh`.
- Status: fixed
- Evidence: `setup.sh:9` (`DATA_URL=...`); README "Q4" item 10
- Detected: manual

### 2. Engine script deleted; only .venv + a stale cache remained
- Step: Step 1
- Category: agent-process
- Symptom: `sentiment_classifier.py` was gone from disk; the environment held only the `.venv` and a
  JSON cache whose keys/values no longer matched any live model.
- Cause: the script was deleted outside the session; the cache had been written without prompt
  versioning and partly by a throwaway "mock-model" server.
- Fix / workaround: recreated the engine from the spec (strict-JSON + regex parsing, retries,
  disk cache).
- Status: fixed (correction: the surviving cache measures **120** `mock-model` entries + 1,585
  legacy plain-string entries + 100 pv2 + 150 pv3 = 1,955 total — the "1,205" figure from the
  session log does not match the file; 120 is what is actually stored).
- Evidence: cache census of `.llm_classifier_cache.json` (2026-09-28): prefixes `mock-model` ×120,
  `DeepSeek-V4-Flash-0731` (legacy plain strings) ×1,585, `pv2` ×100, `pv3` ×150; recreated engine
  at `src/sentiment_classifier.py`
- Detected: manual

### 3. Changed prompt reused old cached answers
- Step: Steps 5–6
- Category: code
- Symptom: after editing the prompt, cached labels from the previous prompt would be served.
- Cause: the cache key was `model|sha256(title⏎text)` — no prompt identity.
- Fix / workaround: cache key is now `PROMPT_VERSION|model|sha256(title⏎text)` (`"pv3"` current);
  Step 5 bumped to pv2 and Step 6 to pv3, so the 150 three-class calls were all live (0 stale hits).
- Status: fixed
- Evidence: `src/sentiment_classifier.py:80,119`; `outputs/step6_metrics.json`
  `run_details.prompt_version = "pv3"`; cache census above
- Detected: manual (designed into Step 5 before its run)

### 4. 3-star reviews marginalised by the binary design
- Step: Steps 1–2 and Step 6
- Category: design decision
- Symptom: the middle of the rating scale had no class of its own under "rating ≥ 4 → POSITIVE,
  else NEGATIVE".
- Cause: binary ground truth lumped 3★ into NEGATIVE; the early 500-row run additionally **excluded
  3★ entirely**; Step 2's final seed-42 sample is representative (2 of 100 rows are 3★, matching
  the ~2.3% pool share) but invisible as a class. Step 6 reintroduced 3★ as NEUTRAL with a balanced
  50/class draw.
- Fix / workaround: three-class ground truth (4–5 → POS / 3 → NEU / 1–2 → NEG) with balanced
  sampling in Step 6; the binary runs are preserved and labelled as such.
- Status: resolved by design in Step 6 (the underlying rating-vs-text mismatch is #10)
- Evidence: `data/attic/predictions.csv` star census (500 rows: 5★ 423, 4★ 21, 1★ 51, 2★ 5 — zero
  3★); `outputs/step2_predictions.csv` (3★ ×2); `outputs/step2_metrics.json`
  `accuracy_by_rating["3"] = {n: 2, correct: 2}`; `outputs/step6_metrics.json`
  `run_details.class_pool_sizes` (3★ pool 3,083 of 133,009)
- Detected: manual; the 3★-inclusion detail was re-verified against the files when writing this
  report (the "excluded in Step 2" framing in the original notes was corrected — Step 2's final
  sample does contain 3★)

### 5. Accuracy looked inflated vs. the always-positive baseline
- Step: Step 2
- Category: evaluation
- Symptom: 98.0% accuracy reads as near-perfect, but predicting "positive" for every review already
  scores 88.0% on the imbalanced seed-42 sample.
- Cause: the sample mirrors the population, which is ~87% 5-star.
- Fix / workaround: every evaluation reports the majority baseline and balanced statistics
  alongside accuracy (Step 2: baseline 0.88, balanced accuracy 0.9886, macro-F1 0.9558). Step 6
  replaced the imbalanced design with a balanced 50/class sample and a 33.3% majority baseline.
- Status: fixed
- Evidence: `outputs/step2_metrics.json` (`class_balance` 0.88/0.12, `baseline_always_predict_positive`
  0.88, `accuracy` 0.98); `outputs/step6_metrics.json` (`baseline_always_predict_majority`
  0.3333…, `accuracy` 0.72)
- Detected: manual

### 6. Gradio loads Google Fonts by default (offline break)
- Step: Step 3
- Category: UI
- Symptom: the dashboard reaches out to fonts.googleapis.com; offline or firewalled use renders
  with fallback fonts / delay.
- Cause: Gradio's default theme imports web fonts.
- Fix / workaround: explicit system font stack (`ui-sans-serif`, `system-ui`, `-apple-system`, …);
  the app is fully offline.
- Status: fixed
- Evidence: `src/dashboard.py:1050` (`SYSTEM_FONT = (...)`); README "Dashboard" section
  ("offline")
- Detected: manual

### 7. Tiny nonzero bars could render at zero width
- Step: Step 7
- Category: charts
- Symptom: values like 1 of 150 would be invisible (sub-pixel width), and labels inside short bars
  collided or shrank the bar.
- Cause: purely width-proportional layout with in-bar labels.
- Fix / workaround: nonzero fills get `min-width:3px`; labels live in a fixed side column outside
  the bar; zero values render as explicit "0" with genuinely 0 width; all enforced by Playwright
  `getBoundingClientRect` measurements.
- Status: fixed and regression-checked
- Evidence: `src/dashboard.py:1200` (`.hb-fill{…min-width:3px}`), `:1211` (`.st-seg{…
  min-width:3px}`), `:1212` (`.st-seg.st-zero{min-width:0;width:0}`); `src/verify_dashboard.py`
  min-size checks; final run: "63/63 checks passed" (exit 0)
- Detected: manual (design rule) → enforced by the verification script

### 8. `###` section headings rendered as literal text
- Step: Steps 3–8 (present since Step 3, found in Step 8)
- Category: UI
- Symptom: every section heading after a kicker `<div>` rendered as literal `### How well did it
  do?` text instead of a heading; only one heading in the app was a real `<h3>`.
- Cause: `Gradio 6.28` treats a markdown block starting with raw HTML as an HTML block — the
  markdown inside is never parsed.
- Fix / workaround: blank line between the `</div>` and the `###` (as `\n\n` inside the string
  literal or a line break between adjacent literals) at all 11 section sites; verified the DOM now
  contains `<h3>`; section screenshots re-captured against `<h3>` anchors.
- Status: fixed
- Evidence: `src/dashboard.py` (9 × `</div>\n\n` + 2 × inline `\n\n` before `###`); the six
  re-captured section screenshots in `docs/screenshots/`; README "Q4" item 3
- Detected: manual (DOM probe) — the verification script did **not** catch it because its checks
  assert values, not heading tags (gap noted in `src/capture_screenshots.py`)

### 9. CSV `"False"` is truthy → by-star chart showed 100% everywhere
- Step: Step 7
- Category: charts
- Symptom: every star rating displayed 100% accuracy; the generated takeaway claimed "1★ reviews
  are the hardest" — wrong (the weakest class is 3★ at 22%).
- Cause: the predictions CSV stores `correct` as the string `"True"`/`"False"`, and `if r["correct"]`
  is always truthy.
- Fix / workaround: compute correctness as `ground_truth == prediction` (same rule as
  `compute_stats`); the verifier now re-derives by-star accuracy from the CSV the same way.
- Status: fixed
- Evidence: README "Q4" item 2; `src/verify_dashboard.py` by-star check (63/63); real by-star
  numbers in `outputs/step6_metrics.json` `accuracy_by_rating` (3★ = 0.22)
- Detected: manual (pre-verifier smoke check found the wrong takeaway) → locked in by the script

### 10. Ratings and text disagree for 3★ reviews (NEUTRAL is a words-vs-stars problem)
- Step: Step 6
- Category: data
- Symptom: a third of the balanced sample is 3★, but the model labels 60% of those NEGATIVE
  (30/50) and 18% POSITIVE (9/50); NEUTRAL recall is 22%.
- Cause: Gift-Card 3★ texts rarely read neutral ("The gift card itself was fine, no issues" on a
  3★ is complaint-dominant); the model reads the words, the ground truth reads the stars.
- Fix / workaround: none — a finding, not a bug. Documented with curated why-notes per error; the
  balanced sample makes the mismatch visible instead of hidden.
- Status: open limitation
- Evidence: `outputs/step6_metrics.json` `true_neutral_destination` (negative 30 / neutral 11 /
  positive 9), `per_class.neutral.recall = 0.22`; `outputs/step6_errors.csv` (42 rows with
  pattern + why columns); README Q3 / Q4 item 9
- Detected: manual (error-group autopsy); numbers locked by the verifier

### 11. NRC EmoLex: no-match and tie explosion
- Step: Steps 5–6
- Category: data
- Symptom: Step 6: 20/150 texts have no lexicon match ("none") and 78/150 tie between emotions;
  LLM↔NRC agreement is 36.7% (42.3% excluding "none", 6% strict).
- Cause: EmoLex v0.92 covers only 4,454 of 14,154 words for the 8 emotions; common review words
  (`great`, `nice`, `easy`, `works`, `fast`) are absent; most words light up 2–5 emotion columns;
  the lexicon is context-blind (negation, sarcasm).
- Fix / workaround: tie rule (agreement when the LLM emotion is in the tied set), "none" reported
  separately, coverage quantified and disclosed; negation detection remains a known weakness (the
  heuristic even flags "no issues" as negation).
- Status: worked around (open limitation)
- Evidence: `outputs/step5_metrics.json` `agreement` (0.52 / 0.7123 excl-none / 0.11 strict,
  n_excluding_none 73); `outputs/step6_metrics.json` `agreement` (0.3667 / 0.4231 / 0.06, 130),
  `nrc = {n_none: 20, n_tie: 78}`, `disagreements.by_reason` = {no_matched_words 20, tie 32,
  negation 26, context_mismatch 16, terse 1}; `outputs/step7_dataset_stats.json`
  (`lexicon_words_total` 14,154 / `lexicon_words_with_emotion` 4,454); README Q4 item 5
- Detected: manual (coverage audit); counts locked by the verifier

### 12. Macro-F1 computed as a harmonic mean
- Step: Step 2
- Category: code
- Symptom: reported macro-F1 (0.9547) did not equal the arithmetic mean of the per-class F1s.
- Cause: the metrics code blended per-class F1 harmonically instead of averaging.
- Fix / workaround: arithmetic mean per the assignment spec; value re-checked against
  `outputs/step2_metrics.json` (0.9558 = (0.98851 + 0.92308) / 2).
- Status: fixed
- Evidence: README "Q4" item 6; `outputs/step2_metrics.json` `macro_f1` 0.9557913351016799 vs
  `per_class` F1s
- Detected: manual (number audit)

### 13. verify_dashboard.py first run: 17 failures — all harness bugs, not app bugs
- Step: Step 7
- Category: agent-process
- Symptom: 17 failing checks while every rendered value was correct.
- Cause: float tolerance 1e-9 vs 6-decimal `data-value` strings; wrong JSON key case ("accuracy"
  vs "Accuracy"); a dashed border on zero segments made them 1px wide (defeating `width:0`);
  an over-strict label-overlap condition.
- Fix / workaround: 1e-4 tolerance, key-case alignment, border removed from `.st-zero`, simpler
  box-intersection test.
- Status: fixed (63/63)
- Evidence: README "Q4" item 11 (harness issues); final run: "63/63 checks passed", exit 0
- Detected: by the verification script itself (flagged → diagnosed as probe bugs)

### 14. Gradio combobox ignores synthetic clicks; on_select crashes via raw API
- Step: Steps 4–6
- Category: agent-process (test harness)
- Symptom: dropdown filters appeared unclickable from test code, and the row-click detail feature
  looked broken; invoking the row `select` handler with a synthetic event crashed
  (`'NoneType' object is not subscriptable`).
- Cause: Gradio 6 svelte components ignore `el.click()` (real pointer events required); the raw
  handler needs a full `SelectData` payload (real UI clicks carry one); clicks at off-viewport
  coordinates are silently dropped.
- Fix / workaround: drive dropdowns with CDP `Input.dispatchMouseEvent` after
  `scrollIntoView({block:'center'})`; always test row interactions through the real UI.
- Status: worked around
- Evidence: README "Q4" item 11; browser gauntlet results in the Step-4/6 reports
- Detected: manual (plus a genuine crash from the harness attempt)

### 15. Duplicate screenshots from cached frames
- Step: Steps 5–7
- Category: agent-process (test harness)
- Symptom: consecutive captures were byte-identical (same MD5) even though the UI had changed.
- Cause: the browser-screenshot harness served a cached identical frame.
- Fix / workaround: superseded by Playwright element-clip screenshots
  (`src/verify_dashboard.py`, `src/capture_screenshots.py`).
- Status: fixed
- Evidence: README "Q4" item 12; the distinct screenshot set in `docs/screenshots/`
  (`step7_desktop.png` vs `step7_narrow.png` differ; six section shots from Playwright)
- Detected: manual (screenshot diffing)

### 16. Port 7860 occupied by an unrelated app
- Step: Steps 3–8
- Category: environment
- Symptom: the dashboard could not bind to the default Gradio port.
- Cause: another app already listens on 7860 on this machine.
- Fix / workaround: `find_free_port(7860)` fallback; the app prints the real port
  (typically 127.0.0.1:7861).
- Status: worked around
- Evidence: `src/dashboard.py:1400` (`def find_free_port(start=7860, …)`); README "Dashboard"
  run note
- Detected: manual

### 17. Hardcoded API key default in the engine
- Step: Steps 1–7 → hardened in Step 8
- Category: code / security
- Symptom: `sentiment_classifier.py` fell back to a course-provided key when `LLM_API_KEY` was
  unset — a credential baked into source.
- Cause: convenience default from early development.
- Fix / workaround: the key is read ONLY from `LLM_API_KEY` (no default anywhere); `_api_key()`
  raises `RuntimeError` when unset; `.env.example` documents the variable; Grep audit + gitleaks
  clean; the key value was also scrubbed from the helper skill and memory notes.
- Status: fixed
- Evidence: `src/sentiment_classifier.py:59` (`_api_key()`), `:165`
  (`OpenAI(..., api_key=_api_key())`); README "Q4" item 1; gitleaks v8.30.1 run on commit
  `4d33029`: "no leaks found"
- Detected: manual (Step-8 secrets audit)

### 18. Auto-generated disagreement reasons were dishonest
- Step: Step 5 (revisited in Step 6)
- Category: code
- Symptom: the automated reason rules labelled 8 rows "negation" where only 3 were genuine
  negations and reported 0 sarcasm anywhere; some labels misled the analysis.
- Cause: simple keyword heuristics over-trigger (e.g. the word "no").
- Fix / workaround: hand-verified curated labels for Step 5; Step 6 keeps auto-grouped buckets but
  discloses them as uncurated in the README, with per-error why-notes for the largest group.
- Status: worked around
- Evidence: README "Q4" item 8 and Q3 note (the negation over-trigger on "no issues");
  `outputs/step6_errors.csv` (`pattern` + `why` columns)
- Detected: manual

### 19. `int("5.0")` crash on CSV ratings
- Step: Steps 4+
- Category: code
- Symptom: code that trusted `int(rating)` crashed, since CSV ratings are float strings.
- Cause: pandas writes ratings like `5.0`.
- Fix / workaround: parse as `int(float(rating))` everywhere ratings are read.
- Status: fixed
- Evidence: README "Q4" item 4
- Detected: manual (dashboard crash)

### 20. EmoLex download URL is case-sensitive
- Step: Step 5 (Setup)
- Category: setup
- Symptom: the NRC lexicon zip 404'd with a lowercase path.
- Cause: the official path is `WebDocs/Lexicons/` (capital L).
- Fix / workaround: exact URL frozen in `setup.sh` with an explanatory comment; download verified
  (4,454 emotion-bearing words loaded).
- Status: fixed
- Evidence: `setup.sh:12-13`; `outputs/step7_dataset_stats.json` lexicon counts; README "Q4"
  item 10
- Detected: manual
