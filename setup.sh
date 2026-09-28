#!/usr/bin/env bash
# Download the data + lexicon used by this project into ./data/ (gitignored).
# Run from the repo root:  ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p data

DATA_URL="https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Gift_Cards.jsonl.gz"
DATA_FILE="data/Gift_Cards.jsonl.gz"

LEXLINK_PAGE="http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm"
LEX_URL="https://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Lexicon.zip"
LEX_DIR="data/nrc_lexicon"
LEX_TXT="$LEX_DIR/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

if [ -f "$DATA_FILE" ]; then
  echo "Data present: $DATA_FILE"
else
  echo "Downloading Gift Cards reviews ($DATA_URL) ..."
  curl -fL --retry 3 -o "$DATA_FILE" "$DATA_URL"
fi

if [ -f "$LEX_TXT" ]; then
  echo "Lexicon present: $LEX_TXT"
else
  echo "Downloading NRC Emotion Lexicon v0.92 ($LEX_URL) ..."
  echo "  (announced at $LEXLINK_PAGE; note the capital L in 'Lexicons')"
  mkdir -p "$LEX_DIR"
  curl -fL --retry 3 -o "$LEX_DIR/NRC-Emotion-Lexicon.zip" "$LEX_URL"
  (cd "$LEX_DIR" && unzip -q -o NRC-Emotion-Lexicon.zip)
  rm -f "$LEX_DIR/NRC-Emotion-Lexicon.zip"
fi

echo "Done. Install Python deps: python -m venv .venv && .venv/bin/pip install -r requirements.txt"
echo "Then: export LLM_API_KEY=... (see .env.example) and run the pipelines from src/."
