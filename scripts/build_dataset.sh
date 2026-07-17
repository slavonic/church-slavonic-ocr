#!/usr/bin/env bash
# Regenerate the synthetic training set from the corpus + fonts.
# This is the single source of truth for how the ground truth was produced.
set -euo pipefail
cd "$(dirname "$0")/.."

FONTS=(
  fonts/Ponomar/fonts/ttf/Ponomar-Regular.ttf
  fonts/Triodion/fonts/ttf/Triodion-Regular.ttf
  fonts/Pochaevsk/fonts/ttf/Pochaevsk-Regular.ttf
  fonts/Acathist/fonts/ttf/Acathist-Regular.ttf
  fonts/Monomakh/fonts/ttf/Monomakh-Regular.ttf
)

python3 scripts/cu_make_training_data.py \
  --corpus corpus \
  --out data/cu-ground-truth \
  --fonts "${FONTS[@]}" \
  --dedupe --degrade \
  --hyphenate 0.20 --hyphen-glyph '-' \
  --seed 1 \
  --limit "${LIMIT:-40000}"
