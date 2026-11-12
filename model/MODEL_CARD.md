# Model card — `cu.traineddata`

_Fill in the bracketed fields at release time ([how](../docs/manual.md#step-8-fill-in-the-model-card));
the structure is fixed so downstream users know what to expect._

## Overview

- **Task:** OCR of printed **Church Slavonic** liturgical texts (Cyrillic with
  titla, superscript letters, accents and breathing marks, letter-numerals).
- **Engine:** Tesseract LSTM (v5.3.4). Use with `-l cu`, `--psm 13` for
  single-line crops or `--psm 4` for a single column of a page.
- **Download:** `cu.traineddata` from the repo's [Releases](../../releases) page (asset name is constant; the tag carries the version).
- **Version / date:** v1.0 — 2026-11-12
- **Training regime:** fine-tuned from the model produced with synthetic data, `MAX_ITERATIONS=8000`.

## Training data

- **Source text:** [`slavonic/cu-md-sandbox`](https://github.com/slavonic/cu-md-sandbox)
  at commit `e2033fe`. Cleaned to a Church-Slavonic character set (letters, combining
  marks, print punctuation, brackets, liturgical symbols, Arabic digits); markup,
  links, URLs, and editorial apparatus removed. See [data generation](../docs/data-generation.md).
- **Rendering:** ~40000 distinct lines rendered in 5 typefaces (Ponomar,
  Triodion, Pochaevsk, Acathist, Monomakh) with HarfBuzz shaping, light
  degradation (skew/blur/noise), and hyphenated line-ends injected at rate 0.20.
- **Real adaptation:** 2465 hand-corrected lines from [the books listed](../data/real-lines/README.md), held-out
  eval set of 257 lines.
- **Character set:** 150 classes. Verify with
  `combine_tessdata -u cu.traineddata /tmp/x. && head -1 /tmp/x.lstm-unicharset`.

## Metrics

Measured on the held-out **real** eval set (not the synthetic split, which is
much easier and not comparable across data changes — see
[troubleshooting](../docs/troubleshooting.md#training-error-and-real-error-dont-match)).

| metric | value |
|--------|-------|
| CER (full) | 5.75% |
| CER (marks stripped) | 3.70% |
| WER (full) | 31.46% |

Ground-truth convention: a dash between syllables or at a line break, whether
line-end hyphenation or a syllable divider in chant lyrics, is transcribed as a
single token, `_`. The model does not distinguish the two; position does
(line-final = hyphenation, mid-line = syllable boundary).

## Intended use & limits

- **Intended:** modern printed Church Slavonic editions in a typeface close to
  the training faces.
- **Out of scope / weaker:** manuscripts and handwriting; typefaces unlike the
  five above; heavy red-rubric pages after naive binarization; very low-quality
  or skewed scans (preprocess first — Sauvola binarization + deskew).
- **Known confusions:** к/в, и/н under low resolution; diacritical marks are often doubled.

## Reproduction

Follow the [manual](../docs/manual.md): `make setup`, `make dataset`,
`make train-seeded`, then collect real lines, fine-tune and `make eval`.
