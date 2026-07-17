# Model card — `cu.traineddata`

_Fill in the bracketed fields at release time; the structure is fixed so
downstream users know what to expect._

## Overview

- **Task:** OCR of printed **Church Slavonic** liturgical texts (Cyrillic with
  titla, superscript letters, accents and breathing marks, letter-numerals).
- **Engine:** Tesseract LSTM (v[5.x]). Use with `-l cu`, `--psm 13` for
  single-line crops or `--psm 4` for a single column of a page.
- **Download:** `cu.traineddata` from the repo's [Releases](../../releases) page (asset name is constant; the tag carries the version).
- **Version / date:** [vX.Y — YYYY-MM-DD]
- **Training regime:** [from scratch | fine-tuned from <base>], `MAX_ITERATIONS=[…]`.

## Training data

- **Source text:** [`slavonic/cu-md-sandbox`](https://github.com/slavonic/cu-md-sandbox)
  at commit [`…`]. Cleaned to a Church-Slavonic character set (letters, combining
  marks, print punctuation, brackets, liturgical symbols, Arabic digits); markup,
  links, URLs, and editorial apparatus removed. See `docs/data-generation.md`.
- **Rendering:** ~[N] distinct lines rendered in [5] typefaces (Ponomar,
  Triodion, Pochaevsk, Acathist, Monomakh) with HarfBuzz shaping, light
  degradation (skew/blur/noise), and hyphenated line-ends injected at rate [0.20].
- **Real adaptation:** [N] hand-corrected lines from [which books], held-out
  eval set of [N] lines.
- **Character set:** [N] classes. Verify with
  `combine_tessdata -u cu.traineddata /tmp/x. && head -1 /tmp/x.lstm-unicharset`.

## Metrics

Measured on the held-out **real** eval set (not the synthetic split, which is
much easier and not comparable across data changes — see `docs/troubleshooting.md`).

| metric | value |
|--------|-------|
| CER (full) | [X.X %] |
| CER (marks stripped) | [X.X %] |
| WER (full) | [X.X %] |

Ground-truth convention: line-break hyphenation is transcribed as `_`.

## Intended use & limits

- **Intended:** modern printed Church Slavonic editions in a typeface close to
  the training faces.
- **Out of scope / weaker:** manuscripts and handwriting; typefaces unlike the
  five above; heavy red-rubric pages after naive binarization; very low-quality
  or skewed scans (preprocess first — Sauvola binarization + deskew).
- **Known confusions:** [e.g. ъ/ѣ, ж/ѧ] under low resolution; the `-`/`_`
  distinction (compound hyphen vs line-break) is positional and imperfect if the
  book prints them identically.

## Reproduction

`make setup && make dataset && make train && make eval` (see `docs/pipeline.md`).
