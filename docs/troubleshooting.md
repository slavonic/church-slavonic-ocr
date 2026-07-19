# Troubleshooting

Symptoms we actually hit, what they mean, and the fix. The recurring lesson:
**change one thing at a time, and diagnose before rebuilding.**

## Synthetic CER and real CER are not comparable

The BCER printed during training is measured on a validation split of the *same*
data you trained on. Clean synthetic renders score very low (< 1 %); realistic,
degraded, hyphenated data scores higher — and a *higher* synthetic BCER on
harder, more scan-like data can mean a *better* model on real scans. Only compare
synthetic BCER across runs when the data difficulty is unchanged. The real verdict
is `cu_eval.py` on held-out real lines. Target there: ≤ 2 % CER.

## Foreign glyphs in the output (schwa, Latin, `©`)

The model can only emit classes in its unicharset. Foreign glyphs mean either
(a) the charset was contaminated by junk in the ground truth, or (b) a stale
Cyrillic-derived charset survived a supposedly from-scratch run.

- Confirm which: `head -1 training/cu/unicharset` (~200–300 = clean CU;
  thousands = stale) and grep it for Latin/schwa.
- Fix (a): clean the corpus (the generator now strips links/URLs/editorial
  markup and allow-set-filters the rest — `docs/data-generation.md`) and rebuild.
- Fix (b): `make reset-charset` (see "Resetting the charset" in
  `docs/training.md`), then retrain. A changed allow-set (adding digits, `_`)
  **requires** this rebuild.

## Long runs of garbage: `ЩщОощеҹоҹҹ…` where short text should be

The recognizer is reading **noise**, not text — doubled superscripts and stacked
accents are the diacritic band of a neighboring line plus scan speckle. This is
an **image-quality / segmentation** problem, not the model.

- Prove it: OCR one *clean* line crop with `--psm 13`. If it reads fine, the
  model is healthy and the page path is at fault.
- Fix: binarize before OCR (Sauvola handles uneven historical paper) and
  deskew, then segment with `--psm 4`. `extract_lines.py --deskew --binarize`
  does both directly (see `docs/evaluation.md`), applied before segmentation/OCR
  and before crops are cut, so the boxes and the saved lines see the same
  cleaned-up page. `--sauvola-window`/`--sauvola-k` tune the binarization if the
  defaults over- or under-ink a particular scan.

## Model garbles even a *real* line, but structure is preserved

If word count, comma and accent positions are right and some substrings are
correct, the network is decoding but there's a mismatch or a domain gap:

- **Test the decisive case:** OCR a clean line straight from `data/cu-ground-truth`.
  - Garbled too → the deployed traineddata doesn't match the trained network
    (unicharset/recoder mismatch, usually from retraining on a changed charset
    without `make reset-charset`). Rebuild clean.
  - Reads fine → the model is healthy; the real-scan failure is a **domain gap**
    (image appearance + typeface). Binarize the scans and fine-tune on real lines.
- Also rule out a stale install: `--tessdata-dir model` to force *this* model.

## `make training` looks stuck in a loop generating `.box` files

Not a loop — it's the one-time box→lstmf preprocessing, one subprocess per file,
over ~200k pairs. The filename index climbing proves progress
(`watch 'find data/cu-ground-truth -name "*.box" | wc -l'`). Run with
`-j$(nproc)` to parallelize; boxes are `.PRECIOUS` so it resumes. If the count
does *not* climb or the same file rebuilds every run, suspect filesystem clock
skew (network mounts) — keep the ground truth on a local disk.

## BCER pinned ~98–100% after tens of thousands of iterations (from scratch)

Signature: `BCER train` ≈ 97–100% and moving ~0.02%/100 iters, `BWER` ≈ 100%,
`mean rms` low (~6%) and slowly falling, `skip ratio` 0 — and decoding a
training image with the current checkpoint yields empty output or a short
near-constant stub of high-frequency glyphs (e.g. `п҆ъ`) regardless of the
input image. That is **CTC collapse**: the net emits blank at almost every
timestep (plus, at the few committed steps, its highest-prior classes) and is
stuck in that basin. It will not escape with more iterations; restarting
unchanged just rerolls the init dice.

Fix: seed the feature layers instead of random init — `make train-seeded`
(see `docs/training.md`). Do **not** use tesstrain's `START_MODEL=Cyrillic`
for this, which merges the foreign charset back in; the seeded runner
continues from Cyrillic's extracted `.lstm` with `--old_traineddata` so the
output layer is rebuilt against the clean CU unicharset, and its watchdog
aborts+retries automatically if a run collapses anyway.

## ъ/ѣ, ж/ѧ and similar minimal-pair confusions

The distinguishing stroke (yat's crossbar, the yus bowl) is exactly what a
low-quality scan blurs away. Higher DPI + better binarization recover much of it;
a real-line fine-tune in the target face teaches the rest in context.

## Shell / file manager chokes on the ground-truth directory

Hundreds of thousands of files overflow `*` globs and `ls`. Use
`scripts/review_samples.py` (lazy `os.scandir`, never lists everything) or
`find … -exec … {} +`, and slice a handful with `head` before opening.

## `cu_eval.py` shows no change after retraining

OCR is cached per line as `.hyp.txt`. Pass `--reocr` to re-run the new model;
otherwise you re-score the old outputs.
