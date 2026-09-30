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
- Fix (b): `make clean-output` + remove the traineddata, then retrain. A changed
  allow-set (adding digits, `_`) **requires** this rebuild.

## Long runs of garbage: `ЩщОощеҹоҹҹ…` where short text should be

The recognizer is reading **noise**, not text — doubled superscripts and stacked
accents are the diacritic band of a neighbouring line plus scan speckle. This is
an **image-quality / segmentation** problem, not the model.

- Prove it: OCR one *clean* line crop with `--psm 13`. If it reads fine, the
  model is healthy and the page path is at fault.
- Fix: binarize before OCR (Sauvola handles uneven historical paper), deskew,
  despeckle, suppress show-through, and segment with `--psm 4`. Quick test:
  `convert page.png -colorspace Gray -lat 25x25+10% -despeckle /tmp/bw.png`.

## Eval doesn't move at all after fine-tuning, and errors look like the charset never changed (e.g. a fine-tune-only character never once appears in output)

Check whether the fine-tuned model was ever actually **packaged and deployed**
before assuming anything about the model itself. tesstrain's `training` target
only writes checkpoint files (`training/<model>/checkpoints/`); it does not
produce a usable `.traineddata`, and nothing in the plain fine-tune recipe
copies a result into `model/`. If eval reads `--tessdata-dir model`, it will
silently keep scoring whatever was there before — typically the pre-fine-tune
model from `make train-seeded` — with no error, and `--reocr` won't help
because the model *file* genuinely hasn't changed. Confirm with
`ls -la model/cu.traineddata training/cu/checkpoints/` — if the checkpoints
are newer than the deployed traineddata, that's it. Fix: package the
checkpoint and copy it over (see the end of the fine-tuning section in
`docs/training.md`) before re-evaluating.

## Ordinary text in a line reads well, then a number at the end explodes into garbage

Symptom: a line's main text decodes nearly perfectly, then a trailing number
— a page marker, verse number, or Cyrillic letter-numeral like `г҃` — turns
into unrelated symbol garbage, often after a noticeably wide gap in the
reference text (`Послѣ́дованїе ѡ҆ и҆сповѣ́данїи 73` → correct text, then `73`
becomes `:с:с:с::с:стз`).

The wide gap is the tell: that number is very likely a distinct typographic
zone — a running header or verse marker — that `extract_lines.py`'s
segmentation is incidentally sweeping into the main text-line crop. Numbering
apparatus is a small fraction of continuous liturgical prose, so it's
plausibly just undertrained relative to running text — structurally the same
shape of problem as the melisma-marker case above (a construct that's rare in
the training distribution, failing much harder than its rarity alone would
predict). Quantify it with `cu_eval.py`'s built-in `numerals` bucket
(`docs/evaluation.md`) before deciding it's worth chasing — check whether it
carries a disproportionate share of total error the same way melisma did.

If it does: check the crops in `report.html` first, to confirm these really
are separate zones rather than ordinary in-line numerals. If confirmed, either
crop tighter to exclude the marginal number, or make sure enough
number-containing real lines are in `data/real-lines/finetune/` for the model
to actually learn that register.

## A character introduced only in fine-tune data gets consistently misread as a similar-looking existing character

Symptom: a mark that appears **only** in your hand-corrected real lines (never
in the synthetic corpus) — e.g. the `~`/`‿` melisma divider — comes out as a
different, visually similar mark that *was* in the synthetic data (e.g. `_`,
the hyphenation token), almost every time it occurs, even though ordinary text
in the same fine-tune is reading fine.

Cause: `START_MODEL=cu` fine-tuning merges unicharsets
(`merge_unicharsets` in tesstrain — see `docs/training.md`). A character with
no synthetic examples enters the merged unicharset as a **brand-new output
class with random initial weights**, competing against a visually similar
class that carries the full weight of your original run (tens of thousands of
iterations). A short fine-tune is nowhere near enough for the new class to
catch up — the network defaults to its confident, well-trained neighbor.
Quantify how much of your total CER this accounts for with `cu_eval.py`'s
built-in split (`docs/evaluation.md`) before deciding it's worth fixing.

Fix: give the new class real exposure — more real-line examples containing
it in `data/real-lines/finetune/`, and/or more fine-tune iterations (a cold
class needs materially more than a warm one to converge). This is a data/
iteration problem, not a sign the model or the character choice is broken.

## Words come out flipped to full caps or alternating case, otherwise legible

Symptom: an isolated word decodes with mostly-correct letters but wrong case
throughout (`Сла́ва,` → `СѧЛА́ВАѧ`), rather than scattered single-letter errors.

This is structured, not random — check the actual crop in `report.html`
before assuming it's a model defect. Liturgical books often set incipits,
exclamations (`Сла́ва`, `Ны́нѣ`), or headings in versals/rubricated display
capitals, a typographic register your synthetic corpus likely never renders
(plain body-text weight only). If the crop confirms decorative/enlarged caps,
this is a domain gap like any other: either normalize how such words are
transcribed, or add examples of that register to training. If the crop is
ordinary lowercase print, it's a genuine — and separate — model confusion.

## `combine_tessdata -u … Error 1` when fine-tuning (`START_MODEL=cu`)

The line right before the `Error 1` names the command that failed — it's
tesstrain unpacking `START_MODEL` via `combine_tessdata -u
$(TESSDATA)/$(START_MODEL).traineddata …`. If `TESSDATA` wasn't passed on the
command line, it defaults to a `usr/share/tessdata/` folder next to
`DATA_DIR` — a path this repo never creates, so the file it's looking for
doesn't exist. Fix: pass `TESSDATA=$PWD/training` (or `$PWD/model`) explicitly
— see `docs/training.md`'s fine-tune section. This is specific to the plain
`START_MODEL=` fine-tune path; `train_seeded.py` never hits it, since it
extracts and continues from the seed model directly rather than going through
tesstrain's own `START_MODEL` lookup.

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

## Model garbles even a *real* line, but structure is preserved

If word count, comma and accent positions are right and some substrings are
correct, the network is decoding but there's a mismatch or a domain gap:

- **Test the decisive case:** OCR a clean line straight from `data/cu-ground-truth`.
  - Garbled too → the deployed traineddata doesn't match the trained network
    (unicharset/recoder mismatch, usually from retraining on a changed charset
    without `clean-output`). Rebuild clean.
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

## ъ/ѣ, ж/ѧ and similar minimal-pair confusions

The distinguishing stroke (yat's crossbar, the yus bowl) is exactly what a
low-quality scan blurs away. Higher DPI + better binarization recover much of it;
a real-line fine-tune in the target face teaches the rest in context.

## Shell / file manager chokes on the ground-truth directory

Hundreds of thousands of files overflow `*` globs and `ls`. Use
`scripts/review_samples.py` (lazy `os.scandir`, never lists everything) or
`find … -exec … {} +`, and slice a handful with `head` before opening.

## `cu_eval.py` shows no change after retraining

OCR is cached per line as `.hyp.txt`. The tool auto-detects a cache older than
the current model file and re-OCRs it, printing how many it refreshed — so
this should now self-correct. It can only compare timestamps when it can find
the model file, though: if you ran with a bare `--model` and no
`--tessdata-dir` (relying on `TESSDATA_PREFIX`), it can't locate the file and
warns instead of silently trusting the cache — pass `--reocr` explicitly in
that case, or just always pass `--tessdata-dir` so the check can run.
