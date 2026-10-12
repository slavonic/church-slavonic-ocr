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
checkpoint under its own name and evaluate that file with `--model <name>
--tessdata-dir training` (see "Package and evaluate" in `docs/training.md`)
before copying it into `model/`.

## Ordinary text in a line reads well, then a number at the end explodes into garbage

Symptom: a line's main text decodes nearly perfectly, then a trailing number
(a page marker, verse number, or Cyrillic letter-numeral like `г҃`) turns into
unrelated symbol garbage, often after a noticeably wide gap in the reference
text (`Послѣ́дованїе ѡ҆ и҆сповѣ́данїи 73` → correct text, then `73` becomes
`:с:с:с::с:стз`).

**Measure before chasing it.** A few striking lines in the worst-lines list
are not a pattern. `cu_eval.py`'s `numerals` bucket (`docs/evaluation.md`)
reports how many lines and characters actually contain Arabic digits and what
share of total error they carry. In this project's eval set that bucket was a
single line, 0.3% of reference characters and 1.5% of total error, so the
worst-list impression was misleading and there was nothing to fix. Cyrillic
letter-numerals are not counted by that bucket, and a "contains titlo" bucket
would not isolate them either, since the titlo also marks ordinary
abbreviations like `бг҃ъ`.

If the bucket does turn out to carry a disproportionate share, a wide gap
before the number suggests a distinct typographic zone (a running header or
verse marker) that `extract_lines.py`'s segmentation sweeps into the
main-text crop. Check the crops in `report.html` to confirm. Then either exclude
the marginal number by re-extracting with `--margin` (see
`docs/evaluation.md`), or add enough number-containing real
lines to `data/real-lines/finetune/` for the model to learn that register.

## A mark that appears only in the fine-tune data is misread as a similar-looking mark

Symptom: a mark that occurs only in your hand-corrected real lines (never in
the synthetic corpus) comes out as a different mark that *was* in the synthetic
data, almost every time, while ordinary text in the same lines reads fine. The
case we hit: a syllable divider labeled `~` in real chant lyrics, read back as
`_` (the hyphenation token) or as stray dashes and spaces.

First decide which of two causes you have; they need opposite fixes.

**A. Two labels for one image.** If the printed marks are visually the same
(a syllable dash and a line-end hyphen are both just a short dash), the model
is being trained on contradictory pairs: the same pixels labeled `_` in the
synthetic data and `~` in the real data. Nothing visual separates them, so
more data and more iterations cannot fix it. Fix the labels: use **one token**
for marks that look alike. This project uses `_` for both (see
`data/README.md`); position already tells you which is which (line-final =
hyphenation, mid-line = syllable boundary), so nothing is lost.

**B. A genuinely distinct mark that is just rare.** `merge_unicharsets` adds a
character that exists only in the fine-tune data as a new output class with
random initial weights, competing against classes with tens of thousands of
iterations behind them. If the mark really does look different, the fix is
exposure: more real lines containing it in `data/real-lines/finetune/`, and
more fine-tune iterations (a cold class needs materially more than a warm one).

Test for A before spending effort on B: open a few misread crops in
`report.html` and ask whether you could tell the mark from its confusion
partner by eye. Quantify the damage first with `cu_eval.py`'s per-construct
split (`docs/evaluation.md`).

Even after the labels agree, lines with mid-word dividers can stay weak: the
synthetic corpus never contains a divider inside a word, so letters next to it
are unfamiliar. If that persists, render syllabified corpus text with a drawn
dash as the divider.

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

## `Code range changed from N to M!` / `Must supply the old traineddata for code conversion!`

`N` and `M` are the number of output classes. The message appears in two very
different places.

*At the start of a fine-tune,* right after a line naming
`…/checkpoints/<name>_checkpoint`: `lstmtraining` first tries to resume from a
checkpoint already sitting in the output directory, rejects it because the
class count changed, and falls back to `--continue_from`. By itself this is
harmless, but it tells you the `MODEL_NAME` was reused and an old run's
checkpoint is still there. Use a fresh name per run (`docs/training.md`, rule 1).

*At `lstmtraining --stop_training`, ending in `Failed to read continue from`:*
fatal. The checkpoint and the proto `--traineddata` you passed describe
networks with different class counts, so the unicharset you supplied is not the
one the checkpoint was trained against. The usual cause is a checkpoint that is
**older than the proto**: the proto was rebuilt after the ground truth changed
(a relabel, a dropped character), and the training run that should have
produced a matching checkpoint never ran or never finished. Confirm with
`ls -l --time-style=full-iso` on both files, and check whether anything in
`checkpoints/` is newer than the proto.

Do not silence it with `--old_traineddata`. That makes the command succeed by
converting the *old* checkpoint's weights to the new charset, so you deploy the
previous model under a fresh timestamp and the next eval looks like the change
did nothing. Fix it by getting a new run to finish, under a new `MODEL_NAME`.

## Fine-tune log stops after `Continuing from …`; `make` reports no error

Symptom: `finetune.log` ends right after the network summary and
`Continuing from …/<name>.lstm`, followed by `make: Leaving directory`, with no
`At iteration …` lines and no `Error`. No new checkpoint is ever written.

Two things combine here.

- **`make` cannot see the failure.** tesstrain runs
  `lstmtraining … 2>&1 | tee -a training.log`. A pipeline's exit status is that
  of its last command, so a crashed `lstmtraining` looks like success. Treat
  `At iteration` lines as the only proof of training. To see the real exit
  status and any crash text, run the `lstmtraining` command printed at the top
  of the log by hand, without the `| tee`, and check `echo $?`.
- **The start-model files are probably inconsistent.** `--continue_from`
  loads weights from `<MODEL_NAME>/<name>.lstm`, which tesstrain extracts from
  the start model only if it is missing or out of date, so a file left over
  from an earlier run is reused. `--old_traineddata` reads the class map from
  `<TESSDATA>/<START_MODEL>.traineddata`, which may have since been
  overwritten by a packaging step. The network summary in the log shows which
  weights actually loaded: `Fc<N>` is that model's output width, and
  `Code range changed from <N> to <M>` should start from the class count of the
  model you meant to continue from. If `N` is the wrong model's width, the
  files are mixed.

Fix: freeze the intended start model as its own file, pick a fresh
`MODEL_NAME`, and rerun (`docs/training.md`, "Fine-tuning on real lines"). That
leaves no stale `.lstm`, checkpoint or proto for the run to pick up.

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
