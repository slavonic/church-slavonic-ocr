# Troubleshooting

Problems we have actually hit, grouped by the [manual](manual.md) step where
they show up. Find your step, then the symptom that matches.

Two rules apply throughout:

- **Change one thing at a time.**
- **Find out what's wrong before you rebuild anything.**

| Step | Symptom |
|---|---|
| [Practice data](#practice-data) (2) | [Shell or file manager freezes on the ground-truth folder](#shell-or-file-manager-freezes-on-the-ground-truth-folder) |
| [Training](#training) (3) | [Error rate stuck at 98–100% for a long time](#bcer-stuck-at-98100-for-tens-of-thousands-of-iterations) |
| | [Seems stuck in a loop making `.box` files](#make-training-seems-stuck-making-box-files) |
| | [Output contains schwa, Latin letters or `©`](#foreign-glyphs-in-the-output-schwa-latin-) |
| [Collecting real lines](#collecting-real-lines) (4) | [A page reads as long runs of garbage](#a-page-reads-as-long-runs-of-garbage) |
| [Fine-tuning](#fine-tuning) (6–7) | [`combine_tessdata -u … Error 1`](#combine_tessdata--u--error-1-at-the-start-of-a-fine-tune) |
| | [Log stops after `Continuing from …`, no error](#fine-tune-log-stops-after-continuing-from--make-reports-no-error) |
| | [`Code range changed` / `Must supply the old traineddata`](#code-range-changed-from-n-to-m--must-supply-the-old-traineddata-for-code-conversion) |
| | [Score doesn't change at all after fine-tuning](#score-doesnt-change-at-all-after-fine-tuning) |
| [Reading the scores](#reading-the-scores) (5, 7) | [Training error is low, but the real score is high](#training-error-and-real-error-dont-match) |
| | [Score doesn't change after retraining](#cu_evalpy-shows-no-change-after-retraining) |
| | [The model garbles a real line, but word count and accents are in the right places](#the-model-garbles-a-real-line-but-the-structure-is-right) |
| | [A number at the end of a line turns into garbage](#a-number-at-the-end-of-a-line-turns-into-garbage) |
| | [A mark that's only in the real lines is read as a similar mark](#a-mark-thats-only-in-the-real-lines-is-read-as-a-similar-mark) |
| | [Words come out in all caps or alternating case](#words-come-out-in-all-caps-or-alternating-case) |
| | [ъ/ѣ, ж/ѧ and similar pairs are confused](#ъѣ-жѧ-and-similar-pairs-are-confused) |

---

## Practice data

### Shell or file manager freezes on the ground-truth folder

**Cause:** `data/cu-ground-truth/` holds hundreds of thousands of files, which is
too many for `*` globs, `ls` or a file manager.

**Fix:** look at samples with `make review` (or
`python3 scripts/review_samples.py data/cu-ground-truth --n 8`), which never
lists the whole folder. To count or process files, use
`find data/cu-ground-truth -name '*.gt.txt' | wc -l`, or `find … -exec … {} +`.

---

## Training

### BCER stuck at 98–100% for tens of thousands of iterations

**Symptom:** a from-scratch run (`make train`) shows `BCER train` around
97–100%, barely moving; `BWER` around 100%; `mean rms` low (around 6%). Reading
a training image with the current checkpoint gives empty output or the same
short string (such as `п҆ъ`) whatever the image.

**Cause:** CTC collapse. The network outputs "nothing here" almost everywhere
and has settled there. More iterations won't get it out, and restarting
unchanged just takes another random chance.

**Fix:** use `make train-seeded` ([training.md](training.md#recommended-seeded-training-make-train-seeded)).
It starts from Cyrillic's trained feature layers instead of random ones, and
its watchdog stops and retries a run that collapses anyway. Don't use
tesstrain's `START_MODEL=Cyrillic` instead: that merges Cyrillic's character
list (Latin, schwa, …) into yours.

### `make training` seems stuck making `.box` files

**Cause:** not a loop. This is the one-time conversion of every training pair,
one program run per file, over around 200,000 files.

**Fix:** check it's progressing:
`watch 'find data/cu-ground-truth -name "*.box" | wc -l'`. The count should
climb. Run with `-j$(nproc)` (the `make` targets already do this) to use every
CPU core. If you interrupt it, it carries on from where it stopped next time. If
the count doesn't climb, or the same files are rebuilt every run, suspect
clock skew on a network drive: keep the ground truth on a local disk.

### Foreign glyphs in the output (schwa, Latin, `©`)

**Cause:** the model can only output characters in its character list (its
*unicharset*). Foreign ones mean either:

- **(a)** junk characters in the practice data put them there, or
- **(b)** an old Cyrillic-based character list survived what was meant to be a
  fresh run.

**Find out which:** `head -1 training/cu/unicharset`. Around 200–300 is a clean
Church Slavonic list; thousands means (b). Search the file for Latin letters or
schwa.

**Fix (a):** check the cleaning report printed by `make dataset`
([data-generation.md](data-generation.md#cleaning-and-the-character-set)), fix
the cause, and rebuild the data.

**Fix (b):** reset the character list
([training.md, Resetting the charset](training.md#resetting-the-charset)) and
retrain. Changing the allowed characters (adding digits, `_`, …) **always**
needs this reset.

---

## Collecting real lines

### A page reads as long runs of garbage

**Symptom:** where a short line of text should be, the output is a long run
like `ЩщОощеҹоҹҹ…`.

**Cause:** the model is reading **noise**, not text. The doubled superscripts
and stacked accents are the mark area of the next line plus scan specks. The
problem is the image or how the page was cut into lines, not the model.

**Prove it:** OCR one clean line crop with `--psm 13`. If that reads well, the
model is fine.

**Fix:** re-extract with `--deskew --binarize --psm 4`, and `--margin` if there
are page edges or margin notes
([evaluation.md](evaluation.md#1-extract-real-lines--extract_linespy)).

---

## Fine-tuning

Most of these come from breaking one of the
[fine-tune rules](training.md#rules-that-keep-a-fine-tune-from-breaking). The
quickest fix is usually the same: freeze the start model under its own name,
pick a new `MODEL_NAME`, and run again ([manual, Step 6](manual.md#step-6-fine-tune-on-the-real-lines)).

### `combine_tessdata -u … Error 1` at the start of a fine-tune

**Cause:** tesstrain unpacks the start model from
`$(TESSDATA)/$(START_MODEL).traineddata`. If you don't pass `TESSDATA`, it
looks in a `usr/share/tessdata/` folder next to `DATA_DIR`, which this project
never creates, so the file isn't found.

**Fix:** pass `TESSDATA=$PWD/training` (or `$PWD/model`) on the command line, as
in [the manual](manual.md#step-6-fine-tune-on-the-real-lines). `make train-seeded`
never hits this error.

### Fine-tune log stops after `Continuing from …`; `make` reports no error

**Symptom:** `finetune.log` ends right after the network summary and
`Continuing from …/<name>.lstm`, then `make: Leaving directory`. There are no
`At iteration` lines and no `Error`, and no new checkpoint is written.

**Two causes together:**

- **`make` can't see the failure.** tesstrain runs `lstmtraining … | tee`, and
  `make` only sees `tee`'s result, so a crashed `lstmtraining` looks like
  success. Only `At iteration` lines prove it trained. To see the real error,
  run the `lstmtraining` command from the top of the log yourself, without
  `| tee`, then `echo $?`.
- **The start-model files don't match.** The weights come from
  `training/<MODEL_NAME>/<START_MODEL>.lstm`, which is reused if left over from
  an earlier run. The character map comes from
  `training/<START_MODEL>.traineddata`, which may since have been overwritten by
  packaging. In the log, `Fc<N>` is the loaded network's output width, and
  `Code range changed from <N> to <M>` should start from the class count of the
  model you meant. A wrong `N` means mixed files.

**Fix:** freeze the start model as its own file, choose a new `MODEL_NAME`,
and rerun.

### `Code range changed from N to M!` / `Must supply the old traineddata for code conversion!`

`N` and `M` are numbers of output classes. Where it appears decides whether it
matters.

**At the start of a fine-tune**, right after a line naming
`…/checkpoints/<name>_checkpoint`: harmless in itself. `lstmtraining` found an
old checkpoint with a different class count, rejected it, and started from the
start model instead. It does mean you reused a `MODEL_NAME`; use a new one next
time.

**At packaging (`lstmtraining --stop_training`), ending in
`Failed to read continue from`:** fatal. The checkpoint and the `--traineddata`
you passed have different class counts. Usually the checkpoint is **older than
the proto `traineddata`**: the proto was rebuilt after the ground truth changed,
and the run that should have made a matching checkpoint never ran or never
finished. Check with `ls -l --time-style=full-iso` on both
([manual, Step 7a](manual.md#step-7-package-the-new-model-and-keep-it-only-if-it-is-better)).

**Don't silence it with `--old_traineddata`.** That converts the *old*
checkpoint to the new character list, so you'd publish the previous model under
a new date, and the next test would look as if your change did nothing. Fix it
by getting a new run to finish, under a new `MODEL_NAME`.

### Score doesn't change at all after fine-tuning

**Symptom:** the test score is identical to before, and a character that exists
only in the fine-tune lines never appears in the output.

**Cause:** the fine-tuned model was never packaged and installed. tesstrain's
`training` target only writes checkpoints, and nothing copies a result into
`model/`. `make eval` then keeps testing the old `model/cu.traineddata`, with
no error.

**Check:** `ls -la model/cu.traineddata training/<name>/checkpoints/`. If the
checkpoints are newer than the model, this is it.

**Fix:** package and test the run by its own name, then copy it into `model/`
([manual, Step 7](manual.md#step-7-package-the-new-model-and-keep-it-only-if-it-is-better)).

---

## Reading the scores

### Training error and real error don't match

The `BCER` printed during training is measured on the practice data, which is
much easier than real scans. Clean practice data scores under 1%. Harder,
blurred, hyphenated practice data scores higher, and a *higher* training error
on more realistic data can mean a *better* model on real books. Only compare
training errors between runs that used the same practice data.

The real verdict is `make eval` on the real test lines
([manual, Step 5](manual.md#step-5-measure-how-well-the-model-reads)). The goal
there is 2% CER or less.

### `cu_eval.py` shows no change after retraining

The test caches each model reading as a `.hyp.txt` file and only re-reads lines
when it can tell the model is newer. When you switch between models, or run
without `--tessdata-dir`, pass `--reocr`. Details:
[evaluation.md, Caveats](evaluation.md#caveats).

### The model garbles a real line, but the structure is right

**Symptom:** the word count and the positions of commas and accents are right,
and parts of words are correct, but much of the text is wrong.

**Test:** OCR a clean line straight from `data/cu-ground-truth`.

- **Garbled too:** the installed model doesn't match the trained network,
  usually from retraining after changing the allowed characters without a reset.
  Reset the character list ([training.md](training.md#resetting-the-charset))
  and rebuild.
- **Reads fine:** the model is healthy. Real scans just look different from the
  practice data. Clean the scans up (`--binarize`) and fine-tune on real lines.

Also make sure you're testing the model you think you are: always pass
`--tessdata-dir`.

### A number at the end of a line turns into garbage

**Symptom:** the line's text reads well, then a trailing page number, verse
number or letter-numeral (`г҃`) turns into junk, often after a wide gap
(`…и҆сповѣ́данїи 73` → `…и҆сповѣ́данїи :с:с:с::с:стз`).

**First, measure.** A few striking lines in the worst-lines list aren't a
pattern. The `numerals` row in `cu_eval.py`'s per-construct breakdown
([evaluation.md](evaluation.md#per-construct-breakdown)) shows what share of
the total error comes from lines with digits. In this project's test set it was
one line and 1.5% of the error, so there was nothing to fix. (Cyrillic
letter-numerals aren't counted there.)

**If it is a real share:** the wide gap suggests a margin number or running
header that got cut into the same line. Confirm by looking at the crops in
`report.html`, then either re-extract with `--margin` to leave it out
([evaluation.md](evaluation.md#1-extract-real-lines--extract_linespy)), or add
enough lines with numbers to `finetune/` for the model to learn them.

### A mark that's only in the real lines is read as a similar mark

**Symptom:** a mark that appears in your corrected real lines but never in the
practice data comes out as a different, similar-looking mark that *is* in the
practice data. Our case: a syllable dash labeled `~` in chant lyrics was read
as `_` (the hyphen label) or as stray dashes and spaces.

Look at a few misread crops in `report.html` and ask: **could you tell the two
marks apart by eye?**

- **No (same mark, two labels):** the model is being taught two answers for the
  same picture, which it can't learn. Use **one label** for marks that look
  alike. This project uses `_` for both
  ([data/README.md](../data/README.md#ground-truth-conventions)).
- **Yes (a different but rare mark):** a character only in the fine-tune lines
  starts from nothing, against characters with long training behind them. It
  needs exposure: more real lines containing it, and more fine-tune iterations.

Even with consistent labels, lines with a dash in the middle of a word can
stay weak, because the practice data never has one. If so, render syllabified
corpus text with a drawn dash as the divider.

### Words come out in all caps or alternating case

**Symptom:** a single word has mostly the right letters but the wrong case
throughout (`Сла́ва,` → `СѧЛА́ВАѧ`).

**Check the crop in `report.html` first.** Liturgical books often print opening
words, exclamations (`Сла́ва`, `Ны́нѣ`) or headings in large decorative
capitals, which the practice data never draws. If the crop shows decorative
capitals, either agree on how to type such words, or add examples of them to
the fine-tune lines. If the crop shows ordinary lowercase print, it's a real
model confusion.

### ъ/ѣ, ж/ѧ and similar pairs are confused

The stroke that tells them apart (the yat's crossbar, the yus's bowl) is
exactly what a poor scan blurs away. A higher `--dpi` and `--binarize` when
extracting recover much of it; fine-tuning on real lines in that typeface
teaches the rest.
