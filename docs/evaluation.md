# Evaluation

Reference for [manual](manual.md) Steps 4 and 5: getting real lines out of
scanned books with `extract_lines.py`, and scoring a model on them with
`cu_eval.py`. For the step-by-step version, follow the manual; this page lists
the options and explains the output.

## 1. Extract real lines — `extract_lines.py`

Rasterizes book pages, runs the model to find the lines and make a first-pass
transcription, and writes each line as a crop plus an editable `.gt.txt`.

```bash
python3 scripts/extract_lines.py book.pdf --pages 12-15 --out data/real-lines/staging \
        --model cu --dpi 400 --tessdata-dir model
```

- PDF via PyMuPDF; DJVU via `ddjvu` (`sudo apt install djvulibre-bin`).
- Each line becomes `<book>_p<pg>_l<ln>.png` plus a `.gt.txt` pre-filled with
  the OCR guess. `<book>` is the input file name without its extension; it is
  also how `cu_eval.py` groups scores per book and how
  [data/real-lines/README.md](../data/real-lines/README.md) counts sources.
- Write to `data/real-lines/staging/`, then correct and sort the lines with
  `make review-staging` ([manual, Step 4c](manual.md#step-4-collect-real-lines-from-scanned-books)).
  It moves each pair into `eval/` or `finetune/` together. Keep those two sets
  disjoint: never train on an eval line.

| Option | Default | Use |
|---|---|---|
| `--pages` | `all` | `12-15`, `3,5,7` or `all`; 1-based PDF page numbers |
| `--dpi` | `400` | 300–400 is the sweet spot. Below ~300 the accent band smears and you measure the rasterization, not the model. |
| `--psm` | `6` | Tesseract page segmentation mode for finding lines. Try `4` (single column) for noisy pages. |
| `--pad` | `6` | Pixels of padding around each crop |
| `--margin` | none | Drop a page border before finding lines; see below |
| `--margin-mode` | `crop` | `crop` or `mask`; see below |
| `--deskew` | off | Correct page tilt before finding lines |
| `--binarize` | off | Sauvola local-threshold black-and-white conversion, for grey, stained or uneven scans |

`--margin`, `--deskew` and `--binarize` change the page once, before lines
are found *and* before crops are cut, so the saved crops show exactly what the
model saw.

**`--margin TOP,RIGHT,BOTTOM,LEFT`** removes a printed frame, a scan edge, a
running header or a marginal verse number. Values are pixels (`40`, `40px`) or
a share of the page (`5%`). Like CSS, one value applies to all sides and two
mean vertical,horizontal. For example, `--margin 12%` trims 12% off every
side. `--margin-mode crop` (the default) cuts the border off.
`--margin-mode mask` paints it white but keeps the page size, which leaves a
quiet zone around the text block. Try `mask` if cropping makes lines at the
edge come out badly.

If a page OCRs as noise, the scan is the problem, not the model. Use
`--deskew --binarize --psm 4`
([troubleshooting](troubleshooting.md#a-page-reads-as-long-runs-of-garbage)).

## 2. Score — `cu_eval.py`

`make eval` runs this on `model/cu.traineddata`. To score another model, name it
and the folder it's in:

```bash
python3 scripts/cu_eval.py data/real-lines/eval --model cu_v3 --tessdata-dir training \
        --reocr --report training/cu_v3_report.html --tsv training/cu_v3_metrics.tsv
```

### The headline numbers

CER and WER are **micro-averaged** (total edits ÷ total reference characters or
words), and computed two ways:

- **CER (full):** every character, including combining marks.
- **CER (no marks):** combining marks stripped from both sides, which isolates
  *letterform* accuracy.

The gap between them shows where the errors are:

- no-marks near 0 but full high → errors are **diacritic placement**.
  Oversample lines heavy in marks; often tolerable in the meantime.
- no-marks also high → real **letterform** confusion. Check the typeface is
  close to one of the training fonts, and add data or fine-tune.

Aim for **≤ 2% CER (full)** on real scans; ≤ 1% is print quality. The `BCER`
printed during training is always lower and isn't comparable
([troubleshooting](troubleshooting.md#training-error-and-real-error-dont-match)).

Before comparing, both texts are normalized in one way: spaces around the `_`
divider are removed, so `foo_bar`, `foo _bar`, `foo_ bar` and `foo _ bar` all
score the same.

### Per-book breakdown

The summary also lists CER **per source book**, worst first, using the
`<book>` part of each file name. Error concentrated in a few books usually
means those books need more lines, or use a typeface nothing else shares.
Error spread evenly points at something that affects every book, such as scan
preprocessing. Files whose names don't fit the pattern are listed under
`(unrecognized filename pattern)`. Turn this off with `--no-by-book`.

### Per-construct breakdown

The summary also splits CER into buckets by what a line contains. Each bucket
shows its line and character counts, its own CER, and its **share of the total
error** compared with its share of the text. That comparison is the point: it
answers "how much of my CER comes from X" precisely, which the worst-lines list
can't.

The default buckets are `melisma` (`~`, `‿`) and `numerals` (`0`–`9`). Since
dividers are now typed as `_`
([data/README.md](../data/README.md#ground-truth-conventions)), `melisma`
matches nothing and is skipped. To get a dividers bucket, pass your own
buckets. `--split` replaces the defaults, so list every bucket you want:

```bash
python3 scripts/cu_eval.py data/real-lines/eval --model cu --tessdata-dir model \
  --split 'dividers=_' --split 'numerals=0123456789'
```

A line goes in the first bucket it matches; the rest are "other".
`--split ''` turns the breakdown off.

### Confidence intervals and comparing models

The test set is a few hundred lines, so a score carries real sampling noise.

- **Interval on each score.** `CER (full)` and `CER (no marks)` print a 95%
  bootstrap interval. It resamples whole *lines* with replacement
  (`--bootstrap N` resamples, default 2000, fixed seed; `0` turns it off).
  Lines, not characters, are the unit, because errors cluster within lines:
  one bad crop yields many errors.
- **Comparing two models: `--compare OLD_TSV`.** This takes the `--tsv` file
  from an earlier run and runs a *paired* bootstrap over the lines both runs
  share. Each resample draws the same lines for both models, so the shared
  difficulty of the test set cancels out. That makes it much tighter than
  comparing the two separate intervals.
  - It prints the change in points with its 95% interval.
  - The verdict is **BETTER** if the whole interval is below 0, **WORSE** if
    the whole interval is above 0, and **NO CLEAR DIFFERENCE** otherwise.
  - If the two runs scored different line sets, only the shared lines are
    compared, and it says so.
- `metrics.tsv` has a `char_edits` column for this. Older TSVs without it
  still work: the edits are rebuilt from `cer_full` × `ref_chars`.

```bash
python3 scripts/cu_eval.py data/real-lines/eval --model cu_v3 --tessdata-dir training \
  --reocr --tsv training/cu_v3_metrics.tsv --compare model/eval/metrics.tsv
```

### Report files

- `--report report.html`: a self-contained page. It opens with an **error
  breakdown** over all lines, then shows the worst lines (`--top`, default 60),
  each with its crop, reference, OCR and a character diff.
  - Errors are counted per *glyph unit*, meaning a base letter plus its
    combining marks (NFD), aligned by edit distance. A wrong accent is one
    error, not a deleted and an inserted character.
  - Each error is classed as wrong letter, right letter with wrong marks,
    missing or extra letter, or a space or `_` error.
  - Tables list the most frequent mark confusions (by Unicode mark name), the
    letters that most often get wrong marks, and the most frequent letter,
    space and divider pairs. Each row gives up to three example lines, linked
    when the line is shown further down.
  - It also counts lines where one letter carries the same mark twice. The
    model does this; correct text never does. The page names any *reference*
    line with a repeated mark, since that is a typo.

  The terminal summary prints the counts by kind and the repeated-mark count.
- `--tsv metrics.tsv`: one row per line, worst first. The last column,
  `char_edits`, is the line's edit count, which `--compare` reads.

### Caveats

- **The OCR cache.** Each line's OCR is cached beside its image as
  `<image>.hyp.txt`. A cache older than the model file is redone automatically,
  with a note in the summary. `--reocr` forces every line to be redone.
- **Switching models.** The automatic check only asks whether a cache is older
  than the model being scored. If you score a newer model and then an older one,
  the newer model's cache looks fresh and is reused, so you'd be scoring the
  wrong model. **Pass `--reocr` whenever you change `--model`.**
- **No `--tessdata-dir`.** Without it (relying on `TESSDATA_PREFIX`), the tool
  can't find the model file to compare dates with. It warns you; pass `--reocr`.
- **Uncorrected lines.** The summary counts lines where the OCR matches the
  reference exactly. A high count suggests some references were never
  corrected and are flattering the score.

## Re-checking lines where OCR equals the reference — `review_identical.py`

`cu_eval.py` prints `NOTE: N line(s) had OCR == reference exactly` when some
`.gt.txt` files match the cached `.png.hyp.txt`. Those lines score 0% CER, so
if their references were never checked against the image they bias CER down.
`review_identical.py` serves a web page (like `review_staging.py`) that walks
through exactly those lines: the crop, the editable `.gt.txt`, and the cached
OCR for reference.

```bash
make review-identical                       # data/real-lines/eval, port 8767
python3 scripts/review_identical.py --dir data/real-lines/finetune
python3 scripts/review_identical.py --port 8080 --no-browser
```

- It compares with the same rule as `cu_eval.py` (spaces around `_` ignored),
  so the count matches the NOTE.
- It reads the cached `.hyp.txt` files, so refresh them first with
  `cu_eval.py ... --reocr`; stale caches give a wrong list.
- The list is fixed when the script starts. Corrected lines stay in it;
  restart to rescan.
- Save writes the `.gt.txt` in place. Ctrl+Enter saves and advances,
  Ctrl/Alt+Left/Right move without saving.

## Reviewing practice samples — `review_samples.py`

The practice-data folder has too many files for a file manager or a `*` glob.
To look at a few (by default, ones containing `_`):

```bash
make review                                                         # → model/eval/review.png
python3 scripts/review_samples.py data/cu-ground-truth --contains 'ѣ' --n 12 --out sheet.png
```
