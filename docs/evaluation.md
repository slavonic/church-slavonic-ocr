# Evaluation

Two scripts: `extract_lines.py` gets real line pairs out of scanned books, and
`cu_eval.py` scores the model on them.

## 1. Extract real lines — `extract_lines.py`

Rasterizes book pages and runs the model to produce line crops with a first-pass
transcription you then correct.

```bash
python3 scripts/extract_lines.py book.pdf --pages 12-15 --out data/real-lines/eval \
        --model cu --dpi 400 --tessdata-dir model
```

- PDF via PyMuPDF; DJVU via `ddjvu` (`sudo apt install djvulibre-bin`).
- Each text line → `<book>_p<pg>_l<ln>.png` + a `.gt.txt` pre-filled with the
  OCR guess. **Correct every `.gt.txt`** against its crop (every titlo and
  superscript), and delete lines where segmentation grabbed noise.
- `--dpi 300–400` is the sweet spot; below ~300 the accent band smears and you
  measure the rasterization, not the model.
- If a page OCRs as noise, its scan quality is the problem, not the model —
  binarize first (Sauvola + deskew) and use `--psm 4`. See `docs/troubleshooting.md`.

Split the results: held-out lines into `data/real-lines/eval/`, adaptation lines
into `data/real-lines/finetune/`. Keep them disjoint.

## 2. Score — `cu_eval.py`

```bash
python3 scripts/cu_eval.py data/real-lines/eval --model cu --tessdata-dir model \
        --report model/eval/report.html --tsv model/eval/metrics.tsv
```

Reports CER and WER, **micro-averaged**, computed two ways:

- **CER (full)** — every character, including combining marks.
- **CER (no marks)** — combining marks stripped from both sides, isolating
  *letterform* accuracy.

The gap between them tells you where errors live:

- no-marks ≈ 0 but full high → errors are **diacritic placement**; oversample
  mark-heavy lines, often tolerable meanwhile.
- no-marks also high → real **letterform** confusion; check the typeface is among
  your faces, and add data / fine-tune.

Outputs a per-line `metrics.tsv` sorted worst-first and a self-contained
`report.html` with each crop, its reference, the OCR, and a character diff — read
this to spot systematic swaps.

The console summary also breaks CER down **per source book** by default,
derived from each file's `<book>_p<pg>_l<ln>.gt.txt` name (the convention
`extract_lines.py` writes), sorted worst-first. This is what tells you whether
error is *concentrated* in specific books — usually a data-volume question for
that book, or a typeface it doesn't share with anything else — versus spread
evenly across all of them, which points at something book-independent (scan
preprocessing, for instance, since that would drag down even your
well-covered books). A filename that doesn't fit the convention is grouped
under `(unrecognized filename pattern)` rather than silently mis-tagged, so it
surfaces instead of hiding. Turn it off with `--no-by-book`.

By default the console summary also splits CER into named buckets by
construct — **melisma** (`~`/`‿`) and **numerals** (`0`-`9`) — since both have
turned out to need separate diagnosis (see `docs/troubleshooting.md`): each
reports lines, characters, its own CER, and what fraction of *total
edit-distance error* it accounts for, against how big a fraction of the text
it actually is. That last ratio is the point — it's a precise answer to "how
much of my CER is construct X" rather than eyeballing the worst-line list, and
it's what showed melisma lines carrying ~25% of total error from under 8% of
the characters even as the aggregate CER improved.

Pass `--split 'name=chars'` (repeatable) to define your own buckets instead —
a line lands in the first bucket it matches, checked in the order given, and
whatever's left becomes "other". `--split ''` disables the whole feature.

Caveats:

- OCR is cached per line as `.hyp.txt`. The tool now compares each cache's
  timestamp against the model file's and **automatically re-OCRs anything
  older than the current model**, printing a count when it does — so a plain
  rerun after retraining is safe by default. `--reocr` still forces every line
  fresh (e.g. if you changed `--psm` and want a clean comparison). If the tool
  can't locate the model file to compare against (a bare language name with no
  `--tessdata-dir`, relying on `TESSDATA_PREFIX`), it warns instead and you
  should pass `--reocr` explicitly.
- The tool flags lines where OCR == reference exactly; a high count means some
  references were left uncorrected and are flattering the score.

## Interpreting the number

- Aim for **≤ 2 % CER on real scans** (≤ 1 % is print-quality). The synthetic
  training-split BCER is always lower and is **not** comparable to this — see
  `docs/troubleshooting.md`.

## Reviewing samples — `review_samples.py`

The ground-truth dir has too many files for a file manager or `*` glob. To
eyeball a few (e.g. hyphenated ones):

```bash
python3 scripts/review_samples.py data/cu-ground-truth --n 8       # montage → review.png
```
