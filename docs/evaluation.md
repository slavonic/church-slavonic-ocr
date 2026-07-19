# Evaluation

Two scripts: `extract_lines.py` gets real line pairs out of scanned books, and
`cu_eval.py` scores the model on them.

## 1. Extract real lines — `extract_lines.py`

Rasterizes book pages and runs the model to produce line crops with a first-pass
transcription you then correct.

```bash
python3 scripts/extract_lines.py book.pdf --pages 12-15 --out data/real-lines/staging \
        --model cu --dpi 400 --tessdata-dir model
```

Point `--out` at `data/real-lines/staging/` — a scratch area, not `eval/` or
`finetune/` directly. You don't yet know which lines are good, which are noise,
or which set a given line belongs in, so nothing should land in the committed
directories until you've reviewed it:

- PDF via PyMuPDF; DJVU via `ddjvu` (`sudo apt install djvulibre-bin`).
- Each text line → `<book>_p<pg>_l<ln>.png` + a `.gt.txt` pre-filled with the
  OCR guess. **Correct every `.gt.txt`** against its crop (every titlo and
  superscript), and delete pairs where segmentation grabbed noise.
- `--dpi 300–400` is the sweet spot; below ~300 the accent band smears and you
  measure the rasterization, not the model.
- If a page OCRs as noise, its scan quality is the problem, not the model —
  add `--deskew --binarize` (see below) and use `--psm 4`. See `docs/troubleshooting.md`.

### Preprocessing noisy scans — `--deskew` / `--binarize`

Both are opt-in and off by default (clean scans don't need them and Sauvola
binarization does throw away gray information, so don't reach for it
reflexively); they run on the rasterized page *before* line segmentation/OCR
and before crops are cut, so the boxes tesseract finds and the crop you end up
correcting both see the same cleaned-up image:

```bash
python3 scripts/extract_lines.py book.pdf --pages 12-15 --out data/real-lines/staging \
        --model cu --dpi 400 --tessdata-dir model --deskew --binarize --psm 4
```

- **`--deskew`** estimates page rotation with a projection-profile search
  (rotate a downscaled ink mask through candidate angles, keep the one whose
  horizontal ink-per-row profile has the most variance — tightly-packed text
  baselines produce sharp peaks; a skewed page smears them out) and rotates the
  full-resolution page to correct it. `--deskew-range`/`--deskew-step` control
  the search (default ±5°, 0.2° steps).
- **`--binarize`** applies Sauvola local-threshold binarization: each pixel is
  judged against a threshold derived from its own neighborhood's mean and
  contrast, rather than one global cutoff — so uneven lighting or aged/foxed
  paper that would blow out a global threshold in one area and crush it in
  another gets a locally-appropriate cutoff instead. `--sauvola-window` (default
  25px) sets the neighborhood size; `--sauvola-k` (default 0.2) trades off how
  strict the threshold is.

Once every `.gt.txt` in the staging dir is corrected (or deleted, for noise),
each surviving `<stem>.png` + `<stem>.gt.txt` pair still needs to land in
whichever of the two committed directories it belongs in — **as a pair**, they
must travel together, which is exactly what's error-prone about doing it by
hand with `cp`/`mv` (typo the destination, move only one of the two files,
copy instead of move and end up with the same line staged *and* committed).
`review_staging.py` (below) does this as part of the same review pass instead.
`staging/` is gitignored and meant to be cleared out after each batch, not
accumulated in.

### Correcting the staging batch — `review_staging.py`

Editing dozens of `.gt.txt` files by hand against their crops (opening each PNG,
retyping the text, deleting the noise ones, then sorting the survivors into
`eval/`/`finetune/`) is exactly the kind of thing worth a small UI for.
`scripts/review_staging.py` serves the staging dir as a local web page: crop on
top, an editable textarea prefilled with the current guess below, and buttons
to save, delete, or file the pair straight into its destination:

```bash
make review-staging                          # data/real-lines/staging, opens a browser tab
python3 scripts/review_staging.py --dir data/real-lines/staging --port 8080
```

- **Save & next** (`Ctrl+Enter`) writes the edited text to that `.gt.txt` and
  advances to the next pair, without moving it out of staging.
- **→ eval** / **→ finetune** (`Ctrl+E` / `Ctrl+F`) save the edited text, then
  move *both* `.png` and `.gt.txt` — together, atomically, never one without
  the other — straight into `data/real-lines/eval/` or `data/real-lines/finetune/`.
  If a file with that name already exists at the destination, the move is
  refused (with the conflict shown in the page) rather than silently
  overwriting anything; resolve that by hand.
- **Delete pair** (`Ctrl+Delete`) removes both the `.png` and `.gt.txt` —
  segmentation noise, not a real line — after a confirmation prompt (there's no
  undo).
- **Skip** moves on without changing anything.
- The textarea renders in a real CU face (Ponomar by default, `--font` to pick
  another `.ttf`) instead of the system monospace font, so titla and diacritics
  in what you're typing are visibly correct while you correct it, not just
  ASCII-transliterated guesswork.
- A pair drops out of the queue for that run once saved, moved, or deleted, so
  repeatedly hitting one action works through the whole batch; closing and
  reopening the tool re-offers anything still left in staging (harmless — a
  pair already moved out is simply gone from the directory it's scanning).
- `--eval-dir`/`--finetune-dir` override the destinations if you're not
  working from the repo root.

It's a stdlib-only local server (no new dependency), bound to `127.0.0.1`.

Split the corrected results between two directories with different jobs:

- **`data/real-lines/eval/`** — the held-out evaluation set. This is the only
  thing `cu_eval.py` scores against (§2 below) and the only number that reflects
  real-scan accuracy (see `docs/troubleshooting.md` on why the synthetic
  training-split score isn't comparable). **Never train or fine-tune on these
  lines** — if the model has seen them, the score just measures memorization,
  not generalization to scans it hasn't seen.
- **`data/real-lines/finetune/`** — corrected lines used to adapt the model to
  real scans (see `docs/training.md`). For a fine-tune run these get copied
  into `data/cu-ground-truth/` alongside the synthetic pairs, so the charset
  and the model both pick up real-scan glyphs/quirks the synthetic corpus
  doesn't reproduce.

Keep the two sets strictly disjoint — a line that ends up in both defeats the
held-out set's purpose the moment it's trained on.

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

Caveats:

- OCR is cached per line as `.hyp.txt`. **After retraining, pass `--reocr`** or
  you will re-score the old model's outputs.
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
