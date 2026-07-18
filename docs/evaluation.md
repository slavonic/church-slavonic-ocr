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
  binarize first (Sauvola + deskew) and use `--psm 4`. See `docs/troubleshooting.md`.

Once every `.gt.txt` in the staging dir is corrected (or deleted, for noise),
move each surviving `<stem>.png` + `<stem>.gt.txt` pair — as a pair, they must
travel together — into whichever of the two committed directories it belongs
in. `staging/` is gitignored and meant to be cleared out after each batch, not
accumulated in.

### Correcting the staging batch — `review_staging.py`

Editing dozens of `.gt.txt` files by hand against their crops (opening each PNG,
retyping the text, deleting the noise ones) is exactly the kind of thing worth
a small UI for. `scripts/review_staging.py` serves the staging dir as a local
web page: crop on top, an editable textarea prefilled with the current guess
below, and Save/Delete/Skip:

```bash
make review-staging                          # data/real-lines/staging, opens a browser tab
python3 scripts/review_staging.py --dir data/real-lines/staging --port 8080
```

- **Save & next** (`Ctrl+Enter`) writes the edited text to that `.gt.txt` and
  advances to the next pair.
- **Delete pair** (`Ctrl+Delete`) removes both the `.png` and `.gt.txt` —
  segmentation noise, not a real line — after a confirmation prompt (there's no
  undo).
- **Skip** moves on without changing anything.
- Once a pair is saved or deleted it drops out of the queue for that run, so
  repeatedly hitting save-and-next works through the whole batch; closing and
  reopening the tool re-offers everything still on disk (harmless, since
  staging is meant to be a one-batch-at-a-time scratch area anyway).

It's a stdlib-only local server (no new dependency), bound to `127.0.0.1`; it
doesn't touch `eval/`/`finetune/` or decide the split — that's still your call,
made by moving the corrected pairs afterward as described above.

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
