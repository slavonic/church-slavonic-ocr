# Training

Training uses [tesstrain](https://github.com/tesseract-ocr/tesstrain) as the
harness; this repo supplies the ground truth and holds the result. tesstrain is
**not part of this repo** — it's not a submodule, and it isn't installed inside
this checkout. It's kept as a separate, independent checkout outside this
repo's tree entirely (set `TESSTRAIN=` for the Makefile, default `../tesstrain`,
a sibling of this directory). The clone command below must be run from
*outside* this repo (e.g. one level up) — cloning it inside `church-slavonic-ocr/`
leaves an untracked directory that doesn't match the default path.

## Prerequisites

Tesseract **and its training tools** (`lstmtraining`, `combine_lang_model`, …).
The engine installs from apt; the training tools usually must be built:

```bash
sudo apt install tesseract-ocr
which lstmtraining || {           # build the training tools if missing
  sudo apt install libtool pkg-config libleptonica-dev libicu-dev \
                   libpango1.0-dev libcairo2-dev g++ automake
  git clone https://github.com/tesseract-ocr/tesseract && cd tesseract
  ./autogen.sh && ./configure && make -j"$(nproc)"
  sudo make install && make training && sudo make training-install && sudo ldconfig
  cd ..
}
cd ..                                                     # OUT of church-slavonic-ocr/
git clone https://github.com/tesseract-ocr/tesstrain      # the harness, as a SIBLING dir
cd church-slavonic-ocr                                    # back into this repo
```

## From-scratch run

Church Slavonic is far enough from any base model that from-scratch with a
clean, corpus-built unicharset is the right default (fine-tuning `rus` fights its
priors; a stale Cyrillic charset reintroduces foreign glyphs). `make train`
wires tesstrain to read this repo's ground truth and write scratch to `training/`:

```bash
make train                       # MAX_ITERATIONS=100000 JOBS=$(nproc) by default
```

which expands to, roughly:

```bash
make -C $TESSTRAIN training -j"$(nproc)" \
  MODEL_NAME=cu MAX_ITERATIONS=100000 \
  DATA_DIR=$PWD/training GROUND_TRUTH_DIR=$PWD/data/cu-ground-truth
cp training/cu.traineddata model/
```

- **`-j$(nproc)` is important.** The first phase generates a `.box` then a
  `.lstmf` per line — one subprocess each. Over ~200k pairs this is slow and
  serial looks like a hang; parallelism turns hours into much less. Boxes/lstmf
  are `.PRECIOUS`, so an interrupted run resumes rather than restarts.
- **From-scratch is slow to start.** Character error sits near 99 % (an all-blank
  CTC collapse) for thousands of iterations, then breaks downward. Don't judge it
  before ~25–30k; watch the trend with `scripts/watch_training.py`.

## Fine-tuning on real lines

`data/real-lines/finetune/` isn't part of this repo's automated pipeline — it's
built by hand from real scans with `scripts/extract_lines.py`, which rasterizes
book pages and runs the current model to produce a line crop plus a first-pass
`.gt.txt` guess per line; you then correct every `.gt.txt` against its crop
(titla, superscripts, everything) before it's usable as ground truth. See
`docs/evaluation.md` for the extraction command and correction workflow — the
same process also produces `data/real-lines/eval/`, so make sure the lines you
put in `finetune/` are disjoint from `eval/`, or the held-out score stops
meaning anything.

The highest-leverage step once the synthetic model is clean. Copy corrected real
lines into the ground-truth dir (so the charset picks up their glyphs), keeping
the eval set out, and continue from the current model:

```bash
cp data/real-lines/finetune/* data/cu-ground-truth/
make -C $TESSTRAIN training MODEL_NAME=cu START_MODEL=cu \
  DATA_DIR=$PWD/training GROUND_TRUTH_DIR=$PWD/data/cu-ground-truth \
  MAX_ITERATIONS=<a few thousand>
```

## How the unicharset is built

You never invoke this directly — `make train` triggers it as an early tesstrain
phase — but it's worth knowing where `training/cu/unicharset` comes from when
debugging a garbled model. tesstrain concatenates every `.gt.txt` line under
`GROUND_TRUTH_DIR` into `training/cu/all-gt`, then runs Tesseract's
`unicharset_extractor` over that file to collect the distinct characters actually
present in the ground truth (plus `combine_lang_model` to fold in the recoder/
Unicode properties). The result is a from-scratch charset scoped to exactly what
your corpus contains — no leftover glyphs from `rus` or any other base model. This
is why the charset always tracks the ground truth: add `_` for hyphenation or
widen `--allow-extra`, regenerate the dataset, and the next `unicharset_extractor`
run picks up the new symbols automatically. It's also why stale state is
dangerous — an old `unicharset` built before an allow-set change doesn't know
about the new characters until you clear it (see below).

## Resetting the charset

Changing the allow-set (e.g. adding digits or `_`) requires rebuilding the
unicharset, or old state collides with new — a classic cause of a model that
garbles even its own synthetic lines:

```bash
make -C $TESSTRAIN clean-output MODEL_NAME=cu DATA_DIR=$PWD/training
rm -f training/cu.traineddata model/cu.traineddata
make train
```

`clean-output` drops the unicharset/recoder/checkpoints but keeps the `.lstmf`
(charset-independent), so you don't pay the long regeneration again.

## Sanity checks

```bash
head -1 training/cu/unicharset                 # class count: ~200-300 = clean CU
combine_tessdata -u model/cu.traineddata /tmp/x. && grep -nP '^[A-Za-zəә]\t' /tmp/x.lstm-unicharset
```
