# Training

Training uses [tesstrain](https://github.com/tesseract-ocr/tesstrain) as the
harness; this repo supplies the ground truth and holds the result. tesstrain is
kept as a separate checkout (set `TESSTRAIN=` for the Makefile, default `../tesstrain`).

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
git clone https://github.com/tesseract-ocr/tesstrain     # the harness
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

The highest-leverage step once the synthetic model is clean. Copy corrected real
lines into the ground-truth dir (so the charset picks up their glyphs), keeping
the eval set out, and continue from the current model:

```bash
cp data/real-lines/finetune/* data/cu-ground-truth/
make -C $TESSTRAIN training MODEL_NAME=cu START_MODEL=cu \
  DATA_DIR=$PWD/training GROUND_TRUTH_DIR=$PWD/data/cu-ground-truth \
  MAX_ITERATIONS=<a few thousand>
```

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
