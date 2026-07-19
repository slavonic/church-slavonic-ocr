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

## Recommended: seeded training (`make train-seeded`)

Two naive strategies both fail here, in opposite ways:

- `START_MODEL=Cyrillic` **merges** Cyrillic's unicharset into yours
  (`merge_unicharsets` in the tesstrain Makefile), so the model can emit
  schwa, Latin, `©` — classes that were never in your ground truth.
- Pure from-scratch keeps the charset clean but random init can fall into the
  **CTC all-blank basin**: BCER pinned ~98–100% for tens of thousands of
  iterations, low rms, near-constant short decode. Whether it escapes is a
  dice roll on the init.

`scripts/train_seeded.py` (run via `make train-seeded`) threads between them:

1. builds the unicharset, proto model, and train/eval lists from **your
   ground truth only** (tesstrain's from-scratch path — no merge), then
   **audits the unicharset** and refuses to train if any non-CU class is in it;
2. extracts the LSTM from stock `Cyrillic.traineddata` and continues from it
   with `--old_traineddata`, which makes `lstmtraining` **remap the output
   layer to your clean charset** — Cyrillic contributes only its lower
   feature layers ("Cyrillic's eyes, your alphabet"), so training starts
   outside the blank basin and converges much faster;
3. a **watchdog** parses training progress and kills the run if BCER is still
   ≥90% past ~6k iterations with no downtrend, then retries with a doubled
   learning rate (a collapsed attempt costs minutes, not a night). A slow but
   genuinely falling BCER is left alone.

```bash
make train-seeded                 # or directly:
python3 scripts/train_seeded.py --tesstrain ../tesstrain --max-iterations 100000
```

Expect BCER to break downward within the first 1–2k iterations. The final
model lands at `training/cu.traineddata` (and `model/` via the make target).

## Plain from-scratch run

The unseeded path — clean charset, but subject to the init dice-roll above.
`make train` wires tesstrain to read this repo's ground truth and write
scratch to `training/`:

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
make reset-charset
make train
```

`reset-charset` wraps tesstrain's own `clean-output` (`make -C $(TESSTRAIN)
clean-output MODEL_NAME=$(MODEL_NAME) DATA_DIR=$(CURDIR)/training`, using this
repo's Makefile variables rather than a `$TESSTRAIN` you'd have to export
yourself) plus removing the stale `.traineddata` copies. It drops the
unicharset/recoder/checkpoints but keeps the `.box`/`.lstmf` files
(charset-independent), so you don't pay the long regeneration again.

This repo also has its own, blunter version of the same idea — `make clean-train`
wipes the entire `training/` scratch dir (`rm -rf training/*`), including the
`.box`/`.lstmf` files `reset-charset` preserves:

```bash
make clean-train    # rm -rf training/* (keeps data/cu-ground-truth and model/)
make train
```

Reach for `reset-charset` when you only need a fresh charset and want to keep the
expensive `.lstmf` generation; reach for `clean-train` when you want `training/`
back to a truly empty slate (e.g. it's in some inconsistent state you don't want
to reason about) and are fine re-paying that cost.

## Sanity checks

```bash
head -1 training/cu/unicharset                 # class count: ~200-300 = clean CU
combine_tessdata -u model/cu.traineddata /tmp/x. && grep -nP '^[A-Za-zəә]\t' /tmp/x.lstm-unicharset
```
