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

**`make train` and `make train-seeded` overwrite `model/cu.traineddata`
automatically.** Before retraining, keep the model you may want back:
`cp model/cu.traineddata training/cu_v1.traineddata` (a frozen copy is also what
you want as a fine-tune start model; see "Fine-tuning on real lines").

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

The highest-leverage step once the synthetic model is clean: continue from the
current model on corrected real lines, with the eval set kept out.

### Rules that keep a fine-tune from breaking

Each of these comes from a run that failed silently.

1. **Every run gets its own `MODEL_NAME`, and the start model is a frozen
   file under a different name** (`cu_seed` → `cu_v2`). With
   `START_MODEL=cu MODEL_NAME=cu`, the start model and the output share
   `training/cu/` and `training/cu.traineddata`, so checkpoints, the extracted
   `cu.lstm` and the proto traineddata from earlier runs get reused or rejected
   without any error.
2. **Never write a packaged model to a path tesstrain reads as a start
   model.** With `TESSDATA=$PWD/training`, tesstrain passes
   `training/<START_MODEL>.traineddata` to `lstmtraining` as
   `--old_traineddata` (its class map). Packaging a fine-tuned model over that
   file leaves the next run with weights from one network and a class map from
   another.
3. **Keep a copy of every model you might want to roll back to.** Package into
   a new filename each time, and copy into `model/` only after the new model
   beats the current one on the real eval set. `model/cu.traineddata` is the
   published copy, not a working file.
4. **Charsets only grow across a `START_MODEL` run.** tesstrain builds the new
   unicharset with `merge_unicharsets` (start model + your ground truth), so a
   class the start model learned stays even if you have since removed that
   character from the ground truth. To genuinely drop a character, start from
   a model that never had it.
5. **Success means `At iteration` lines, not the absence of `Error 1`.**
   tesstrain pipes `lstmtraining` through `tee`, so `make` reports `tee`'s exit
   status and a crashed `lstmtraining` can look like a clean run. See
   `docs/troubleshooting.md`.

### Recipe

```bash
# one time: freeze the start model (e.g. the seeded model from `make train-seeded`)
cp <seeded cu.traineddata> training/cu_seed.traineddata

cp data/real-lines/finetune/* data/cu-ground-truth/
make -C $TESSTRAIN training MODEL_NAME=cu_v2 START_MODEL=cu_seed \
  DATA_DIR=$PWD/training GROUND_TRUTH_DIR=$PWD/data/cu-ground-truth \
  TESSDATA=$PWD/training \
  MAX_ITERATIONS=10000 2>&1 | tee finetune.log
```

As a rule of thumb, use around 10,000 iterations when continuing from a model
that has never seen real lines, and around 5,000 when continuing from one that
already has; compare the numbered checkpoints rather than trusting the last.

`TESSDATA` must be passed explicitly. tesstrain unpacks `START_MODEL` with
`combine_tessdata -u $(TESSDATA)/$(START_MODEL).traineddata …`, and its default
for `TESSDATA` is a `usr/share/tessdata/` folder next to `DATA_DIR`, a path this
repo never creates. Left at the default, that step fails with
`combine_tessdata -u … Error 1` and no further explanation.

**Check that it is really training.** Near the top of `finetune.log` you should
see the network summary ending in `Fc<N>` and a line
`Code range changed from <start model classes> to <merged classes>`, followed
soon by `At iteration …` lines. If the log stops after `Continuing from …` with
no iteration lines, the run did not train even if `make` printed no error.

### Package and evaluate

`training` only writes checkpoints; it does not produce a usable
`.traineddata`. Package under the run's own name, then evaluate that file
directly:

```bash
ls -l --time-style=full-iso training/cu_v2/cu_v2.traineddata \
                            training/cu_v2/checkpoints/cu_v2_checkpoint
#   the checkpoint must be NEWER than the proto traineddata

lstmtraining --stop_training \
  --continue_from training/cu_v2/checkpoints/cu_v2_checkpoint \
  --traineddata training/cu_v2/cu_v2.traineddata \
  --model_output training/cu_v2.traineddata

python3 scripts/cu_eval.py data/real-lines/eval --model cu_v2 --tessdata-dir training \
  --report model/eval/report.html --tsv model/eval/metrics.tsv

# only once it beats the model currently in model/:
cp training/cu_v2.traineddata model/cu.traineddata
```

If the checkpoint is older than the proto, no training happened after the
proto was built (the proto is rebuilt whenever the ground truth changes).
Packaging then fails with `Code range changed … Must supply the old
traineddata`; do not paper over it with `--old_traineddata` (see
`docs/troubleshooting.md`).

## Resetting the charset

Changing the allow-set (e.g. adding digits or `_`) requires rebuilding the
unicharset, or old state collides with new, a classic cause of a model that
garbles even its own synthetic lines. For a from-scratch or seeded run:

```bash
make -C $TESSTRAIN clean-output MODEL_NAME=cu DATA_DIR=$PWD/training
rm -f training/cu.traineddata
make train-seeded
```

`clean-output` drops the unicharset/recoder/checkpoints but keeps the `.lstmf`
(charset-independent), so you don't pay the long regeneration again. The
published `model/cu.traineddata` is left alone until a replacement has been
evaluated. For a fine-tune, don't reset in place at all: start a new
`MODEL_NAME` (rule 1 above).

If you relabel ground truth (e.g. a divider character), re-copy the changed
files into `data/cu-ground-truth/` so their timestamps are newer than the
`.box`/`.lstmf` derived from them; make then regenerates just those.

## Sanity checks

```bash
head -1 training/cu/unicharset                 # class count: ~200-300 = clean CU
combine_tessdata -u model/cu.traineddata /tmp/x. && grep -nP '^[A-Za-zəә]\t' /tmp/x.lstm-unicharset
```
