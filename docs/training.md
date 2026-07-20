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

## Training a model (`make train-seeded`)

If you've never trained a Tesseract LSTM model before, the short version: a
"model" here is a `.traineddata` file containing a neural network plus a
**unicharset** (the fixed list of character classes it's allowed to output).
Training feeds it your ground truth in small steps ("iterations"), each one
nudging the network's weights to reduce the gap between what it currently
outputs for a line image and that line's `.gt.txt`. tesstrain is the harness
that turns a directory of `.png`+`.gt.txt` pairs into that process; this
section explains what it actually does. For anything not covered here — the
full set of tesstrain Makefile variables, `lstmtraining` flags, general LSTM
training theory — see [tesstrain's own README](https://github.com/tesseract-ocr/tesstrain)
and the [Tesseract training wiki](https://tesseract-ocr.github.io/tessdoc/tess4/TrainingTesseract-4.00.html);
this doc only covers what's specific to *this* project's setup.

**Two naive strategies both fail here**, in opposite ways:

- `START_MODEL=Cyrillic` **merges** Cyrillic's unicharset into yours
  (`merge_unicharsets` in the tesstrain Makefile), so the model can emit
  schwa, Latin, `©` — classes that were never in your ground truth.
- Pure from-scratch keeps the charset clean but random init can fall into the
  **CTC all-blank basin**: BCER pinned ~98–100% for tens of thousands of
  iterations, low rms, near-constant short decode. Whether it escapes is a
  dice roll on the init.

`scripts/train_seeded.py` (run via `make train-seeded`) threads between them
in four stages:

1. **Box/lstmf generation, unicharset, proto model, lists** — tesstrain's
   from-scratch path (no `START_MODEL`, so no merge), run as
   `make -C $TESSTRAIN unicharset lists proto-model ...`:
   - For every `.gt.txt`+`.png` pair, tesstrain first generates a `.box` file
     (per-character bounding boxes, derived by aligning the ground-truth text
     against the image) and then a `.lstmf` file (that image plus its
     alignment, serialized into Tesseract's internal training format). This
     is one subprocess *per line* — over ~200k pairs this is genuinely slow,
     not stuck; run with `-j$(nproc)` (the Makefile default) or it's a
     multi-hour serial crawl that looks like a hang. Both are `.PRECIOUS` to
     tesstrain, so an interrupted run resumes rather than regenerating
     everything.
   - The **unicharset** is then just the distinct characters/grapheme
     clusters seen across every generated `.box` — built from *your* ground
     truth only, so it can only ever contain classes your data actually has.
     `train_seeded.py` additionally **audits** this unicharset afterward and
     refuses to continue if it finds Latin letters, schwa, or anything else
     that isn't Church Slavonic (a corpus-cleaning bug would otherwise only
     surface hours later, mid-training).
   - The **proto model** is an untrained network shaped to match that
     unicharset's output layer size — not yet seeded with any weights.
   - **Lists** (`list.train`/`list.eval`) are a train/held-out split of the
     generated `.lstmf` files, used internally by `lstmtraining` to report a
     running character error rate (BCER) as it trains. This internal split is
     entirely synthetic and separate from `data/real-lines/eval/` — it tells
     you whether training is converging at all, not whether the model works
     on real scans (see `docs/troubleshooting.md`, "Synthetic CER and real
     CER are not comparable").
2. **Cyrillic feature seed** — downloads stock `Cyrillic.traineddata` (a
   general Cyrillic LSTM model, cached in `training/`) and extracts just its
   `.lstm` component (`combine_tessdata -e`) — the trained weights, without
   its unicharset.
3. **Training, continued from that seed but remapped to your charset** —
   `lstmtraining --continue_from Cyrillic.lstm --old_traineddata
   Cyrillic.traineddata --traineddata <your proto model>` starts from
   Cyrillic's already-useful lower feature layers ("Cyrillic's eyes") while
   `lstmtraining` remaps the output layer to your clean CU-only unicharset
   ("your alphabet") — so training starts already able to see glyph shapes,
   instead of random noise, without inheriting Cyrillic's foreign classes.
   Each **iteration** here is one gradient-descent step over one `.lstmf`
   line; `--max-iterations` caps how many of those steps run.
   A **watchdog** parses `lstmtraining`'s own progress lines and kills the
   run if BCER is still ≥90% past ~6k iterations with no downtrend, then
   retries with a doubled learning rate (a collapsed attempt costs minutes,
   not a night). A slow but genuinely falling BCER is left alone.
4. **Packaging** — once training stops (iteration cap reached, or you
   interrupt it), the best checkpoint is turned into a final `.traineddata`
   via `lstmtraining --stop_training`.

```bash
make train-seeded                 # or directly:
python3 scripts/train_seeded.py --tesstrain ../tesstrain --max-iterations 100000
```

Expect BCER to break downward within the first 1–2k iterations, since the
network isn't starting from scratch. Watch the trend with
`scripts/watch_training.py`.

## The resulting model

Stage 4 above writes the packaged model to `training/cu.traineddata`; the
`make train-seeded` wrapper then copies it to `model/cu.traineddata` — the
copy this repo's other tooling expects (`make eval`, the model card, the
Releases page). From there:

```bash
make eval                          # score model/cu.traineddata (docs/evaluation.md)
make release VERSION=v1.0          # gh release create v1.0 model/cu.traineddata …
```

`make release` publishes `model/cu.traineddata` as a GitHub Release asset
(requires the `gh` CLI). Update `model/MODEL_CARD.md` in the same release so
its metrics and provenance match the published binary — see `model/README.md`
for the full publishing workflow, including why the asset is always named
exactly `cu.traineddata` while the release **tag** carries the version.

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

## Cleaning before retraining

Stale `training/` state is a classic cause of a model that garbles even its
own synthetic lines — most often a **unicharset mismatch**: you changed the
allow-set (e.g. added digits or `_`), but the old unicharset/checkpoints are
still lying around and collide with the new one. Two cleaning targets exist,
at different levels of aggression:

- **`make reset-charset`** — wraps tesstrain's own `clean-output` (`make -C
  $(TESSTRAIN) clean-output MODEL_NAME=$(MODEL_NAME)
  DATA_DIR=$(CURDIR)/training`, using this repo's Makefile variables rather
  than a `$TESSTRAIN` you'd have to export yourself) plus removing the stale
  `.traineddata` copies. It drops the unicharset/recoder/checkpoints but
  **keeps** the `.box`/`.lstmf` files (they're charset-independent), so you
  don't pay for regenerating them again.
- **`make clean-train`** — the blunter option: wipes the entire `training/`
  scratch dir (`rm -rf training/*`), including the `.box`/`.lstmf` files
  `reset-charset` preserves.

```bash
make reset-charset   # keep .box/.lstmf, just rebuild the charset
# or, more aggressively:
make clean-train      # rm -rf training/* (keeps data/cu-ground-truth and model/)

make train-seeded
```

Reach for `reset-charset` when you only need a fresh charset and want to keep
the expensive `.lstmf` generation; reach for `clean-train` when `training/` is
in some inconsistent state you don't want to reason about and you're fine
re-paying that cost. Either way, a changed allow-set **requires** one of these
before retraining — old state silently colliding with new is not something
`make train-seeded` detects on its own.

## Sanity checks

```bash
head -1 training/cu/unicharset                 # class count: ~200-300 = clean CU
combine_tessdata -u model/cu.traineddata /tmp/x. && grep -nP '^[A-Za-zəә]\t' /tmp/x.lstm-unicharset
```
