# Manual: from nothing to a published model

This manual walks you through the whole job, one step at a time: making
practice data, training a model, teaching it to read real books, checking how
well it reads, and publishing it.

You don't need to know how OCR or machine learning works. Do the steps in
order. Each step tells you:

- **What it does**, in a sentence or two.
- **What to type.** Copy each command exactly.
- **What you should see**, so you know it worked.
- **Where to look if it didn't**, with a link.

## Contents

- [Before you start](#before-you-start)
- [Step 1: Set up your computer](#step-1-set-up-your-computer) (once)
- [Step 2: Make the practice data](#step-2-make-the-practice-data)
- [Step 3: Train the base model](#step-3-train-the-base-model)
- [Step 4: Collect real lines from scanned books](#step-4-collect-real-lines-from-scanned-books)
- [Step 5: Measure how well the model reads](#step-5-measure-how-well-the-model-reads)
- [Step 6: Fine-tune on the real lines](#step-6-fine-tune-on-the-real-lines)
- [Step 7: Package the new model and keep it only if it is better](#step-7-package-the-new-model-and-keep-it-only-if-it-is-better)
- [Step 8: Fill in the model card](#step-8-fill-in-the-model-card)
- [Step 9: Publish the release](#step-9-publish-the-release)
- [When something goes wrong](#when-something-goes-wrong)
- [Reference](#reference)

---

## Before you start

### Which steps do you need?

| You want to… | Do |
|---|---|
| Just read Church Slavonic text with the finished model | Not this manual: see [Just want to OCR?](../README.md#just-want-to-ocr) |
| Build everything for the first time | Steps 1 to 9 |
| Improve an existing model with more corrected lines | Steps 4 to 9 (skip 2 and 3) |
| Publish a model you have already checked | Steps 8 and 9 |

### The big picture

```
 corpus/ (text) + fonts/                         scanned books (PDF / DJVU)
        │                                                  │
        ▼  Step 2: make dataset                            ▼  Step 4: extract_lines.py
 data/cu-ground-truth/                           data/real-lines/staging/
 (practice lines, made by computer)                        │
        │                                                  ▼  Step 4: make review-staging
        ▼  Step 3: make train-seeded             (you correct each line by hand)
 model/cu.traineddata  (base model)                        │
        │                                     ┌────────────┴─────────────┐
        │                                     ▼                          ▼
        │                         data/real-lines/finetune/   data/real-lines/eval/
        │                         (lines it learns from)      (lines it is tested on)
        │                                     │                          │
        └──────────▶ Step 6: fine-tune ◀──────┘                          │
                            │                                            │
                            ▼  Step 7: package                           │
                     training/cu_vN.traineddata ──▶ Step 5/7: cu_eval ◀──┘
                            │                         (error rate)
                            ▼  only if better
                     model/cu.traineddata ──▶ Step 8: model card ──▶ Step 9: release
```

### Words used in this manual

| Word | Meaning |
|---|---|
| **line image** | A picture of one line of printed text, saved as a `.png` file. |
| **ground truth** | The correct text for a line image, typed into a `.gt.txt` file with the same name. `foo.png` goes with `foo.gt.txt`. |
| **practice data** (synthetic data) | Line images the computer draws itself from a text collection, using Church Slavonic fonts. Very many lines, and the text is always right. |
| **real lines** | Line images cut from real scanned books, whose text a person has checked and corrected. Few lines, but they look like the books you actually want to read. |
| **model** | The file that does the reading: `cu.traineddata`. Tesseract (the OCR program) uses it. |
| **training** | The computer looking at line images and their ground truth many times over until it learns to read them. |
| **iteration** | One small learning step during training. A run is tens of thousands of them. |
| **fine-tuning** | A short extra round of training that starts from an existing model and teaches it the real lines. |
| **CER** | Character error rate: out of every 100 letters, how many the model got wrong. Lower is better. 2% means 2 wrong letters in 100. |
| **eval set** | Real lines kept aside **only for testing**. The model never learns from them, so the test is fair. |
| **finetune set** | Real lines the model **does** learn from in Step 6. |

### How to type the commands

- Open a terminal and go to the project folder. Every command in this manual is
  typed from there:
  ```bash
  cd church-slavonic-ocr
  ```
- Each time you open a new terminal, switch on the project's Python
  environment first (you create it in Step 1):
  ```bash
  source .venv/bin/activate
  ```
- In the commands, a line starting with `#` is a note for you. You don't need to
  type it.
- A `\` at the end of a line means the command continues on the next line.
  Copy all the lines together.
- Where a command contains something like `cu_v3`, the text explains what to put
  there.

---

## Step 1: Set up your computer

**What it does:** installs the programs the rest of the manual needs. You only
do this once per computer.

These instructions are for Ubuntu or Debian Linux.

**1a. Get this project, with its fonts and texts:**

```bash
git clone --recurse-submodules https://github.com/slavonic/church-slavonic-ocr.git
cd church-slavonic-ocr
python3 -m venv .venv
source .venv/bin/activate
make setup
```

**1b. Install Tesseract and its training tools.** Tesseract itself comes from the
package manager. Its training tools usually have to be built from source, which
takes a while:

```bash
sudo apt install tesseract-ocr
which lstmtraining || {
  sudo apt install libtool pkg-config libleptonica-dev libicu-dev \
                   libpango1.0-dev libcairo2-dev g++ automake
  git clone https://github.com/tesseract-ocr/tesseract ../tesseract
  cd ../tesseract
  ./autogen.sh && ./configure && make -j"$(nproc)"
  sudo make install && make training && sudo make training-install && sudo ldconfig
  cd -
}
```

**1c. Get tesstrain** (the training helper) and put it **next to** this project
folder, not inside it:

```bash
git clone https://github.com/tesseract-ocr/tesstrain ../tesstrain
```

If you put it somewhere else, add `TESSTRAIN=/path/to/tesstrain` to
`make train-seeded` in Step 3, and use that path instead of `../tesstrain` in
Step 6.

**1d. Optional: DJVU support.** Only if your scanned books are `.djvu` files:

```bash
sudo apt install djvulibre-bin
```

**You should see:** each of these commands prints something and none says
`not found`:

```bash
tesseract --version        # first line: tesseract 5.x
which lstmtraining         # a path, e.g. /usr/local/bin/lstmtraining
ls ../tesstrain/Makefile   # ../tesstrain/Makefile
ls corpus fonts            # folders with files in them
```

**If it didn't work:** an empty `corpus` or `fonts` folder means the submodules
were not fetched; run `make setup` again. More detail on the tools:
[training.md, Prerequisites](training.md#prerequisites).

---

## Step 2: Make the practice data

**What it does:** takes the Church Slavonic text collection in `corpus/`, draws
each line in five different fonts, and saves every drawing next to its correct
text in `data/cu-ground-truth/`. Some lines are made slightly blurry or tilted,
and some end with a hyphen, so they look more like real scans.

```bash
make dataset
```

By default this uses 40,000 different text lines, drawn in 5 fonts, so about
200,000 line images. To make fewer (faster, but the model learns less), set
`LIMIT`:

```bash
make dataset LIMIT=10000
```

**You should see:** at the end, a short report of how many lines were thrown
out because they contained characters that don't belong in Church Slavonic,
and which characters those were. Read the list: if a real Church Slavonic
letter appears there, it is being thrown out by mistake (see
[data-generation.md, Cleaning and the character set](data-generation.md#cleaning-and-the-character-set)).

**Look at a few samples** to make sure the drawings look right:

```bash
make review
```

This writes `model/eval/review.png`, a sheet of 8 sample lines with their text
printed in the terminal. Open it in an image viewer. The letters, accents and
titla should look like a printed book.

> **Don't open `data/cu-ground-truth/` in a file manager.** It holds hundreds of
> thousands of files and the file manager will freeze. Use `make review` instead.

**Want to change how the practice data is made?** See
[data-generation.md](data-generation.md).

---

## Step 3: Train the base model

**What it does:** teaches a new model to read, using the practice data from
Step 2. This is the longest step. Expect it to run for hours, so start it
when you can leave the computer alone.

**3a. Save the model you have now, if any.** This step overwrites
`model/cu.traineddata`. If that file exists and you might want it back:

```bash
cp model/cu.traineddata training/cu_old.traineddata
```

**3b. Start training:**

```bash
make train-seeded
```

**3c. Watch it learn** (optional, in a second terminal):

```bash
source .venv/bin/activate
python3 scripts/watch_training.py training/training.log
```

A window shows a graph of the error rate. It should go **down** over time.

**You should see:**

- Within the first 1,000 to 2,000 iterations, the error rate (`BCER`) starts
  falling. It is allowed to be high at first.
- If it stays stuck near 100%, the program notices by itself, stops, and tries
  again with different settings. You don't need to do anything.
- At the end, the new model is at `model/cu.traineddata`.

**3d. Freeze a copy as the starting point for fine-tuning.** Step 6 needs a copy
of this base model under a name that never changes:

```bash
cp model/cu.traineddata training/cu_seed.traineddata
```

If you ever repeat Step 3 later, freeze the new one under a new name
(`cu_seed2`, `cu_seed3`, …) so the old one stays intact.

**Quick health check:**

```bash
head -1 training/cu/unicharset
```

This prints the number of different characters the model knows. Around 200 to
300 is right. Thousands means something went wrong; see
[Foreign glyphs in the output](troubleshooting.md#foreign-glyphs-in-the-output-schwa-latin-).

**If it didn't work:** see [Troubleshooting: training](troubleshooting.md#training).
For how this training method works and why it's used, see
[training.md](training.md#recommended-seeded-training-make-train-seeded).

---

## Step 4: Collect real lines from scanned books

**What it does:** cuts real scanned pages into single lines, has the model take
a first guess at each line's text, and then lets you correct those guesses in a
web page. The corrected lines are the most valuable data in the project: they
are what teaches the model to read real books, and what tells you how good it
is.

**4a. Name the book file.** The file name (without `.pdf` or `.djvu`) becomes
the book's tag in every line it produces, and in the
[list of sources](../data/real-lines/README.md). Use a short name with no
spaces, for example `prazdniki.pdf` or `menaion1.pdf`.

**4b. Cut the pages into lines.** Choose a few pages first, to see how it goes:

```bash
python3 scripts/extract_lines.py prazdniki.pdf --pages 12-15 \
        --out data/real-lines/staging --tessdata-dir model
```

`--pages` takes a range (`12-15`), a list (`3,5,7`), or `all`. Page numbers are
the PDF's own page numbers, starting at 1.

The lines land in `data/real-lines/staging/`, named like
`prazdniki_p0012_l003.png` (book `prazdniki`, page 12, line 3) with a matching
`.gt.txt`.

**If the lines come out wrong**, add one of these and run it again:

| Problem | Add |
|---|---|
| Page numbers, headers, a printed frame or notes in the margin end up in the lines | `--margin 12%` (cuts 12% off every side; adjust to fit) |
| Lines are tilted | `--deskew` |
| The scan is grey, stained or uneven | `--binarize` |
| Text comes out as gibberish across the whole page | `--deskew --binarize --psm 4` |

All the options are explained in
[evaluation.md, Extract real lines](evaluation.md#1-extract-real-lines--extract_linespy).

**4c. Correct each line.**

```bash
make review-staging
```

A web page opens in your browser. It shows one line at a time: the picture on
top and the model's guess below it, in a box you can edit.

1. Compare the text with the picture, letter by letter. Fix **everything**:
   every accent, breathing mark, titlo and superscript letter. A line left
   wrong teaches the model a mistake, or makes the test look better than it is.
2. Type a dash between syllables, or at the end of a line, as `_`
   (underscore). See [the ground-truth conventions](../data/README.md#ground-truth-conventions).
3. Then press one button:

| Button | Shortcut | What it does |
|---|---|---|
| **→ finetune** | Ctrl+F | Save, and put the line in the set the model learns from. Use this for most lines. |
| **→ eval** | Ctrl+E | Save, and put the line in the test set. Use this for about **one line in ten**. |
| **Delete pair (noise)** | Ctrl+Delete | The picture isn't a proper line of text (a smudge, half a line, a picture). Throws it away. |
| **Save & next** | Ctrl+Enter | Save, but leave it in staging to decide later. |
| **Skip** | | Move on without saving. |

When the page says *Nothing left to review*, go back to the terminal and press
**Ctrl+C**. The [list of sources](../data/real-lines/README.md) then updates its
line counts by itself.

**4d. If this is a new book,** open
[data/real-lines/README.md](../data/real-lines/README.md), find the row that
says `[TODO: add citation]`, and replace that text with the book's title,
place and year.

**4e. Save your work in git.** Corrected lines can't be regenerated, so commit
them straight away:

```bash
git add data/real-lines
git commit -m "Add corrected lines from prazdniki"
```

> **Never move a line from `eval/` to `finetune/` or back.** If the model learns
> from a test line, the test stops being fair.

---

## Step 5: Measure how well the model reads

**What it does:** has the model read every line in the test set
(`data/real-lines/eval/`) and compares what it read with your corrected text.

```bash
make eval
```

**You should see** something like this (your numbers will differ):

```
  lines scored     : 225
  reference chars  : 9842
  ── micro-averaged ──
  CER (full)       :   3.10%
  CER (no marks)   :   1.20%   <- letterform-only error
  WER (full)       :  14.50%
  diacritic share  :   1.90 pts of CER live in the marks
```

followed by the same numbers for each book.

What the numbers mean:

- **CER (full)** is the main score: wrong characters per 100, counting accents
  and other marks. **2% or less** is the goal; 1% or less is excellent.
- **CER (no marks)** is the same with all accents and marks ignored. If this is
  much lower than the full CER, the model reads the letters well and mostly gets
  the marks wrong.
- **WER** is the share of whole words with at least one mistake. It is always
  higher than CER.
- **The per-book list** shows which books it reads worst. Those books need more
  corrected lines.

Spaces before or after `_` don't count as mistakes, so `foo _bar` and `foo_bar`
score the same.

**Write down the CER (full) number.** You'll compare against it in Step 7.

**See the mistakes:** open `model/eval/report.html` in a web browser. It shows
the worst lines, each with its picture, your text, the model's text, and the
differences marked.

**If you see `NOTE: N line(s) had OCR == reference exactly`:** some lines may
never have been corrected (the model's first guess was saved as-is). Check those
lines, or the score will look better than it really is.

More on reading the results: [evaluation.md](evaluation.md#2-score--cu_evalpy).

---

## Step 6: Fine-tune on the real lines

**What it does:** gives the model a short extra round of training on the real
lines from `data/real-lines/finetune/`, so it gets used to how real books look.
This is usually the step that improves the score the most.

**6a. Choose two names.**

- **The starting model:** the frozen copy from Step 3, `cu_seed`. (You can also
  start from an earlier fine-tuned model such as `cu_v2`, if it's in
  `training/` as `cu_v2.traineddata`.)
- **A new name for this run:** `cu_v` plus the next unused number. See which
  names are already taken:
  ```bash
  ls training/
  ```
  If you see `cu_v1` and `cu_v2`, use `cu_v3`.

> **Always use a new name.** Never reuse a name, and never make the new name
> the same as the starting model's. Reusing a name mixes leftovers from an old
> run into the new one, and nothing warns you.
> ([Why](training.md#rules-that-keep-a-fine-tune-from-breaking))

**6b. Add the real lines to the training data.** Copy only `finetune/`, never
`eval/`:

```bash
cp data/real-lines/finetune/*.png data/real-lines/finetune/*.gt.txt data/cu-ground-truth/
```

**6c. Start the fine-tune.** Replace `cu_v3` and `cu_seed` with your two names:

```bash
make -C ../tesstrain training MODEL_NAME=cu_v3 START_MODEL=cu_seed \
  DATA_DIR=$PWD/training GROUND_TRUTH_DIR=$PWD/data/cu-ground-truth \
  TESSDATA=$PWD/training \
  MAX_ITERATIONS=10000 2>&1 | tee finetune.log
```

Use `MAX_ITERATIONS=10000` when starting from `cu_seed` (a model that has never
seen real lines), and about `5000` when starting from a model that has.

**6d. Watch it** (optional, in a second terminal):

```bash
source .venv/bin/activate
python3 scripts/watch_training.py finetune.log
```

**6e. Check that it really trained.** This is important: the fine-tune can
fail without printing any error.

```bash
grep -c "At iteration" finetune.log
```

**You should see** a number greater than 0. If you see `0`, the model did not
train at all, even if no error was shown. Go to
[Fine-tune log stops after "Continuing from"](troubleshooting.md#fine-tune-log-stops-after-continuing-from--make-reports-no-error).

**If it stopped with an error,** see
[Troubleshooting: fine-tuning](troubleshooting.md#fine-tuning).

---

## Step 7: Package the new model and keep it only if it is better

**What it does:** turns the fine-tune's progress files into a model file you can
use, tests it, and keeps it only if it reads better than the model you have now.

In every command below, replace `cu_v3` with your run's name from Step 6.

**7a. Check the progress files are new.**

```bash
ls -l --time-style=full-iso training/cu_v3/cu_v3.traineddata \
                            training/cu_v3/checkpoints/cu_v3_checkpoint
```

**You should see** both files. The time on the `checkpoint` line must be
**later** than the time on the `traineddata` line. If it's earlier, the run
didn't train; go back to Step 6e.

**7b. Package the model:**

```bash
lstmtraining --stop_training \
  --continue_from training/cu_v3/checkpoints/cu_v3_checkpoint \
  --traineddata training/cu_v3/cu_v3.traineddata \
  --model_output training/cu_v3.traineddata
```

**You should see** a new file, `training/cu_v3.traineddata`. If it fails with
`Code range changed` or `Must supply the old traineddata`, see
[that error](troubleshooting.md#code-range-changed-from-n-to-m--must-supply-the-old-traineddata-for-code-conversion).

**7c. Test the new model** on the test set, the same way as Step 5:

```bash
python3 scripts/cu_eval.py data/real-lines/eval --model cu_v3 --tessdata-dir training \
  --reocr --report training/cu_v3_report.html --tsv training/cu_v3_metrics.tsv
```

`--reocr` makes sure the new model reads every line again, instead of reusing
an older model's results.

**7d. Compare** its `CER (full)` with the number you wrote down in Step 5.

- **Lower (better):** make it the main model, then refresh the report:
  ```bash
  cp training/cu_v3.traineddata model/cu.traineddata
  make eval
  ```
- **Not lower:** keep the model you have. Ideas for next time: correct more lines
  for the books that score worst, check `training/cu_v3_report.html` for a
  repeated mistake and look it up in [Troubleshooting](troubleshooting.md), or
  try a different number of iterations.

**Optional: try an earlier checkpoint.** A fine-tune doesn't always end on its
best moment. The run also saves numbered checkpoints:

```bash
ls training/cu_v3/checkpoints/
# cu_v3_4.223_408_2100.checkpoint  ← the first number is its error rate on the training data
```

Package one by putting its name after `--continue_from` in 7b (with a different
`--model_output`, such as `training/cu_v3b.traineddata`), then test it as in 7c.

> **Never package over the starting model** (`training/cu_seed.traineddata`).
> Always package to a new file name.

---

## Step 8: Fill in the model card

**What it does:** updates [model/MODEL_CARD.md](../model/MODEL_CARD.md), the
page that tells people what the model is, what it was trained on, and how well
it reads. Do this every time you publish.

Open `model/MODEL_CARD.md` and replace every part in `[square brackets]`. Here's
where to find each value:

| Field | Where to get it |
|---|---|
| Engine version `v[5.x]` | `tesseract --version \| head -1` |
| Version / date | The version you'll publish in Step 9 (e.g. `v1.2`) and today's date |
| Training regime | For example: `fine-tuned from cu_seed (seeded from Cyrillic)`, and the `MAX_ITERATIONS` you used in Step 6 |
| Source text commit | `git -C corpus rev-parse --short HEAD` |
| `~[N] distinct lines` | The `LIMIT` from Step 2 (40,000 by default) |
| `[5] typefaces`, rate `[0.20]` | Unchanged unless you edited [scripts/build_dataset.sh](../scripts/build_dataset.sh) |
| Real adaptation: lines and books | The **total** row and the book list in [data/real-lines/README.md](../data/real-lines/README.md) |
| Character set `[N] classes` | See the command below |
| Metrics table | The `CER (full)`, `CER (no marks)` and `WER (full)` lines from `make eval` in Step 7d |
| Known confusions | Letters the model mixes up repeatedly in `model/eval/report.html` |

The number of character classes:

```bash
combine_tessdata -u model/cu.traineddata /tmp/cu_check. && head -1 /tmp/cu_check.lstm-unicharset
```

Then save the card in git and send it to GitHub **before** publishing, so the
published version includes it:

```bash
git add model/MODEL_CARD.md data/real-lines
git commit -m "Model card for v1.2"
git push
```

---

## Step 9: Publish the release

**What it does:** uploads `model/cu.traineddata` to the project's
[GitHub Releases page](https://github.com/slavonic/church-slavonic-ocr/releases),
where anyone can download it.

**9a. Make sure the `gh` tool is logged in:**

```bash
gh auth status
```

If it says you're not logged in, run `gh auth login` and follow the questions.

**9b. Pick the version number.** See what's already published:

```bash
gh release list
```

Use the next number. Small improvement: `v1.1` → `v1.2`. Big change (such as
retraining from Step 2): `v1.2` → `v2.0`. Use the same number you wrote in the
model card.

**9c. Publish:**

```bash
make release VERSION=v1.2
```

**You should see** a web address for the new release. Open it and check that
`cu.traineddata` is listed under *Assets*.

**9d. Try the published model**, the way a user would:

```bash
gh release download v1.2 --pattern cu.traineddata --dir /tmp/cu_release
tesseract data/real-lines/eval/prazdniki_p0007_l016.png stdout --psm 13 \
          -l cu --tessdata-dir /tmp/cu_release
```

**You should see** the line's text printed in Church Slavonic.

**Published by mistake, or the wrong file?** Delete the release and its tag,
fix the problem, and repeat 9c:

```bash
gh release delete v1.2 --cleanup-tag
```

Do this only right after publishing, before anyone has downloaded it.

**You're done.** For the next round of improvements, go back to
[Step 4](#step-4-collect-real-lines-from-scanned-books).

---

## When something goes wrong

[troubleshooting.md](troubleshooting.md) is arranged by step. Find the step
you were on, then the symptom that matches:

- [Practice data (Step 2)](troubleshooting.md#practice-data)
- [Training (Step 3)](troubleshooting.md#training)
- [Collecting real lines (Step 4)](troubleshooting.md#collecting-real-lines)
- [Fine-tuning and packaging (Steps 6–7)](troubleshooting.md#fine-tuning)
- [Reading the scores (Steps 5 and 7)](troubleshooting.md#reading-the-scores)

The general rule: **change one thing at a time**, and find out what is wrong
before you rebuild anything.

## Reference

The manual tells you *what to do*. These pages explain *how it works* and list
every option:

| Page | Covers |
|---|---|
| [data-generation.md](data-generation.md) | Step 2 in detail: fonts, the allowed characters, hyphenation, blurring |
| [training.md](training.md) | Steps 3, 6 and 7 in detail: why seeded training, the fine-tune rules, resetting the character set |
| [evaluation.md](evaluation.md) | Steps 4 and 5 in detail: every `extract_lines.py` option, every number `cu_eval.py` prints |
| [troubleshooting.md](troubleshooting.md) | Problems we've actually hit, and their fixes |
| [data/README.md](../data/README.md) | What's in `data/`, and how to type the text (`_` and other conventions) |
| [data/real-lines/README.md](../data/real-lines/README.md) | Which books the real lines come from |
| [model/README.md](../model/README.md) | Installing and using the published model |
| [model/MODEL_CARD.md](../model/MODEL_CARD.md) | The model card |

All the `make` commands:

| Command | What it does | Step |
|---|---|---|
| `make setup` | Fetch the fonts and texts, install Python packages | 1 |
| `make dataset` | Make the practice data | 2 |
| `make review` | Save a sheet of sample practice lines to `model/eval/review.png` | 2 |
| `make train-seeded` | Train the base model (recommended) | 3 |
| `make train` | Train the base model the plain way ([why not recommended](training.md#plain-from-scratch-run)) | 3 |
| `make review-staging` | Open the web page for correcting lines | 4 |
| `make eval` | Test `model/cu.traineddata` on the test set | 5, 7 |
| `make release VERSION=vX.Y` | Publish `model/cu.traineddata` on GitHub | 9 |
| `make clean-train` | Delete all training work files (keeps data and model) | — |
| `make reset-charset` | Reset the character list after changing the allowed characters. **Also deletes `model/cu.traineddata`**, so save a copy first ([details](training.md#resetting-the-charset)) | — |
