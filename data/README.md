# Data

Two very different kinds of ground truth live here. Keep them straight — the
distinction drives what is committed and what is regenerated.

## `cu-ground-truth/` — synthetic (generated, **not** committed)

Line images rendered from the corpus text in the project fonts, each paired with
a `.gt.txt`. Hundreds of thousands of files. This is a **build artifact**: it is
fully reproducible from `corpus/` + `fonts/` + the generator, so it is
`.gitignore`d and regenerated on demand:

```bash
make dataset            # or: scripts/build_dataset.sh
```

`build_dataset.sh` pins the exact parameters (fonts, `--dedupe --degrade`,
`--hyphenate 0.20`, `--seed 1`, `--limit`) so the set is reproducible. This is
also the directory tesstrain reads during training.

Do not open this directory in a file manager or glob it with `*` — the file
count will overflow your shell. Use `scripts/review_samples.py` to inspect a few
samples, or `find` to list them.

## `real-lines/` — hand-corrected scan lines (**committed**)

Line crops from actual scanned books with human-verified transcriptions. These
are **not** reproducible and are precious, so they are committed. They're tiny (a
few KB each), so plain git handles them — no Git LFS needed.

- `real-lines/eval/` — the held-out evaluation set. Used only for measuring the
  model (`make eval`). **Never** train or fine-tune on these.
- `real-lines/finetune/` — corrected lines used to adapt the model to real
  scans. For a fine-tune run, copy these into `cu-ground-truth/` alongside the
  synthetic pairs so the charset picks up their glyphs.

Produce more of either with `scripts/extract_lines.py` (PDF/DJVU → crops +
pre-filled `.gt.txt`), then correct each `.gt.txt` against its crop. Keep the
eval and finetune sets strictly disjoint.

## Ground-truth conventions

**One divider token: `_`.** A dash printed between syllables or at a line end
is transcribed as `_`, whether it is a line-break hyphen or a syllable divider
in chant lyrics:

```
се_бѣ̀ пред̾_ꙋ_го_то́_ва: ᲂу҆_твер_
```

The synthetic data already uses `_` for line-end hyphenation. In the books we
transcribe, a syllable dash looks like a hyphen, so giving it a second label
(`~`, `‿`) would teach the model two answers for one image, which it cannot
resolve (see `docs/troubleshooting.md`). Nothing is lost by merging them:
position says which is which (line-final = hyphenation, mid-line = syllable
boundary), and the syllable boundary is what a later notation model aligns to.

Use a separate token only if a book prints a visibly different mark (for
instance a long extension line that is unmistakably not a hyphen) *and* you
want the model to tell them apart.

**Syllabification tool.** The standalone syllabifier (not part of this repo)
produces text in this convention with `--divider '_'` in the
[command-line tool](https://github.com/slavonic/cu-syllabifier), or the
"custom" divider in its [web interface](https://sci.ponomar.net/syllabify/).

**Relabeling older files.** If any real-line files still use `~` or `‿`:

```bash
grep -l '[~‿]' data/real-lines/*/*.gt.txt | wc -l      # how many will change
sed -i 's/[~‿]/_/g' data/real-lines/*/*.gt.txt          # commit first so you can diff
cp data/real-lines/finetune/* data/cu-ground-truth/     # refresh the copies tesstrain reads
```

The `cp` gives the files new timestamps, so make regenerates just their
`.box`/`.lstmf`. Then run a **new** fine-tune under a new `MODEL_NAME`
(`docs/training.md`): a model that has already learned `~` keeps that class
(charsets only grow across a `START_MODEL` run), so for a clean result start
from a model that never saw it. Scores on the relabeled eval set are not
strictly comparable to earlier ones, because the reference text changed.

