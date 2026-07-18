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
pre-filled `.gt.txt`), pointed at `real-lines/staging/` — a gitignored scratch
area, not `eval/`/`finetune/` directly. Then use `scripts/review_staging.py`
(`make review-staging`) to correct each `.gt.txt` against its crop and file the
pair straight into `eval/` or `finetune/` — it moves the `.png`+`.gt.txt` pair
together in one action, so they can't end up separated or copied into the
wrong place the way a manual `cp`/`mv` can. See `docs/evaluation.md` for the
full workflow. Keep the eval and finetune sets strictly disjoint.
