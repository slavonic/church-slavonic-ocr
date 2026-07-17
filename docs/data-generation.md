# Data generation — `cu_make_training_data.py`

Turns the Markdown corpus into Tesseract LSTM training pairs: for every line a
`.gt.txt` (ground truth) and a `.png` (that exact line rendered), one line per
image so text and image can never drift apart.

Pinned invocation: `scripts/build_dataset.sh` (called by `make dataset`).

## Rendering

Rendering goes through Pillow's RAQM layout engine (HarfBuzz), so the fonts'
GPOS mark positioning is honoured — titla, pokrytie, stacked breathing+accent,
and superscript letters stack correctly. Tesseract's own `text2image` is avoided
because its older shaping path mispositions stacked Cyrillic marks.

Multiple faces are rendered by default (`--fonts …`); each line is produced in
every face so the model learns text, not one typeface. Each face is checked
against its actual glyph coverage (via fonttools) and a line is skipped for any
face lacking a needed glyph, so you never train on tofu with valid-looking
ground truth. `--rotate-fonts` renders one face per line instead, to cap volume.

## Cleaning and the character set

The corpus is Markdown with editorial apparatus. The cleaner removes: `{{…}}`
metadata, `<<…>>` folios, `[[…]]` notes, markdown image/figure links and bare
URLs (the source of stray Latin/hex/`png` debris), and the inline delimiters
`~ = * ` `` ` `` `_ + ^ \` (rubric, emphasis, italic, variable-text, decorated-initial,
and escape markers) — keeping the text they wrap.

A character **allow-set** then filters anything outside real Church Slavonic:
Cyrillic blocks (minus Ё/ё and Azerbaijani schwa), combining marks, print
punctuation, `()` and `[]`, guillemets, the liturgical symbols (U+1F540–U+1F545),
and Arabic digits. `--charset-filter drop|strip|off` controls handling;
`--no-digits` and `--allow-extra` adjust the set. The run prints how many lines
were affected and the top offending characters — watch this to catch anything
legitimate being dropped.

> Why this matters: a stray character in the ground truth becomes an output
> class in the model, which then emits it on noisy input. Training from scratch
> does **not** fix contamination baked into the data — the clean charset does.

## Hyphenation injection

Real scans hyphenate words at line-ends; the corpus does not. `--hyphenate RATE`
(e.g. `0.20`) splits a fraction of line-crossing words and ends the line with a
hyphenation mark. Splits are grapheme-safe (accents/titla never orphaned) and
prefer vowel boundaries (syllable-ish).

The ground truth records the mark as `_` (the project convention); the rendered
image shows `--hyphen-glyph` (default `-`). Set the glyph to whatever your target
books actually print — if that is a distinct mark like the double-oblique `⸗`,
use a face that contains it (coverage-checked automatically). Note that rendering
line-breaks as `-` collides with compound hyphens, so the `-`/`_` distinction
becomes positional; a distinct glyph avoids that.

## Degradation

`--degrade` applies light skew, blur, and noise so the synthetic images resemble
real scans. Keep it comparable to your actual scans — degradation harsher than
reality inflates synthetic error without buying robustness (see
`docs/troubleshooting.md`). `--seed` makes hyphenation and degradation reproducible.

## Key options

| option | meaning |
|--------|---------|
| `--fonts …` | one or more `.ttf`/`.otf` paths or family names |
| `--dedupe` | drop duplicate lines (the corpus repeats formulae heavily) |
| `--degrade` | light skew/blur/noise |
| `--hyphenate RATE` | fraction of line-breaks to hyphenate (GT `_`, image `--hyphen-glyph`) |
| `--charset-filter` | `drop` (default) / `strip` / `off` for out-of-set characters |
| `--no-digits` | exclude Arabic digits (kept by default) |
| `--limit N` | cap distinct lines (total pairs ≈ N × faces) |
| `--seed N` | reproducible hyphenation/degradation |
