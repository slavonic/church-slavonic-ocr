# Data generation — `cu_make_training_data.py`

Turns the Markdown corpus into Tesseract LSTM training pairs: for every line a
`.gt.txt` (ground truth) and a `.png` (that exact line rendered), one line per
image so text and image can never drift apart.

Pinned invocation: `scripts/build_dataset.sh` (called by `make dataset`), which
expands to:

```bash
python3 scripts/cu_make_training_data.py \
  --corpus corpus \
  --out data/cu-ground-truth \
  --fonts fonts/Ponomar/fonts/ttf/Ponomar-Regular.ttf \
          fonts/Triodion/fonts/ttf/Triodion-Regular.ttf \
          fonts/Pochaevsk/fonts/ttf/Pochaevsk-Regular.ttf \
          fonts/Acathist/fonts/ttf/Acathist-Regular.ttf \
          fonts/Monomakh/fonts/ttf/Monomakh-Regular.ttf \
  --dedupe --degrade \
  --hyphenate 0.20 --hyphen-glyph '_' \
  --seed 1 \
  --limit "${LIMIT:-40000}"
```

## Rendering

Rendering goes through Pillow's RAQM layout engine (HarfBuzz), so the fonts'
GPOS mark positioning is honored — titla, pokrytie, stacked breathing+accent,
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
Cyrillic blocks, combining marks, print punctuation, `()` and `[]`, guillemets,
the liturgical symbols (U+1F540–U+1F545), and Arabic digits — with Ё/ё and the
Azerbaijani schwa Ә/ә explicitly removed from the Cyrillic block, since neither
belongs to Church Slavonic orthography despite sharing Unicode's Cyrillic range.
`--charset-filter drop|strip|off` controls handling;
`--no-digits` and `--allow-extra` adjust the set. The run prints how many lines
were affected and the top offending characters — watch this to catch anything
legitimate being dropped.

> Why this matters: a stray character in the ground truth becomes an output
> class in the model, which then emits it on noisy input. Training from scratch
> does **not** fix contamination baked into the data — the clean charset does.

### Allow-set examples

Say a corpus line reads `Спасѐ на́съ, Nоtа bene†` — real CU text with a stray
editorial aside (`Nоtа bene†` — Latin letters, a footnote dagger, none of it
meant for the model):

```bash
# default: --charset-filter drop — the whole line is dropped, since some of
# its characters (N, o, t, a, b, e, n, e, †) are outside the allow-set
scripts/cu_make_training_data.py --fonts Ponomar --charset-filter drop ...
# -> line skipped entirely; shows up in the drop-reason counts printed at the end

# --charset-filter strip — keep the line, remove just the offending characters
scripts/cu_make_training_data.py --fonts Ponomar --charset-filter strip ...
# -> ground truth becomes "Спасѐ на́съ, "  (Latin/dagger gone, CU kept)
# strip is useful when offending runs are rare interpolations you don't want
# to lose whole lines over, but double-check the result isn't mangled — a
# strip in the middle of a word can glue two unrelated fragments together

# --charset-filter off — keep everything, including out-of-set characters
scripts/cu_make_training_data.py --fonts Ponomar --charset-filter off ...
# -> ground truth keeps "Спасѐ на́съ, Nоtа bene†" verbatim; only use this to
# inspect what the filter would otherwise catch (e.g. piping to a report),
# not for lines that go into an actual training run
```

`--allow-extra` widens the set instead of loosening the filter mode — use it
when a character is legitimate but not covered by the defaults, e.g. the corpus
uses `№` and `§` in rubrics you want to keep:

```bash
scripts/cu_make_training_data.py --fonts Ponomar --allow-extra '№§' ...
# -> lines containing № or § are no longer dropped/stripped; every other
# out-of-set character (Latin letters, stray symbols) is still filtered
```

`--no-digits` narrows the set the other way, dropping Arabic `0-9` from the
allow-set (they're kept by default since most books print Arabic
numerals rather than Cyrillic numerals for things like page numbering):

```bash
scripts/cu_make_training_data.py --fonts Ponomar --no-digits ...
# -> a line like "глава 12" is treated as containing an out-of-set run ("12"),
# so --charset-filter drop/strip applies to the digits same as any other
# disallowed character
```

## Hyphenation injection

Real scans hyphenate words at line-ends; the corpus does not. `--hyphenate RATE`
(e.g. `0.20`) splits a fraction of line-crossing words and ends the line with a
hyphenation mark. Splits are grapheme-safe (accents/titla never orphaned) and
prefer vowel boundaries (syllable-ish).

The ground truth records the mark as `_` (the project convention); the rendered
image shows `--hyphen-glyph` (default `_`, matching the ground truth). Set the
glyph to whatever your target books actually print — e.g. `-`, as in the books
printed by the Commission under Metropolitan Sergius — using a face that
contains it (coverage-checked automatically).

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
