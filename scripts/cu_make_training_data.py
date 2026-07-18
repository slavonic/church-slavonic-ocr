#!/usr/bin/env python3
"""
cu_make_training_data.py

Turn the slavonic/cu-md-sandbox Markdown corpus into Tesseract LSTM
training pairs, rendered across SEVERAL Church Slavonic typefaces so the
model learns the text, not one face.

For every (line, font) it writes a matched pair:

    <out>/<id>_<fonttag>.gt.txt   one line of normalized CU text
    <out>/<id>_<fonttag>.png      that exact line in that face

Rendering goes through Pillow's RAQM layout engine (HarfBuzz), so the
font's GPOS mark positioning is honoured: titlo, pokrytie, stacked
breathing+accent, superscript letters. tesseract's own text2image is
avoided because its older shaping path mispositions stacked Cyrillic marks.

Two safety nets, because mixing heterogeneous CU faces is where silent
data poisoning creeps in:
  * font resolution warns if fontconfig substitutes a fallback face
    (asking for an uninstalled family yields tofu, not an error);
  * each line is checked against each font's actual cmap coverage and
    skipped for any face that lacks a needed glyph (needs fonttools).

Requires:
    pip install pillow numpy fonttools
    python3 -c "from PIL import features; print(features.check('raqm'))"  # True
"""

import argparse
import random
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------- cleaning ---

RE_LINK  = re.compile(r"!?\[[^\]]*\]\([^)]*\)")  # [text](url) / [!image](path): hex
                                                 # filenames, png/media, Рисунок NN -> junk
RE_URL   = re.compile(r"https?://[^\s;,)\]]+")   # bare URLs, keep trailing punctuation
RE_META  = re.compile(r"\{\{.*?\}\}", re.S)      # {{liturgical=... tone=...}}
RE_FOLIO = re.compile(r"<<.*?>>", re.S)          # <<73>>  folio markers
RE_NOTE  = re.compile(r"\[\[.*?\]\]", re.S)      # [[Ѱ. є҃]] citations / notes
RE_HEAD  = re.compile(r"^#+\s*", re.M)           # markdown headings, if any
RE_WS    = re.compile(r"[ \t]+")
# Inline markup delimiters/markers, none of them CU text (verified against the corpus):
#   ~ block start   = rubric   * ` emphasis   _ italic   + variable/emphasis (patriarch
#   names, "Зри")   ^ decorated-initial marker   \ markdown escape before [ ]
# All are deleted while their surrounding text is kept. NB the corpus's underscores are
# italic markers, not the scans' hyphenation char, so they are stripped here; '_' is
# still allow-listed below for the scan/fine-tune domain.
MARKUP_CHARS = str.maketrans("", "", "~=*`_+^\\")


def clean(raw: str) -> str:
    raw = RE_LINK.sub(" ", raw)     # BEFORE [[..]] so single-bracket links go first
    raw = RE_URL.sub(" ", raw)
    raw = RE_META.sub(" ", raw)
    raw = RE_FOLIO.sub(" ", raw)
    raw = RE_NOTE.sub(" ", raw)
    raw = RE_HEAD.sub("", raw)
    return raw.translate(MARKUP_CHARS)


def build_allowed(no_digits: bool = False, extra: str = "") -> set:
    """The character allow-set: real CU letters, marks, punctuation kept in print,
    the liturgical symbols, and Arabic digits. Anything outside triggers the filter."""
    a = set(" \n")
    for cp in range(0x0400, 0x0500):                     # Cyrillic block ...
        if cp not in (0x0401, 0x0451, 0x04D8, 0x04D9):   # ... minus Ё ё and schwa Ә ә
            a.add(chr(cp))
    for lo, hi in [(0x0500, 0x0530),    # Cyrillic Supplement
                   (0x2DE0, 0x2E00),    # Combining Cyrillic letters (superscripts)
                   (0xA640, 0xA6A0),    # Cyrillic Extended-B (kavyka U+A673, etc.)
                   (0x1C80, 0x1C90),    # Cyrillic Extended-C (narrow o, etc.)
                   (0x0300, 0x0370),    # combining accents / breathing
                   (0x1F540, 0x1F548)]: # Slavonic liturgical symbols (crosses, marks-chapter)
        a |= {chr(c) for c in range(lo, hi)}
    a |= set(".,:;!?()[]«»-·…")   # punctuation + brackets that appear in the printed books
    a.add("_")                     # hyphenation mark in ground truth (see --hyphenate)
    if not no_digits:
        a |= set("0123456789")     # CU normally uses letter-numerals, but the books here do
    a |= set(extra)
    return a


def paragraphs(text: str):
    for block in re.split(r"\n\s*\n", text):
        flat = RE_WS.sub(" ", block.replace("\n", " ")).strip()
        if flat:
            yield flat


# Church Slavonic vowels (base letters) -- to bias hyphenation toward syllable breaks.
VOWELS = set("аеѣиіїоᲂѡꙋуюыэєѧѫѩѭѵꙗ")


def grapheme_starts(w: str):
    """Char indices that begin a grapheme cluster (a base letter + its combining
    marks). Splitting only at these keeps accents/titla attached to their letter."""
    idx = [i for i, ch in enumerate(w) if unicodedata.combining(ch) == 0]
    idx.append(len(w))
    return idx


def split_word(w: str, avail: int, min_frag: int = 2):
    """Split w for a line-end hyphen: head (<= avail chars, room already reserved for
    the hyphen) + tail, each >= min_frag clusters, split at a grapheme boundary and
    preferring a head that ends on a vowel. Returns (head, tail) or None."""
    starts = grapheme_starts(w)
    ncl = len(starts) - 1
    if ncl < 2 * min_frag:
        return None
    cand = []
    for k in range(min_frag, ncl - min_frag + 1):   # head = clusters [0, k)
        end = starts[k]
        if end <= avail:
            head = w[:end]
            last_base = next((c for c in reversed(head)
                              if unicodedata.combining(c) == 0), "")
            cand.append((end, head, w[end:], last_base in VOWELS))
    if not cand:
        return None
    vowel = [c for c in cand if c[3]]
    end, head, tail, _ = max(vowel or cand, key=lambda c: c[0])
    return head, tail


def wrap(s: str, width: int, hyph_rate: float = 0.0, rng=None):
    words = s.split(" ")
    line, n, i = [], 0, 0
    while i < len(words):
        w = words[i]
        add = len(w) + (1 if line else 0)
        if line and n + add > width:
            avail = width - n - 1 - 1          # minus joining space, minus the hyphen
            if hyph_rate and rng and rng.random() < hyph_rate and avail >= 2:
                sp = split_word(w, avail)
                if sp:
                    head, tail = sp
                    line.append(head + "_")     # '_' = hyphenation mark in ground truth
                    yield " ".join(line)
                    line, n = [], 0
                    words[i] = tail             # continue the word on the next line
                    continue
            yield " ".join(line)
            line, n = [w], len(w)
        else:
            line.append(w)
            n += add
        i += 1
    if line:
        yield " ".join(line)


def samples(corpus: Path, width: int, min_chars: int, dedupe: bool,
            allowed: set, filter_mode: str, drops, hyph_rate: float = 0.0, rng=None):
    seen = set()
    for md in sorted(corpus.rglob("*.md")):
        rel = md.relative_to(corpus)
        # skip .venv, .git and any other hidden directory: not CU content
        if any(part.startswith(".") for part in rel.parts):
            continue
        if md.name in {"README.md", "markup-tags.md"}:
            continue
        book = rel.parts[0]
        for para in paragraphs(clean(md.read_text(encoding="utf-8"))):
            for line in wrap(para, width, hyph_rate, rng):
                line = unicodedata.normalize("NFC", line).strip()
                if len(line) < min_chars:
                    continue
                if filter_mode != "off":
                    bad = {ch for ch in line if ch not in allowed}
                    if bad:
                        drops["(lines)"] += 1
                        drops.update(bad)
                        if filter_mode == "drop":
                            continue
                        # strip: remove offending chars, re-tidy, re-check length
                        line = "".join(ch for ch in line if ch in allowed)
                        line = RE_WS.sub(" ", line).strip()
                        if len(line) < min_chars:
                            continue
                if dedupe:
                    if line in seen:
                        continue
                    seen.add(line)
                yield book, line


# ------------------------------------------------------------------ fonts ---

def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", s) or "font"


def resolve_font(spec: str):
    """Return (path, tag). Accepts a file path or a fontconfig family name."""
    p = Path(spec)
    if p.exists():
        return str(p), sanitize(p.stem)
    if shutil.which("fc-match"):
        try:
            file = subprocess.check_output(
                ["fc-match", "-f", "%{file}", spec], text=True).strip()
            fam = subprocess.check_output(
                ["fc-match", "-f", "%{family}", spec], text=True).strip()
            if file:
                token = spec.split()[0].lower()
                if token and token not in fam.lower():
                    print(f"  WARNING: '{spec}' resolved to '{fam}'\n"
                          f"           ({file})\n"
                          f"           fontconfig may have substituted a fallback "
                          f"face -> tofu. Pass the .ttf/.otf path directly to be sure.",
                          file=sys.stderr)
                return file, sanitize(spec)
        except subprocess.CalledProcessError:
            pass
    sys.exit(f"ERROR: cannot resolve font '{spec}'. Install it or pass a .ttf/.otf path.")


def coverage(path: str):
    """Set of codepoints the font's cmap covers, or None if fonttools missing."""
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        return None
    f = TTFont(path, fontNumber=0, lazy=True)
    cps = set()
    if "cmap" in f:
        for t in f["cmap"].tables:
            cps.update(t.cmap.keys())
    f.close()
    return cps


def make_renderer(font_path: str, size: int, pad: int):
    from PIL import Image, ImageDraw, ImageFont
    if not ImageFont.core.HAVE_RAQM:
        sys.exit("ERROR: this Pillow lacks RAQM/HarfBuzz shaping; marks would be "
                 "mispositioned. Fix with:  pip install --force-reinstall pillow")
    font = ImageFont.truetype(font_path, size, layout_engine=ImageFont.Layout.RAQM)
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))

    def render(text: str, out: Path):
        box = probe.textbbox((0, 0), text, font=font)
        im = Image.new("L", (box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad), 255)
        ImageDraw.Draw(im).text((pad - box[0], pad - box[1]), text, font=font, fill=0)
        im.save(str(out))

    return render


def degrade(path: Path):
    import random
    from PIL import Image, ImageFilter
    import numpy as np
    im = Image.open(path).convert("L")
    im = im.rotate(random.uniform(-1.5, 1.5), expand=True, fillcolor=255,
                   resample=Image.BICUBIC)
    im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 0.8)))
    arr = np.asarray(im).astype("float32") + np.random.normal(
        0, random.uniform(4, 10), (im.size[1], im.size[0]))
    Image.fromarray(arr.clip(0, 255).astype("uint8")).save(path)


# -------------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path,
                    help="output dir (tesstrain's <MODEL>-ground-truth)")
    ap.add_argument("--fonts", required=True, nargs="+",
                    help="one or more CU faces, each a .ttf/.otf path or a "
                         "fontconfig family name (e.g. Ponomar Triodion Pochaevsk)")
    ap.add_argument("--size", type=int, default=40)
    ap.add_argument("--pad", type=int, default=12)
    ap.add_argument("--width", type=int, default=60, help="wrap width in characters")
    ap.add_argument("--min-chars", type=int, default=8)
    ap.add_argument("--dedupe", action="store_true",
                    help="drop duplicate lines (the corpus repeats formulae a lot)")
    ap.add_argument("--charset-filter", choices=["drop", "strip", "off"], default="drop",
                    help="chars outside the CU allow-set: drop the whole line (default), "
                         "strip just those chars, or off")
    ap.add_argument("--no-digits", action="store_true",
                    help="exclude Arabic 0-9 from the allow-set (kept by default)")
    ap.add_argument("--allow-extra", default="",
                    help="extra characters to add to the allow-set, e.g. '№§'")
    ap.add_argument("--hyphenate", type=float, default=0.0, metavar="RATE",
                    help="fraction of line-breaks to hyphenate (e.g. 0.2), teaching the "
                         "model line-end word splits. 0 = off. GT gets '_'; the image "
                         "shows --hyphen-glyph")
    ap.add_argument("--hyphen-glyph", default="_",
                    help="glyph rendered for a hyphenation break (GT stays '_'). Set to "
                         "whatever your books actually print, e.g. '-' as in the books "
                         "printed by the Commission under Metropolitan Sergius")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed for reproducible hyphenation/degradation")
    ap.add_argument("--rotate-fonts", action="store_true",
                    help="render each line in ONE face (cycled) instead of every "
                         "face; caps volume when you have many distinct lines")
    ap.add_argument("--degrade", action="store_true",
                    help="light degradation (needs pillow+numpy); enable after a "
                         "clean run trains end to end")
    ap.add_argument("--no-coverage-check", action="store_true",
                    help="skip per-font glyph-coverage filtering (not recommended)")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N distinct lines (smoke test). Total pairs ~= "
                         "N x fonts, unless --rotate-fonts")
    ap.add_argument("--no-render", action="store_true",
                    help="write only .gt.txt files")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    allowed = build_allowed(args.no_digits, args.allow_extra)
    drops = Counter()
    rng = random.Random(args.seed)

    # resolve every face once: (tag, path, render_fn, covered_codepoints)
    fonts, used_tags = [], Counter()
    for spec in args.fonts:
        path, tag = resolve_font(spec)
        used_tags[tag] += 1
        if used_tags[tag] > 1:
            tag = f"{tag}{used_tags[tag]}"
        render = None if args.no_render else make_renderer(path, args.size, args.pad)
        cov = None if args.no_coverage_check else coverage(path)
        if cov is None and not args.no_coverage_check and not args.no_render:
            print("  NOTE: fonttools not installed -> no coverage filtering. "
                  "pip install fonttools", file=sys.stderr)
        fonts.append((tag, path, render, cov))
        print(f"  font: {tag:20s} {path}", file=sys.stderr)

    mode = "one cycled face per line" if args.rotate_fonts else "every face per line"
    print(f"  mode: {mode} ({len(fonts)} faces)", file=sys.stderr)

    n_lines, pairs, skipped = 0, 0, Counter()
    for book, gt_line in samples(args.corpus, args.width, args.min_chars, args.dedupe,
                                 allowed, args.charset_filter, drops,
                                 args.hyphenate, rng):
        n_lines += 1
        # ground truth keeps '_'; the rendered image shows the real hyphen glyph
        render_line = gt_line.replace("_", args.hyphen_glyph) if args.hyphenate else gt_line
        cps = set(map(ord, render_line))

        if args.rotate_fonts:
            # pick the first face (in rotated order) that covers the line
            order = [fonts[(n_lines - 1 + i) % len(fonts)] for i in range(len(fonts))]
            chosen = next((f for f in order if f[3] is None or cps <= f[3]), None)
            targets = [chosen] if chosen else []
            if not chosen:
                skipped["(no face covers line)"] += 1
        else:
            targets = []
            for f in fonts:
                if f[3] is None or cps <= f[3]:
                    targets.append(f)
                else:
                    skipped[f[0]] += 1

        for tag, path, render, cov in targets:
            stem = f"cu_{book}_{n_lines:06d}_{tag}"
            (args.out / f"{stem}.gt.txt").write_text(gt_line, encoding="utf-8")
            if render:
                png = args.out / f"{stem}.png"
                render(render_line, png)
                if args.degrade:
                    degrade(png)
            pairs += 1

        if n_lines % 2000 == 0:
            print(f"  {n_lines} lines -> {pairs} pairs...", file=sys.stderr)
        if args.limit and n_lines >= args.limit:
            break

    print(f"\nDone: {n_lines} distinct lines -> {pairs} pairs in {args.out}",
          file=sys.stderr)
    nl = drops.pop("(lines)", 0)
    if nl:
        print(f"  charset filter ({args.charset_filter}): {nl} lines affected; "
              f"top offending chars:", file=sys.stderr)
        for ch, c in drops.most_common(12):
            try:
                name = unicodedata.name(ch)
            except ValueError:
                name = "?"
            print(f"    U+{ord(ch):04X} {name:<34} in {c} lines", file=sys.stderr)
    if skipped:
        print("  skipped (missing glyphs in that face):", file=sys.stderr)
        for tag, c in skipped.most_common():
            print(f"    {tag:24s} {c}", file=sys.stderr)


if __name__ == "__main__":
    main()
