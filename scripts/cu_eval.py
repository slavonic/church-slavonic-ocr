#!/usr/bin/env python3
"""
cu_eval.py -- score a Church Slavonic Tesseract model against real, corrected lines.

Point it at a directory of reference `.gt.txt` files each beside a matching
line image (same stem: .png/.tif/.jpg/.bin.png). For every pair it OCRs the
image with your model and compares against the reference, reporting:

  * CER and WER, MICRO-averaged (total edits / total reference units)
  * the same computed with combining marks stripped, so you can see whether
    the errors are letterforms or just diacritic placement
  * spaces around the `_` divider are ignored on both sides before scoring
  * a per-line table sorted worst-first, plus a self-contained report.html
    with each crop, its reference, the OCR, and a character-level diff.

OCR hypotheses are cached beside each image as `<stem>.hyp.txt`, so re-runs
are instant; pass --reocr to refresh them.

Deps: tesseract on PATH with your model installed (e.g. -l cu). Pure-python
otherwise (no numpy/Pillow needed).
"""

import argparse
import base64
import html
import re
import subprocess
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

IMG_EXT = (".png", ".bin.png", ".nrm.png", ".tif", ".tiff", ".jpg", ".jpeg")


# ------------------------------------------------------------ text metrics ---

def strip_marks(s: str) -> str:
    """Drop all combining marks (titlo, accents, breathing) -> letterforms only."""
    d = unicodedata.normalize("NFD", s)
    d = "".join(ch for ch in d if not unicodedata.combining(ch))
    return unicodedata.normalize("NFC", d)


def normalize_divider(s: str) -> str:
    """Drop spaces around the `_` divider, so `foo_bar`, `foo _bar`,
    `foo_ bar` and `foo _ bar` all score the same."""
    return re.sub(r"\s*_\s*", "_", s)


def levenshtein(a, b) -> int:
    """Edit distance over two sequences (strings or token lists)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1,          # deletion
                           cur[j - 1] + 1,       # insertion
                           prev[j - 1] + (ca != cb)))  # substitution
        prev = cur
    return prev[lb]


def html_diff(ref: str, hyp: str) -> str:
    """Inline char-level diff: <del> = in ref not hyp, <ins> = in hyp not ref."""
    out = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, ref, hyp).get_opcodes():
        r, h = html.escape(ref[i1:i2]), html.escape(hyp[j1:j2])
        if tag == "equal":
            out.append(r)
        elif tag == "delete":
            out.append(f"<del>{r}</del>")
        elif tag == "insert":
            out.append(f"<ins>{h}</ins>")
        else:  # replace
            out.append(f"<del>{r}</del><ins>{h}</ins>")
    return "".join(out)


# ------------------------------------------------------------- bootstrap ---

def bootstrap_ci(pairs, n_boot=2000, seed=0, level=0.95):
    """95% percentile CI of micro-averaged error (in %) from (edits, chars)
    pairs, resampling whole lines with replacement. Lines, not characters,
    are the unit: errors cluster within a line (one bad crop -> many errors)."""
    import random
    if not pairs or n_boot <= 0:
        return None
    rng = random.Random(seed)
    k = len(pairs)
    stats = []
    for _ in range(n_boot):
        d = n = 0
        for _ in range(k):
            e, c = pairs[rng.randrange(k)]
            d += e
            n += c
        stats.append(100 * d / max(1, n))
    stats.sort()
    a = (1 - level) / 2
    return stats[int(a * n_boot)], stats[min(n_boot - 1, int((1 - a) * n_boot))]


def paired_bootstrap(new, old, n_boot=2000, seed=0, level=0.95):
    """Paired bootstrap of the CER change (new - old, in points) over the same
    lines. `new` and `old` are lists of (edits, chars) in the same line order.
    Returns (change, lo, hi, share of resamples where new is better)."""
    import random
    rng = random.Random(seed)
    k = len(new)
    def cer(idx, side):
        d = sum(side[i][0] for i in idx)
        n = sum(side[i][1] for i in idx)
        return 100 * d / max(1, n)
    full = range(k)
    change = cer(full, new) - cer(full, old)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(k) for _ in range(k)]
        diffs.append(cer(idx, new) - cer(idx, old))
    diffs.sort()
    a = (1 - level) / 2
    better = sum(d < 0 for d in diffs) / n_boot
    return change, diffs[int(a * n_boot)], diffs[min(n_boot - 1, int((1 - a) * n_boot))], better


def read_tsv_edits(path):
    """image name -> (char edits, ref chars) from an earlier --tsv file. Uses
    the char_edits column when present, else rebuilds it from cer_full."""
    out = {}
    with open(path, encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        col = {h: i for i, h in enumerate(head)}
        for line in f:
            v = line.rstrip("\n").split("\t")
            n = int(v[col["ref_chars"]])
            if "char_edits" in col:
                d = int(v[col["char_edits"]])
            else:
                d = round(float(v[col["cer_full"]]) * n / 100)
            out[v[col["image"]]] = (d, n)
    return out


# --------------------------------------------------------- error pairs ---

def glyph_units(s: str) -> list:
    """Split into base character + its combining marks (NFD), so `и҆́` is one
    unit and a wrong accent shows up as one substituted unit, not as noise."""
    out = []
    for ch in unicodedata.normalize("NFD", s):
        if unicodedata.combining(ch) and out:
            out[-1] += ch
        else:
            out.append(ch)
    return out


def align_units(a: list, b: list) -> list:
    """Levenshtein alignment of two unit lists -> [(op, ref_unit, hyp_unit)]
    for the non-matching positions only; op is 'sub', 'del' or 'ins'."""
    n, m = len(a), len(b)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1,
                          D[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    ops, i, j = [], n, m
    while i or j:
        if i and j and D[i][j] == D[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            if a[i - 1] != b[j - 1]:
                ops.append(("sub", a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i and D[i][j] == D[i - 1][j] + 1:
            ops.append(("del", a[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", b[j - 1]))
            j -= 1
    return ops


def _base(u):
    return "".join(c for c in u if not unicodedata.combining(c))


def _marks(u):
    return "".join(c for c in u if unicodedata.combining(c))


def mark_names(m: str) -> str:
    if not m:
        return "none"
    return " + ".join(unicodedata.name(c, f"U+{ord(c):04X}")
                      .replace("COMBINING ", "").replace("CYRILLIC ", "").lower()
                      for c in m)


def has_repeated_mark(s: str) -> bool:
    """True if one letter carries the same combining mark twice (e.g. a
    doubled breathing), which never happens in correct text."""
    for u in glyph_units(s):
        m = _marks(u)
        if len(m) != len(set(m)):
            return True
    return False


def error_breakdown(rows) -> dict:
    """Classify every unit-level error across all lines.

    kinds   : Counter of error type -> count
    marks   : Counter of (ref marks, hyp marks) for same-letter mark errors
    letters : Counter of (ref unit, hyp unit) for all other errors ('' = none)
    mark_letters : Counter of base letter -> mark errors on it
    examples: (kind, key) -> up to 3 image names, for tracing back
    repeated: lines whose OCR has a repeated mark on one letter (and refs that do)
    """
    from collections import Counter, defaultdict
    kinds, marks, letters, mark_letters = Counter(), Counter(), Counter(), Counter()
    examples = defaultdict(list)
    total = 0
    for r in rows:
        name = r["img"].name
        for op, a, b in align_units(glyph_units(r["ref"]), glyph_units(r["hyp"])):
            total += 1
            if op == "sub" and _base(a) == _base(b):
                kind, key = "right letter, wrong marks", (mark_names(_marks(a)),
                                                          mark_names(_marks(b)))
                marks[key] += 1
                mark_letters[_base(a)] += 1
                ex = examples[("mark", key)]
            else:
                special = {" ": "space", "_": "“_” divider"}
                u = a or b
                what = special.get(_base(u), "letter")
                kind = {"sub": "wrong letter" if what == "letter" else f"{what} replaced",
                        "del": f"{what} missing", "ins": f"extra {what}"}[op]
                key = (unicodedata.normalize("NFC", a), unicodedata.normalize("NFC", b))
                letters[key] += 1
                ex = examples[("letter", key)]
            kinds[kind] += 1
            if len(ex) < 3 and name not in ex:
                ex.append(name)
    return dict(
        total=total, kinds=kinds, marks=marks, letters=letters,
        mark_letters=mark_letters, examples=examples,
        repeated_hyp=[r["img"].name for r in rows if has_repeated_mark(r["hyp"])],
        repeated_ref=[r["img"].name for r in rows if has_repeated_mark(r["ref"])],
    )


def breakdown_html(bd: dict, nlines: int, top: int = 20, shown=()) -> str:
    """HTML section for the report. `shown` = image names that have a row in
    the report, so examples can link to them."""
    esc = html.escape
    tot = bd["total"] or 1

    def ex_links(key):
        out = []
        for n in bd["examples"].get(key, []):
            stem = esc(n.rsplit(".", 1)[0])
            out.append(f'<a href="#{stem}">{stem}</a>' if n in shown else stem)
        return ", ".join(out)

    def vis(u):
        # show the unit itself; a bare mark gets a dotted circle to sit on
        if u == "":
            return "<span class=m>∅</span>"
        if u == " ":
            return "<span class=m>␣</span>"
        if unicodedata.combining(u[0]):
            u = "◌" + u
        return esc(u)

    p = ["<h2>Error breakdown</h2>",
         f"<p class=m>{bd['total']} errors, counted per letter-with-its-marks "
         "(so a wrong accent is one error, not several). Spaces around “_” are "
         "ignored, as in the scores.</p>"]

    rep = len(bd["repeated_hyp"])
    p.append(f"<p><b>Repeated marks:</b> {rep} of {nlines} lines "
             f"({100 * rep / max(1, nlines):.0f}%) have the same mark twice on one "
             "letter in the OCR")
    if bd["repeated_ref"]:
        p.append(f"; <span class=cer>{len(bd['repeated_ref'])} reference line(s) do too, "
                 f"which is a typo: {esc(', '.join(bd['repeated_ref'][:10]))}</span>")
    p.append(".</p>")

    p.append("<h3>By kind</h3><table><tr><th>kind<th>count<th>share")
    for k, v in bd["kinds"].most_common():
        p.append(f"<tr><td>{esc(k)}<td class=n>{v}<td class=n>{100 * v / tot:.1f}%")
    p.append("</table>")

    p.append(f"<h3>Mark confusions (right letter, wrong marks), top {top}</h3>"
             "<table><tr><th>reference marks<th>OCR marks<th>count<th>examples")
    for (a, b), v in bd["marks"].most_common(top):
        p.append(f"<tr><td>{esc(a)}<td>{esc(b)}<td class=n>{v}"
                 f"<td class=m>{ex_links(('mark', (a, b)))}")
    p.append("</table>")

    p.append("<h3>Letters that most often get wrong marks</h3><p class=txt>")
    p.append(" &nbsp; ".join(f"{esc(l)}&thinsp;<span class=m>{v}</span>"
                             for l, v in bd["mark_letters"].most_common(15)))
    p.append("</p>")

    p.append(f"<h3>Letter, space and divider errors, top {top}</h3>"
             "<table><tr><th>reference<th>OCR<th>count<th>examples")
    for (a, b), v in bd["letters"].most_common(top):
        p.append(f"<tr><td class=txt>{vis(a)}<td class=txt>{vis(b)}<td class=n>{v}"
                 f"<td class=m>{ex_links(('letter', (a, b)))}")
    p.append("</table>")
    return "".join(p)


# -------------------------------------------------------------------- ocr ---

def ocr(image: Path, model: str, psm: int, tessdata: str) -> str:
    cmd = ["tesseract", str(image), "stdout", "--psm", str(psm), "-l", model]
    if tessdata:
        cmd += ["--tessdata-dir", tessdata]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        sys.exit("ERROR: 'tesseract' not found on PATH.")
    except subprocess.CalledProcessError as e:
        sys.exit(f"tesseract failed on {image}:\n{e.stderr}")
    return out.stdout


# ------------------------------------------------------------------- main ---

def find_image(stem_path: Path):
    for ext in IMG_EXT:
        cand = stem_path.with_suffix("")  # strip .txt
        p = Path(str(cand).removesuffix(".gt")) if str(cand).endswith(".gt") else cand
        img = Path(str(p) + ext)
        if img.exists():
            return img
    return None


def model_path(tessdata_dir: str, model: str) -> Path | None:
    """Best-effort path to the .traineddata actually used, for staleness checks."""
    if tessdata_dir:
        p = Path(tessdata_dir) / f"{model}.traineddata"
        return p if p.exists() else None
    for env in ("TESSDATA_PREFIX",):
        import os
        if os.environ.get(env):
            p = Path(os.environ[env]) / f"{model}.traineddata"
            if p.exists():
                return p
    return None


BOOK_RE = re.compile(r"_p\d+_l\d+$")

def book_tag(gt_path: Path) -> str:
    """<book>_p<pg>_l<ln>.gt.txt -> <book>, matching extract_lines.py's naming.
    Anything that doesn't fit the convention is grouped separately rather than
    silently mis-tagged, so a naming outlier shows up instead of hiding."""
    stem = gt_path.name[: -len(".gt.txt")] if gt_path.name.endswith(".gt.txt") \
        else gt_path.stem
    m = BOOK_RE.search(stem)
    return stem[: m.start()] if m else "(unrecognized filename pattern)"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, help="dir of .gt.txt + matching line images")
    ap.add_argument("--model", default="cu", help="tesseract lang/model (default cu)")
    ap.add_argument("--psm", type=int, default=13, help="page-seg mode (13 = raw line)")
    ap.add_argument("--tessdata-dir", default="", help="custom tessdata dir")
    ap.add_argument("--reocr", action="store_true", help="ignore cached .hyp.txt")
    ap.add_argument("--top", type=int, default=60, help="worst-N lines in the HTML")
    ap.add_argument("--by-book", dest="by_book", action="store_true", default=True,
                    help="report CER per source book, derived from each "
                         "<book>_p<pg>_l<ln>.gt.txt filename (default: on). "
                         "Lets you tell whether error is concentrated in "
                         "specific books (a data-volume problem) or spread "
                         "evenly across all of them (points elsewhere, e.g. "
                         "scan preprocessing).")
    ap.add_argument("--no-by-book", dest="by_book", action="store_false")
    ap.add_argument("--split", action="append", metavar="NAME=CHARS",
                    default=None,
                    help="report CER for lines containing any of CHARS as a "
                         "named bucket, vs. the rest. Repeatable for several "
                         "buckets (checked in order given; a line lands in "
                         "the first bucket it matches). Default: "
                         "'melisma=~\u203F' 'numerals=0123456789'. "
                         "Pass --split '' to disable.")
    ap.add_argument("--bootstrap", type=int, default=2000, metavar="N",
                    help="resamples for the 95%% confidence intervals "
                         "(default 2000; 0 = off). Lines are resampled, so "
                         "the interval shows how much the score could move "
                         "with a different test set of the same size.")
    ap.add_argument("--compare", type=Path, metavar="OLD_TSV",
                    help="a --tsv file from an earlier run (e.g. the current "
                         "model's model/eval/metrics.tsv). Prints the CER "
                         "change on the lines both runs share, with a paired "
                         "95%% confidence interval, and whether the change is "
                         "clearly real.")
    ap.add_argument("--report", default="report.html")
    ap.add_argument("--tsv", default="metrics.tsv")
    args = ap.parse_args()

    gts = sorted(args.dir.rglob("*.gt.txt"))
    if not gts:
        sys.exit(f"no *.gt.txt under {args.dir}")

    mpath = model_path(args.tessdata_dir, args.model)
    if mpath is None and not args.reocr:
        print(f"  NOTE: couldn't locate {args.model}.traineddata under "
              f"'{args.tessdata_dir or '(default tessdata dir)'}' to check cache "
              "staleness -- pass --reocr if you've retrained since the last eval "
              "run, or cached .hyp.txt may silently score an older model.",
              file=sys.stderr)
    model_mtime = mpath.stat().st_mtime if mpath else None
    stale_found = 0

    rows = []
    tot = dict(cd=0, cn=0, wd=0, wn=0, scd=0, scn=0)  # char/word dist & counts, full & stripped
    identical = 0

    for gt in gts:
        ref = gt.read_text(encoding="utf-8").strip()
        if not ref:
            continue
        img = find_image(gt)
        if img is None:
            print(f"  (no image for {gt.name}, skipped)", file=sys.stderr)
            continue

        hyp_cache = img.with_suffix(img.suffix + ".hyp.txt")
        cache_stale = (model_mtime is not None and hyp_cache.exists()
                       and hyp_cache.stat().st_mtime < model_mtime)
        if cache_stale:
            stale_found += 1
        if hyp_cache.exists() and not args.reocr and not cache_stale:
            hyp = hyp_cache.read_text(encoding="utf-8").strip()
        else:
            hyp = ocr(img, args.model, args.psm, args.tessdata_dir).strip()
            hyp_cache.write_text(hyp, encoding="utf-8")

        ref, hyp = normalize_divider(ref), normalize_divider(hyp)
        if hyp == ref:
            identical += 1

        ref_s, hyp_s = strip_marks(ref), strip_marks(hyp)
        cd = levenshtein(ref, hyp)
        wd = levenshtein(ref.split(), hyp.split())
        scd = levenshtein(ref_s, hyp_s)
        cer = cd / len(ref)
        wer = wd / max(1, len(ref.split()))
        cer_s = scd / max(1, len(ref_s))

        tot["cd"] += cd; tot["cn"] += len(ref)
        tot["wd"] += wd; tot["wn"] += len(ref.split())
        tot["scd"] += scd; tot["scn"] += len(ref_s)

        rows.append(dict(img=img, ref=ref, hyp=hyp, cer=cer, wer=wer,
                         cer_s=cer_s, nchar=len(ref), book=book_tag(gt),
                         cd=cd, scd=scd, nchar_s=len(ref_s)))

    if not rows:
        sys.exit("no scorable pairs found.")

    rows.sort(key=lambda r: r["cer"], reverse=True)
    micro = lambda d, n: 100 * d / max(1, n)

    print(f"\n  lines scored     : {len(rows)}")
    print(f"  reference chars  : {tot['cn']}")
    print(f"  ── micro-averaged ──")
    def ci(key_d, key_n):
        b = bootstrap_ci([(r[key_d], r[key_n]) for r in rows], args.bootstrap)
        return f"   (95% CI {b[0]:.2f}–{b[1]:.2f})" if b else ""
    print(f"  CER (full)       : {micro(tot['cd'], tot['cn']):6.2f}%{ci('cd', 'nchar')}")
    print(f"  CER (no marks)   : {micro(tot['scd'], tot['scn']):6.2f}%{ci('scd', 'nchar_s')}"
          f"   <- letterform-only error")
    print(f"  WER (full)       : {micro(tot['wd'], tot['wn']):6.2f}%")
    gap = micro(tot['cd'], tot['cn']) - micro(tot['scd'], tot['scn'])
    print(f"  diacritic share  : {gap:6.2f} pts of CER live in the marks")
    if identical:
        print(f"\n  NOTE: {identical} line(s) had OCR == reference exactly. If you did "
              f"not verify\n        those against the image, they may be uncorrected "
              f"and bias CER down.")
    if args.compare:
        old = read_tsv_edits(args.compare)
        common = [r for r in rows if r["img"].name in old]
        print(f"\n  ── compared with {args.compare} ──")
        if not common:
            print("  no lines in common; nothing to compare.")
        else:
            if len(common) != len(rows) or len(common) != len(old):
                print(f"  NOTE: only the {len(common)} lines in both runs are compared "
                      f"(this run {len(rows)}, earlier run {len(old)}).")
            new_p = [(r["cd"], r["nchar"]) for r in common]
            old_p = [old[r["img"].name] for r in common]
            nb = max(args.bootstrap, 2000)
            change, lo, hi, better = paired_bootstrap(new_p, old_p, nb)
            o = 100 * sum(d for d, _ in old_p) / max(1, sum(n for _, n in old_p))
            print(f"  CER (full)       : {o:6.2f}% -> {o + change:6.2f}%   "
                  f"change {change:+.2f} pts (95% CI {lo:+.2f} to {hi:+.2f})")
            if hi < 0:
                verdict = "BETTER: the improvement is larger than the noise."
            elif lo > 0:
                verdict = "WORSE: the new run is clearly worse."
            else:
                verdict = ("NO CLEAR DIFFERENCE: the change is within the noise "
                           "of this test set.")
            print(f"  verdict          : {verdict}")
            print(f"  (new run better in {100 * better:.0f}% of {nb} resamples)")

    if stale_found:
        print(f"\n  NOTE: {stale_found} cached .hyp.txt predated {args.model}.traineddata "
              f"and were\n        automatically re-OCR'd. Scores reflect the CURRENT model.")

    splits = (["melisma=~\u203F", "numerals=0123456789"]
              if args.split is None else [s for s in args.split if s])
    if splits:
        def cer_of(group):
            cd = sum(levenshtein(r["ref"], r["hyp"]) for r in group)
            cn = sum(len(r["ref"]) for r in group)
            return micro(cd, cn), cn, cd

        buckets = []
        for spec in splits:
            name, _, chars = spec.partition("=")
            if not chars:
                sys.exit(f"--split expects NAME=CHARS, got {spec!r}")
            buckets.append((name, chars))

        remaining = list(rows)
        total_err = sum(levenshtein(r["ref"], r["hyp"]) for r in rows)
        print(f"\n  ── split by construct ──")
        any_matched = False
        for name, chars in buckets:
            matched, unmatched = [], []
            for r in remaining:
                (matched if any(c in r["ref"] for c in chars) else unmatched).append(r)
            remaining = unmatched
            if not matched:
                continue
            any_matched = True
            c, n, d = cer_of(matched)
            shown = " ".join(repr(ch) for ch in chars)
            print(f"  {name:<10s}: {len(matched):4d} lines  ({n:5d} chars)  "
                  f"CER {c:6.2f}%   [{shown}]")
            if total_err:
                print(f"  {'':<10s}  {d/total_err*100:5.1f}% of total error, from "
                      f"{n/tot['cn']*100:.1f}% of reference characters")
        if any_matched and remaining:
            c, n, d = cer_of(remaining)
            print(f"  {'other':<10s}: {len(remaining):4d} lines  ({n:5d} chars)  "
                  f"CER {c:6.2f}%")

    if args.by_book:
        by_book = {}
        for r in rows:
            by_book.setdefault(r["book"], []).append(r)
        if len(by_book) > 1:
            def cer_of(group):
                cd = sum(levenshtein(r["ref"], r["hyp"]) for r in group)
                cn = sum(len(r["ref"]) for r in group)
                return micro(cd, cn), cn
            ranked = sorted(by_book.items(),
                            key=lambda kv: cer_of(kv[1])[0], reverse=True)
            name_w = max(len(name) for name, _ in ranked)
            print(f"\n  ── by book ──")
            for name, group in ranked:
                c, n = cer_of(group)
                print(f"  {name:<{name_w}s} : {len(group):4d} lines  "
                      f"({n:5d} chars)  CER {c:6.2f}%")

    bd = error_breakdown(rows)
    print(f"\n  ── error breakdown ({bd['total']} errors; per letter-with-marks) ──")
    for k, v in bd["kinds"].most_common(6):
        print(f"  {k:<26s}: {v:5d}  ({100 * v / max(1, bd['total']):4.1f}%)")
    rep = len(bd["repeated_hyp"])
    print(f"  repeated mark in OCR      : {rep} of {len(rows)} lines")
    if bd["repeated_ref"]:
        print(f"  NOTE: {len(bd['repeated_ref'])} reference line(s) have the same mark "
              f"twice on one letter (typo?): {', '.join(bd['repeated_ref'][:5])}")
    print(f"  (full tables in {args.report})")

    print(f"\n  worst lines:")
    for r in rows[:10]:
        print(f"   {r['cer']*100:6.1f}%  ref: {r['ref'][:55]}")
        print(f"           hyp: {r['hyp'][:55]}")

    # TSV
    with open(args.tsv, "w", encoding="utf-8") as f:
        f.write("cer_full\twer_full\tcer_nomarks\tref_chars\timage\treference\thypothesis\tchar_edits\n")
        for r in rows:
            f.write(f"{r['cer']*100:.2f}\t{r['wer']*100:.2f}\t{r['cer_s']*100:.2f}\t"
                    f"{r['nchar']}\t{r['img'].name}\t{r['ref']}\t{r['hyp']}\t{r['cd']}\n")

    # HTML
    esc = html.escape
    parts = ["""<!doctype html><meta charset=utf-8><title>cu_eval</title><style>
body{font:15px/1.5 system-ui,sans-serif;margin:2rem;max-width:1000px}
.row{border-top:1px solid #ddd;padding:1rem 0}
img{max-width:100%;background:#fff;border:1px solid #ccc}
.m{color:#666;font-size:13px}.cer{font-weight:600;color:#c0504d}
.lbl{display:inline-block;width:3.5em;color:#888}
del{background:#ffd6d6;text-decoration:none}ins{background:#d6ffd6;text-decoration:none}
.txt{font-size:20px}
table{border-collapse:collapse;margin:.5rem 0 1.5rem}
th,td{border-bottom:1px solid #eee;padding:.2rem .7rem;text-align:left;vertical-align:top}
td.n{text-align:right;font-variant-numeric:tabular-nums}</style>"""]
    shown = {r["img"].name for r in rows[:args.top]}
    parts.append(f"<h1>cu_eval — {len(rows)} lines</h1>")
    parts.append(breakdown_html(bd, len(rows), shown=shown))
    parts.append(f"<h2>Worst {min(args.top,len(rows))} lines</h2>")
    for r in rows[:args.top]:
        b64 = base64.b64encode(r["img"].read_bytes()).decode()
        mime = "image/png" if r["img"].suffix.lower().endswith("png") else "image/tiff"
        anchor = html.escape(r["img"].name.rsplit(".", 1)[0])
        parts.append(
            f'<div class=row id="{anchor}"><span class=m>{anchor}</span><br><span class=cer>CER {r["cer"]*100:.1f}%</span> '
            f'<span class=m>(no-marks {r["cer_s"]*100:.1f}% · WER {r["wer"]*100:.1f}%)</span>'
            f'<br><img src="data:{mime};base64,{b64}"><br>'
            f'<div class=txt><span class=lbl>ref</span>{esc(r["ref"])}</div>'
            f'<div class=txt><span class=lbl>ocr</span>{esc(r["hyp"])}</div>'
            f'<div class=txt><span class=lbl>diff</span>{html_diff(r["ref"], r["hyp"])}</div>'
            f'</div>')
    Path(args.report).write_text("".join(parts), encoding="utf-8")
    print(f"\n  wrote {args.tsv} and {args.report}")


if __name__ == "__main__":
    main()
