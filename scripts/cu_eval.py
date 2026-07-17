#!/usr/bin/env python3
"""
cu_eval.py -- score a Church Slavonic Tesseract model against real, corrected lines.

Point it at a directory of reference `.gt.txt` files each beside a matching
line image (same stem: .png/.tif/.jpg/.bin.png). For every pair it OCRs the
image with your model and compares against the reference, reporting:

  * CER and WER, MICRO-averaged (total edits / total reference units)
  * the same computed with combining marks stripped, so you can see whether
    the errors are letterforms or just diacritic placement
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


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, help="dir of .gt.txt + matching line images")
    ap.add_argument("--model", default="cu", help="tesseract lang/model (default cu)")
    ap.add_argument("--psm", type=int, default=13, help="page-seg mode (13 = raw line)")
    ap.add_argument("--tessdata-dir", default="", help="custom tessdata dir")
    ap.add_argument("--reocr", action="store_true", help="ignore cached .hyp.txt")
    ap.add_argument("--top", type=int, default=60, help="worst-N lines in the HTML")
    ap.add_argument("--report", default="report.html")
    ap.add_argument("--tsv", default="metrics.tsv")
    args = ap.parse_args()

    gts = sorted(args.dir.rglob("*.gt.txt"))
    if not gts:
        sys.exit(f"no *.gt.txt under {args.dir}")

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
        if hyp_cache.exists() and not args.reocr:
            hyp = hyp_cache.read_text(encoding="utf-8").strip()
        else:
            hyp = ocr(img, args.model, args.psm, args.tessdata_dir).strip()
            hyp_cache.write_text(hyp, encoding="utf-8")

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
                         cer_s=cer_s, nchar=len(ref)))

    if not rows:
        sys.exit("no scorable pairs found.")

    rows.sort(key=lambda r: r["cer"], reverse=True)
    micro = lambda d, n: 100 * d / max(1, n)

    print(f"\n  lines scored     : {len(rows)}")
    print(f"  reference chars  : {tot['cn']}")
    print(f"  ── micro-averaged ──")
    print(f"  CER (full)       : {micro(tot['cd'], tot['cn']):6.2f}%")
    print(f"  CER (no marks)   : {micro(tot['scd'], tot['scn']):6.2f}%   "
          f"<- letterform-only error")
    print(f"  WER (full)       : {micro(tot['wd'], tot['wn']):6.2f}%")
    gap = micro(tot['cd'], tot['cn']) - micro(tot['scd'], tot['scn'])
    print(f"  diacritic share  : {gap:6.2f} pts of CER live in the marks")
    if identical:
        print(f"\n  NOTE: {identical} line(s) had OCR == reference exactly. If you did "
              f"not verify\n        those against the image, they may be uncorrected "
              f"and bias CER down.")

    print(f"\n  worst lines:")
    for r in rows[:10]:
        print(f"   {r['cer']*100:6.1f}%  ref: {r['ref'][:55]}")
        print(f"           hyp: {r['hyp'][:55]}")

    # TSV
    with open(args.tsv, "w", encoding="utf-8") as f:
        f.write("cer_full\twer_full\tcer_nomarks\tref_chars\timage\treference\thypothesis\n")
        for r in rows:
            f.write(f"{r['cer']*100:.2f}\t{r['wer']*100:.2f}\t{r['cer_s']*100:.2f}\t"
                    f"{r['nchar']}\t{r['img'].name}\t{r['ref']}\t{r['hyp']}\n")

    # HTML
    esc = html.escape
    parts = ["""<!doctype html><meta charset=utf-8><title>cu_eval</title><style>
body{font:15px/1.5 system-ui,sans-serif;margin:2rem;max-width:1000px}
.row{border-top:1px solid #ddd;padding:1rem 0}
img{max-width:100%;background:#fff;border:1px solid #ccc}
.m{color:#666;font-size:13px}.cer{font-weight:600;color:#c0504d}
.lbl{display:inline-block;width:3.5em;color:#888}
del{background:#ffd6d6;text-decoration:none}ins{background:#d6ffd6;text-decoration:none}
.txt{font-size:20px}</style>"""]
    parts.append(f"<h2>cu_eval — {len(rows)} lines, worst {min(args.top,len(rows))} shown</h2>")
    for r in rows[:args.top]:
        b64 = base64.b64encode(r["img"].read_bytes()).decode()
        mime = "image/png" if r["img"].suffix.lower().endswith("png") else "image/tiff"
        parts.append(
            f'<div class=row><span class=cer>CER {r["cer"]*100:.1f}%</span> '
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
