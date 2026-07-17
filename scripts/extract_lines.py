#!/usr/bin/env python3
"""
extract_lines.py -- turn PDF/DJVU book pages into line crops + editable .gt.txt,
ready for correction and then for cu_eval.py (and for fine-tuning).

For each requested page it rasterizes the page, runs your model to get line
boxes + a first-pass transcription, crops each text line, and writes:

    <out>/<book>_p<pg>_l<ln>.png       the line crop
    <out>/<book>_p<pg>_l<ln>.gt.txt    the OCR text (YOU then correct this)

The .gt.txt is pre-filled with the model's guess so correcting is editing,
not transcribing from scratch. After you fix them, the same folder is a valid
input to cu_eval.py and a valid tesstrain fine-tuning set.

Inputs:
  * PDF  -> rendered with PyMuPDF        (pip install pymupdf pillow)
  * DJVU -> rendered with ddjvu          (sudo apt install djvulibre-bin)
  * line segmentation + first-pass OCR   -> tesseract on PATH, model installed
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image


def parse_pages(spec: str, npages: int):
    if spec in ("all", "*", ""):
        return list(range(npages))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a) - 1, int(b)))   # 1-based inclusive -> 0-based
        else:
            out.append(int(part) - 1)
    return [p for p in out if 0 <= p < npages]


def render_pdf(path, pages_spec, dpi, tmp):
    import fitz  # PyMuPDF
    doc = fitz.open(path)
    for p in parse_pages(pages_spec, doc.page_count):
        pix = doc[p].get_pixmap(dpi=dpi)
        out = Path(tmp) / f"page_{p+1:04d}.png"
        pix.save(out)
        yield p + 1, out


def render_djvu(path, pages_spec, dpi, tmp):
    # page count via djvused
    try:
        n = int(subprocess.run(["djvused", str(path), "-e", "n"],
                               capture_output=True, text=True, check=True).stdout.strip())
    except FileNotFoundError:
        sys.exit("ERROR: djvulibre not installed. sudo apt install djvulibre-bin")
    for p in parse_pages(pages_spec, n):
        out = Path(tmp) / f"page_{p+1:04d}.tif"
        # ddjvu: 1-based page numbers; -scale sets output resolution in dpi
        subprocess.run(["ddjvu", "-format=tiff", f"-page={p+1}", f"-scale={dpi}",
                        str(path), str(out)], check=True)
        yield p + 1, out


def tsv_lines(page_img, model, psm, tessdata):
    """Run tesseract TSV, yield (line_text, (l,t,r,b)) grouped by line."""
    cmd = ["tesseract", str(page_img), "stdout", "--psm", str(psm), "-l", model, "tsv"]
    if tessdata:
        cmd += ["--tessdata-dir", tessdata]
    tsv = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout

    lines = {}   # (block,par,line) -> [words...], each word = (num,l,t,r,b,text)
    for row in tsv.splitlines()[1:]:
        c = row.split("\t")
        if len(c) < 12 or c[0] != "5":       # level 5 = word
            continue
        text = c[11].strip()
        if not text:
            continue
        key = (c[2], c[3], c[4])
        l, t, w, h = int(c[6]), int(c[7]), int(c[8]), int(c[9])
        lines.setdefault(key, []).append((int(c[5]), l, t, l + w, t + h, text))

    for key in sorted(lines, key=lambda k: tuple(map(int, k))):
        words = sorted(lines[key], key=lambda x: x[0])
        text = " ".join(w[5] for w in words)
        l = min(w[1] for w in words); t = min(w[2] for w in words)
        r = max(w[3] for w in words); b = max(w[4] for w in words)
        yield text, (l, t, r, b)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="a .pdf or .djvu file")
    ap.add_argument("--pages", default="all", help="e.g. 12-15 or 3,5,7 or all (1-based)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default="cu")
    ap.add_argument("--dpi", type=int, default=400, help="rasterization dpi (300-400 good)")
    ap.add_argument("--psm", type=int, default=6, help="page-seg mode for line finding (6=block)")
    ap.add_argument("--pad", type=int, default=6, help="px padding around each line crop")
    ap.add_argument("--tessdata-dir", default="")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    book = args.input.stem
    ext = args.input.suffix.lower()

    with tempfile.TemporaryDirectory() as tmp:
        if ext == ".pdf":
            pages = render_pdf(args.input, args.pages, args.dpi, tmp)
        elif ext in (".djvu", ".djv"):
            pages = render_djvu(args.input, args.pages, args.dpi, tmp)
        else:
            sys.exit(f"unsupported input: {ext} (use .pdf or .djvu)")

        total = 0
        for pg, img_path in pages:
            page = Image.open(img_path).convert("L")
            W, H = page.size
            n = 0
            for text, (l, t, r, b) in tsv_lines(img_path, args.model, args.psm,
                                                args.tessdata_dir):
                n += 1
                l = max(0, l - args.pad); t = max(0, t - args.pad)
                r = min(W, r + args.pad); b = min(H, b + args.pad)
                stem = args.out / f"{book}_p{pg:04d}_l{n:03d}"
                page.crop((l, t, r, b)).save(f"{stem}.png")
                (Path(f"{stem}.gt.txt")).write_text(text, encoding="utf-8")
            total += n
            print(f"  page {pg}: {n} lines", file=sys.stderr)

    print(f"\nWrote {total} line pairs to {args.out}", file=sys.stderr)
    print("Next: correct every .gt.txt against its .png, delete any garbled-"
          "segmentation lines,\nthen run:  python3 cu_eval.py "
          f"{args.out} --model {args.model}", file=sys.stderr)


if __name__ == "__main__":
    main()
