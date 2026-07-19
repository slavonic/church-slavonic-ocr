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

Optional preprocessing (--deskew, --binarize), applied to the rasterized page
before segmentation/OCR *and* before line crops are cut, so a noisy scan gets
one consistent treatment rather than segmentation seeing the raw page and the
saved crop showing something else:
  * --deskew    projection-profile skew estimate + rotation correction
  * --binarize  Sauvola local-threshold binarization
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
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


def _box_sums(arr: np.ndarray, window: int) -> np.ndarray:
    """Sum of each window x window neighborhood, via an integral image (O(HW)
    regardless of window size). `arr` is edge-reflect padded by the caller."""
    h, w = arr.shape[0] - window, arr.shape[1] - window
    ii = np.zeros((arr.shape[0] + 1, arr.shape[1] + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(arr, axis=0), axis=1)
    y0 = np.arange(h + 1)[:, None]
    x0 = np.arange(w + 1)[None, :]
    y1, x1 = y0 + window, x0 + window
    return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]


def sauvola_binarize(img: Image.Image, window: int = 25, k: float = 0.2, r: float = 128.0) -> Image.Image:
    """Sauvola local-threshold binarization: a pixel is ink if it's darker than
    a threshold set from the local mean and local contrast (std), so uneven
    scan lighting doesn't blow out one part of the page while crushing another
    (the failure mode a single global threshold has on real scans)."""
    if window % 2 == 0:
        window += 1
    arr = np.asarray(img, dtype=np.float64)
    pad = window // 2
    padded = np.pad(arr, pad, mode="reflect")
    n = window * window
    s1 = _box_sums(padded, window)
    s2 = _box_sums(padded * padded, window)
    mean = s1 / n
    var = np.maximum(s2 / n - mean * mean, 0)
    std = np.sqrt(var)
    thresh = mean * (1 + k * (std / r - 1))
    out = np.where(arr > thresh, 255, 0).astype(np.uint8)
    return Image.fromarray(out, mode="L")


def estimate_skew(img: Image.Image, angle_range: float = 5.0, step: float = 0.2,
                   max_side: int = 1000) -> float:
    """Best-fit skew angle in degrees: rotate a downscaled ink mask through
    candidate angles and pick the one whose horizontal projection (row ink
    sums) has the most variance -- text lines packed tightly onto their
    baselines produce sharp peaks/troughs; a skewed page smears them out."""
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    small = img.resize((max(1, round(w * scale)), max(1, round(h * scale)))) if scale < 1 else img
    arr = np.asarray(small, dtype=np.float64)
    mask = Image.fromarray(((arr < arr.mean()) * 255).astype(np.uint8))  # ink -> 255

    best_angle, best_score = 0.0, -1.0
    angle = -angle_range
    while angle <= angle_range + 1e-9:
        rotated = mask.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=0)
        score = np.asarray(rotated, dtype=np.float64).sum(axis=1).var()
        if score > best_score:
            best_score, best_angle = score, angle
        angle += step
    return best_angle


def deskew(img: Image.Image, angle_range: float = 5.0, step: float = 0.2) -> Image.Image:
    angle = estimate_skew(img, angle_range, step)
    if abs(angle) < 1e-6:
        return img
    return img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=255)


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
    ap.add_argument("--deskew", action="store_true",
                    help="correct page skew (projection-profile estimate) before "
                         "segmentation/OCR and before cropping")
    ap.add_argument("--deskew-range", type=float, default=5.0, metavar="DEG",
                    help="search +/-DEG for the best skew correction (default 5)")
    ap.add_argument("--deskew-step", type=float, default=0.2, metavar="DEG",
                    help="skew search resolution in degrees (default 0.2)")
    ap.add_argument("--binarize", action="store_true",
                    help="Sauvola local-threshold binarization before "
                         "segmentation/OCR and before cropping")
    ap.add_argument("--sauvola-window", type=int, default=25, metavar="PX",
                    help="Sauvola local-neighborhood size in px, forced odd (default 25)")
    ap.add_argument("--sauvola-k", type=float, default=0.2,
                    help="Sauvola sensitivity constant, higher = stricter/more ink "
                         "kept only where contrast is high (default 0.2)")
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
            if args.deskew:
                page = deskew(page, args.deskew_range, args.deskew_step)
            if args.binarize:
                page = sauvola_binarize(page, args.sauvola_window, args.sauvola_k)
            if args.deskew or args.binarize:
                # re-run segmentation/OCR on the preprocessed page too, so the
                # boxes and the saved crop reflect the same image
                img_path = Path(tmp) / f"page_{pg:04d}_proc.png"
                page.save(img_path)
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
