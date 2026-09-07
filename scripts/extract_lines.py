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

Optional preprocessing (--margin, --deskew, --binarize), applied to the
rasterized page before segmentation/OCR *and* before line crops are cut, so a
noisy scan gets one consistent treatment rather than segmentation seeing the
raw page and the saved crop showing something else:
  * --margin    drop a page border (printed frame, scan edge) by insetting the
                page, or by masking the band white and keeping the page size
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


def parse_margin(spec: str):
    """Parse a --margin spec into four (value, is_percent) pairs, in
    TOP,RIGHT,BOTTOM,LEFT order.

    CSS-style shorthand is accepted: one value for all four sides, two for
    vertical,horizontal, three for top,horizontal,bottom, or all four. A value
    is px by default ("40", "40px"), or a share of the page's own dimension
    when it ends in % ("5%" -- of height for top/bottom, of width for
    left/right), which is what you want if the same spec has to hold at more
    than one --dpi.
    """
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) == 1:
        parts *= 4
    elif len(parts) == 2:
        parts = [parts[0], parts[1], parts[0], parts[1]]
    elif len(parts) == 3:
        parts = [parts[0], parts[1], parts[2], parts[1]]
    elif len(parts) != 4:
        raise ValueError(f"expected 1, 2, 3 or 4 comma-separated values, got {len(parts)}")

    out = []
    for part in parts:
        pct = part.endswith("%")
        num = part[:-1] if pct else part[:-2] if part[-2:].lower() == "px" else part
        try:
            value = float(num)
        except ValueError:
            raise ValueError(f"bad margin value {part!r} (expected e.g. 40, 40px or 5%)") from None
        if value < 0:
            raise ValueError(f"negative margin value {part!r}")
        out.append((value, pct))
    return out


def resolve_margin(margin, W: int, H: int):
    """Four (value, is_percent) pairs -> a (left, top, right, bottom) box of
    the page area to keep, resolved against this page's own pixel size (so a %
    spec follows pages that differ in size within one book)."""
    # margin order is TOP, RIGHT, BOTTOM, LEFT: the even entries are vertical
    # (measured against height), the odd ones horizontal (against width).
    top, right, bottom, left = (
        round(value * (H if i % 2 == 0 else W) / 100.0) if pct else round(value)
        for i, (value, pct) in enumerate(margin))
    box = (left, top, W - right, H - bottom)
    if box[2] - box[0] < 1 or box[3] - box[1] < 1:
        sys.exit(f"ERROR: --margin leaves nothing of the {W}x{H}px page "
                 f"(kept box {box}); check the units -- px values scale with --dpi")
    return box


def apply_margin(img: Image.Image, box, mask: bool = False) -> Image.Image:
    """Remove everything outside `box`, either by cropping the page down to it
    (the default: the page really does get smaller) or, with mask=True, by
    painting the border band white and leaving the page at its original size.

    Both delete the same ink, so on a page with a comfortable border the two
    usually produce identical output. They part company at the edges, via
    --pad: cropping clamps a line's padding at the new page edge, so a line
    sitting right against the cut gets none of it, while masking leaves real
    white pixels there for the pad to land on. That padding is what tesseract
    needs to read an edge line cleanly, so mask is the better choice when the
    margin has to cut close to the text. Masking also keeps box coordinates
    comparable to the un-margined page."""
    if not mask:
        return img.crop(box)
    l, t, r, b = box
    arr = np.asarray(img).copy()
    arr[:t, :] = 255
    arr[b:, :] = 255
    arr[:, :l] = 255
    arr[:, r:] = 255
    return Image.fromarray(arr, mode="L")


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
    # Set the TSV-output variable directly with -c instead of naming the "tsv"
    # configfile: the configfile has to be *found* (tessdata-dir/configs/tsv),
    # which fails silently -- falling back to plain text, which we'd then
    # parse as zero rows -- whenever --tessdata-dir points at a directory that
    # (like this project's model/) holds only a .traineddata, no configs/.
    cmd = ["tesseract", str(page_img), "stdout", "--psm", str(psm), "-l", model,
           "-c", "tessedit_create_tsv=1"]
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
    ap.add_argument("--margin", default="", metavar="TOP,RIGHT,BOTTOM,LEFT",
                    help="drop a page border before segmentation: a printed frame, "
                         "a scan edge, a running header. Values are px (40, 40px) or "
                         "a share of the page (5%%), and CSS shorthand works: one value "
                         "for all sides, two for vertical,horizontal. See --margin-mode")
    ap.add_argument("--margin-mode", choices=("crop", "mask"), default="crop",
                    help="crop: inset the page, discarding the border (default). "
                         "mask: paint the border white but keep the page size, which "
                         "leaves segmentation a quiet zone around the text block")
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

    margin = None
    if args.margin:
        try:
            margin = parse_margin(args.margin)
        except ValueError as e:
            ap.error(f"--margin: {e}")
    elif args.margin_mode != "crop":
        # a lone --margin-mode does nothing; say so rather than silently
        # running with no border removal at all
        ap.error("--margin-mode has no effect without --margin")

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
            if margin:
                # first, before deskew: a frame's long straight rules and its
                # solid side bands put ink in every row, flattening the
                # projection profile the skew search scores. It also keeps the
                # spec in the coordinates you measured it in -- deskew rotates
                # with expand=True and so changes the page size.
                page = apply_margin(page, resolve_margin(margin, *page.size),
                                    args.margin_mode == "mask")
            if args.deskew:
                page = deskew(page, args.deskew_range, args.deskew_step)
            if args.binarize:
                page = sauvola_binarize(page, args.sauvola_window, args.sauvola_k)
            if margin or args.deskew or args.binarize:
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
    if args.out == Path("data/real-lines/staging"):
        next_step = "make review-staging"
    else:
        next_step = f"python3 scripts/review_staging.py --dir {args.out}"
    print(f"Next: review and correct these in the browser, then file each pair "
          f"into eval/ or finetune/ --  {next_step}", file=sys.stderr)


if __name__ == "__main__":
    main()
