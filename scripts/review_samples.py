#!/usr/bin/env python3
"""review_samples.py -- eyeball a few training pairs from a huge ground-truth dir
without your shell or file manager choking on the file count.

It uses os.scandir (a lazy iterator), so it never materializes the whole file
list. It collects the first N .gt.txt whose text contains a substring (default
'_', i.e. hyphenated lines), prints their ground truth to the terminal, and
stacks their line images into ONE montage PNG you can open.

Usage:
  python3 review_samples.py tesstrain/data/cu-ground-truth --n 8
  python3 review_samples.py <dir> --contains '⸗' --n 12 --out sheet.png
"""

import argparse
import os
import sys

from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--n", type=int, default=8, help="how many samples")
    ap.add_argument("--contains", default="_",
                    help="only ground truth containing this substring (default '_')")
    ap.add_argument("--every", type=int, default=1,
                    help="take every Nth match, to spread across books (default 1)")
    ap.add_argument("--out", default="review.png")
    args = ap.parse_args()

    picks, seen = [], 0
    with os.scandir(args.dir) as it:          # lazy: never builds a full listing
        for e in it:
            if not e.name.endswith(".gt.txt"):
                continue
            try:
                txt = open(e.path, encoding="utf-8").read()
            except OSError:
                continue
            if args.contains and args.contains not in txt:
                continue
            seen += 1
            if (seen - 1) % args.every:
                continue
            png = e.path[:-len(".gt.txt")] + ".png"
            if os.path.exists(png):
                picks.append((txt, png))
            if len(picks) >= args.n:
                break

    if not picks:
        sys.exit(f"no .gt.txt containing {args.contains!r} found in {args.dir}")

    print(f"{len(picks)} sample(s) (ground truth keeps '_'; image shows the hyphen glyph):")
    imgs = []
    for i, (txt, png) in enumerate(picks):
        print(f"  [{i}] {txt}")
        imgs.append((i, Image.open(png).convert("L")))

    pad, gap, lblw = 8, 8, 26
    W = max(im.width for _, im in imgs) + 2 * pad + lblw
    H = sum(im.height for _, im in imgs) + gap * len(imgs) + 2 * pad
    sheet = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(sheet)
    y = pad
    for i, im in imgs:
        d.text((pad, y + im.height // 2 - 4), str(i), fill=0)   # ASCII index label
        sheet.paste(im, (pad + lblw, y))
        y += im.height + gap
    sheet.save(args.out)
    print(f"\nmontage -> {args.out}   (open just this one file)")


if __name__ == "__main__":
    main()
