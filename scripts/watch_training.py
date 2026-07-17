#!/usr/bin/env python3
"""
watch_training.py -- live view of a tesstrain / lstmtraining run.

Tails a training log (that you produce with `... 2>&1 | tee training.log`),
parses the iteration lines, and plots character/word error over iterations,
updating in real time.

Two modes:
  * windowed (default):  a matplotlib window that refreshes itself.
  * headless (--headless): no window; rewrites a PNG every interval, so you
    can run it in the background (even over SSH) and open the image.

Usage
  # terminal 1 -- train, and capture the log:
  make training MODEL_NAME=cu MAX_ITERATIONS=100000 2>&1 | tee training.log

  # terminal 2 -- live window:
  python3 watch_training.py training.log

  # or background, headless, refreshing a PNG every 10s:
  python3 watch_training.py training.log --headless --png live.png --interval 10 &

Needs: pip install matplotlib
"""

import argparse
import os
import re
import sys
import time

# lstmtraining lines look like:
#   At iteration 12800/12800/12800, mean rms=6.864%, delta=72.853%,
#   BCER train=99.570%, BWER train=100.000%, skip ratio=0.000%,
# and, on evaluation passes:
#   At iteration 12800, BCER eval=97.10%, BWER eval=99.80%
# Older tesseract wrote "char train="/"word train=" instead of BCER/BWER.
RE_TRAIN = re.compile(r"At iteration (\d+)/\d+/\d+.*?(?:BCER|char) train=([\d.]+)%", re.I)
RE_BWER  = re.compile(r"(?:BWER|word) train=([\d.]+)%", re.I)
RE_EVAL  = re.compile(r"(?:BCER|char)[ _]?eval=([\d.]+)%", re.I)
RE_ITER  = re.compile(r"At iteration (\d+)")


class LogState:
    """Incrementally tails a growing file and accumulates parsed points."""
    def __init__(self, path):
        self.path = path
        self.pos = 0
        self.train = []   # (iteration, BCER%)
        self.bwer = []    # (iteration, BWER%)
        self.eval = []    # (iteration, BCER eval%)

    def poll(self):
        if not os.path.exists(self.path):
            return
        size = os.path.getsize(self.path)
        if size < self.pos:        # file truncated / rotated -> restart
            self.pos = 0
            self.train.clear(); self.bwer.clear(); self.eval.clear()
        with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(self.pos)
            chunk = f.read()
            self.pos = f.tell()
        for line in chunk.splitlines():
            m = RE_TRAIN.search(line)
            if m:
                it = int(m.group(1))
                self.train.append((it, float(m.group(2))))
                b = RE_BWER.search(line)
                if b:
                    self.bwer.append((it, float(b.group(1))))
                continue
            e = RE_EVAL.search(line)
            if e:
                mi = RE_ITER.search(line)
                it = int(mi.group(1)) if mi else (self.train[-1][0] if self.train else 0)
                self.eval.append((it, float(e.group(1))))


def draw(ax, st, args):
    ax.clear()
    if st.train:
        xs, ys = zip(*st.train)
        ax.plot(xs, ys, lw=1.3, color="#1f4e79", label="BCER train")
    if st.bwer:
        xs, ys = zip(*st.bwer)
        ax.plot(xs, ys, lw=0.8, color="#9db8d2", alpha=0.7, label="BWER train")
    if st.eval:
        xs, ys = zip(*st.eval)
        ax.plot(xs, ys, "o-", ms=4, color="#c0504d", label="BCER eval")

    ax.axhline(args.target, ls="--", lw=1, color="#3c8031",
               label=f"target {args.target:g}%")

    if not args.linear:
        ax.set_yscale("log")
        ax.set_ylim(max(0.1, args.floor), 120)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}%")
    ax.set_xlabel("iteration")
    ax.set_ylabel("error (log scale)" if not args.linear else "error %")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8)

    if st.train:
        it, cer = st.train[-1]
        best = min(c for _, c in st.eval) if st.eval else min(c for _, c in st.train)
        trend = ""
        if len(st.train) > 20:
            recent = [c for _, c in st.train[-20:]]
            trend = "  |  falling" if recent[-1] < recent[0] - 0.05 else "  |  ~flat (watch closely)"
        ax.set_title(f"iter {it:,}   train BCER {cer:.2f}%   best {best:.2f}%{trend}",
                     fontsize=10)
    else:
        ax.set_title("waiting for iteration lines in the log...", fontsize=10)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="path to the training log (produced via tee)")
    ap.add_argument("--interval", type=float, default=5, help="refresh seconds")
    ap.add_argument("--headless", action="store_true", help="no window; write a PNG")
    ap.add_argument("--png", default="training_progress.png", help="PNG path (headless)")
    ap.add_argument("--once", action="store_true", help="render once and exit (implies headless)")
    ap.add_argument("--target", type=float, default=2.0, help="target CER reference line")
    ap.add_argument("--floor", type=float, default=0.1, help="log-scale y floor %%")
    ap.add_argument("--linear", action="store_true", help="linear y axis")
    args = ap.parse_args()

    import matplotlib
    if args.headless or args.once:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st = LogState(args.log)

    if args.once:
        st.poll()
        fig, ax = plt.subplots(figsize=(9, 5))
        draw(ax, st, args)
        fig.tight_layout(); fig.savefig(args.png, dpi=110)
        print(f"wrote {args.png} ({len(st.train)} train pts, {len(st.eval)} eval pts)")
        return

    if args.headless:
        fig, ax = plt.subplots(figsize=(9, 5))
        try:
            while True:
                st.poll()
                draw(ax, st, args)
                fig.tight_layout(); fig.savefig(args.png, dpi=110)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nstopped.")
        return

    # windowed live mode
    from matplotlib.animation import FuncAnimation
    fig, ax = plt.subplots(figsize=(9, 5))

    def update(_):
        st.poll()
        draw(ax, st, args)

    _anim = FuncAnimation(fig, update, interval=int(args.interval * 1000),
                          cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
