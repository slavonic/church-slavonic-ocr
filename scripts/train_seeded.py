#!/usr/bin/env python3
"""
train_seeded.py -- robust Church Slavonic training:
CU-only unicharset  +  Cyrillic feature seed  +  blank-collapse watchdog.

Why this exists
---------------
Two failure modes bracket this project:

  * tesstrain's START_MODEL=Cyrillic MERGES Cyrillic's unicharset into
    yours (merge_unicharsets), so the model can emit schwa, Latin, (c)...
  * from-scratch random init can fall into the CTC all-blank basin and
    sit at ~98-100% BCER forever (low rms, BWER ~100%, near-constant
    decode like a short high-frequency stub).

The escape is to combine the halves that work:

  1. unicharset + proto model are built from YOUR ground truth only
     (tesstrain's from-scratch path -> no merge, no foreign classes);
  2. training then CONTINUES from Cyrillic's extracted .lstm with
     --old_traineddata, so lstmtraining REMAPS the output layer to your
     clean charset while keeping Cyrillic's lower feature layers --
     the network starts already able to see glyphs, outside the basin.

Guards:
  * charset audit refuses to start if the built unicharset contains
    Latin letters, schwa, or other non-CU classes;
  * a watchdog parses lstmtraining's own progress lines; if BCER is
    still ~flat above a threshold past a probation point, it kills the
    run and retries with a stronger learning rate (up to --retries),
    so a collapsed run costs minutes, not a night.

Typical use (from the repo root):

    python3 scripts/train_seeded.py \
        --tesstrain ../tesstrain \
        --max-iterations 100000

Prereqs: tesseract training tools on PATH (lstmtraining, combine_tessdata,
combine_lang_model...), a tesstrain checkout, ground truth in
data/cu-ground-truth. Downloads Cyrillic.traineddata if absent.
"""

import argparse
import re
import shutil
import signal
import subprocess
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

CYRILLIC_URL = ("https://github.com/tesseract-ocr/tessdata_best/"
                "raw/main/script/Cyrillic.traineddata")

PROGRESS_RE = re.compile(
    r"At iteration (\d+)/(\d+)/(\d+),.*?BCER train=([\d.]+)%", re.S)


def run(cmd, **kw):
    print("  $", " ".join(map(str, cmd)), file=sys.stderr)
    return subprocess.run(list(map(str, cmd)), check=True, **kw)


def need(tool):
    if shutil.which(tool) is None:
        sys.exit(f"ERROR: '{tool}' not on PATH. Build/install the tesseract "
                 "training tools first (see docs/training.md).")


# --------------------------------------------------------------- charset ---

def audit_unicharset(path: Path):
    """Fail hard if the built unicharset contains non-CU junk classes."""
    def ok(ch: str) -> bool:
        if len(ch) != 1:
            return True                     # multi-char cluster lines: made of audited chars
        cp = ord(ch)
        if ch.isascii():
            # digits/punct are legitimate print charset; LETTERS are not
            return not ch.isalpha()
        if cp in (0x0401, 0x0451, 0x04D8, 0x04D9, 0x0259):   # Ё ё Ә ә ə
            return False
        return (0x0400 <= cp <= 0x052F      # Cyrillic + Supplement
                or 0x1C80 <= cp <= 0x1C8F   # Extended-C (narrow o...)
                or 0x2DE0 <= cp <= 0x2DFF   # combining Cyrillic letters
                or 0xA640 <= cp <= 0xA69F   # Extended-B
                or 0x0300 <= cp <= 0x036F   # combining accents
                or 0x1F540 <= cp <= 0x1F545 # liturgical symbols
                or cp in (0x00AB, 0x00BB, 0x2010, 0x2011, 0x2013, 0x2014,
                          0x2020, 0x00A0)   # guillemets, hyphens/dashes, dagger
                or unicodedata.category(ch).startswith(("P", "Z")))

    bad = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for ln in lines[1:]:                    # first line is the class count
        sym = ln.split(" ")[0]
        if sym in ("NULL", "Joined", "|Broken|0|1"):
            continue
        for ch in sym:
            if not ok(ch):
                bad.append((sym, ch))
                break
    n = lines[0].strip() if lines else "?"
    print(f"  unicharset: {n} classes", file=sys.stderr)
    if bad:
        print("ERROR: non-CU classes in unicharset -- refusing to train on a "
              "contaminated charset:", file=sys.stderr)
        for sym, ch in bad[:20]:
            print(f"    {sym!r}  (offender U+{ord(ch):04X} "
                  f"{unicodedata.name(ch, '?')})", file=sys.stderr)
        sys.exit("Clean the ground truth (cu_make_training_data.py "
                 "--charset-filter drop) and rerun.")
    print("  charset audit: clean CU", file=sys.stderr)


# -------------------------------------------------------------- training ---

def launch_and_watch(cmd, log_path: Path, probation: int, ceiling: float):
    """Run lstmtraining, tee output to log, watch for blank collapse.

    Returns 'ok' if the run finished, 'collapse' if we killed it because
    BCER stayed >= ceiling past the probation iteration with no downtrend.
    """
    print("  $", " ".join(map(str, cmd)), file=sys.stderr)
    with open(log_path, "a", encoding="utf-8") as log:
        proc = subprocess.Popen(list(map(str, cmd)), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        history = []                        # (iteration, bcer)
        try:
            for line in proc.stdout:
                log.write(line)
                if "At iteration" in line:
                    m = PROGRESS_RE.search(line)
                    if m:
                        it, bcer = int(m.group(2)), float(m.group(4))
                        history.append((it, bcer))
                        if it % 1000 == 0:
                            print(f"    iter {it:6d}  BCER {bcer:6.2f}%",
                                  file=sys.stderr)
                        if it >= probation and bcer >= ceiling:
                            # any meaningful downtrend over the last stretch?
                            past = [b for i, b in history if i <= it - probation // 2]
                            best_past = min(past) if past else 100.0
                            if best_past - bcer < 2.0:   # <2 points of progress
                                print(f"  WATCHDOG: BCER {bcer:.1f}% at iter "
                                      f"{it} with no downtrend -> blank "
                                      "collapse, aborting this attempt.",
                                      file=sys.stderr)
                                proc.send_signal(signal.SIGINT)
                                time.sleep(3)
                                proc.kill()
                                proc.wait()
                                return "collapse"
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGINT)
            proc.wait()
            raise
        proc.wait()
    return "ok" if proc.returncode == 0 else "error"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tesstrain", type=Path, default=Path("../tesstrain"),
                    help="path to a tesseract-ocr/tesstrain checkout")
    ap.add_argument("--model-name", default="cu")
    ap.add_argument("--data-dir", type=Path, default=Path("training"))
    ap.add_argument("--ground-truth", type=Path,
                    default=Path("data/cu-ground-truth"))
    ap.add_argument("--seed-model", type=Path, default=None,
                    help="Cyrillic.traineddata (downloaded if absent)")
    ap.add_argument("--max-iterations", type=int, default=100000)
    ap.add_argument("--learning-rate", type=float, default=1e-3,
                    help="initial learning rate (doubled on each collapse retry)")
    ap.add_argument("--retries", type=int, default=2,
                    help="restart attempts if blank collapse is detected")
    ap.add_argument("--probation", type=int, default=6000,
                    help="iteration after which flat ~100%% BCER = collapse "
                         "(seeded runs break down almost immediately)")
    ap.add_argument("--ceiling", type=float, default=90.0,
                    help="BCER%% treated as 'still collapsed' after probation")
    ap.add_argument("--jobs", default=None, help="parallel jobs for tesstrain "
                    "preprocessing (default: nproc)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and commands without executing training")
    args = ap.parse_args()

    for t in ("lstmtraining", "combine_tessdata", "unicharset_extractor"):
        need(t)
    if not (args.tesstrain / "Makefile").exists():
        sys.exit(f"ERROR: no tesstrain Makefile at {args.tesstrain}. "
                 "git clone https://github.com/tesseract-ocr/tesstrain")

    data_dir = args.data_dir.resolve()
    gt_dir = args.ground_truth.resolve()
    out = data_dir / args.model_name
    ckpt = out / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    log_path = data_dir / "training.log"

    # ---- 1. CU-only unicharset + proto model + lists (NO START_MODEL) ----
    mk = ["make", "-C", args.tesstrain,
          f"MODEL_NAME={args.model_name}",
          f"DATA_DIR={data_dir}", f"GROUND_TRUTH_DIR={gt_dir}",
          f"-j{args.jobs}" if args.jobs else f"-j{__import__('os').cpu_count()}"]
    print("\n[1/4] unicharset + proto model + lists from YOUR ground truth only",
          file=sys.stderr)
    if not args.dry_run:
        run(mk + ["unicharset", "lists", "proto-model"])
        audit_unicharset(out / "unicharset")
    else:
        print("  (dry-run)", " ".join(map(str, mk + ["unicharset", "lists",
              "proto-model"])), file=sys.stderr)

    # ---- 2. seed model -----------------------------------------------------
    print("\n[2/4] Cyrillic seed (feature layers only)", file=sys.stderr)
    seed_td = args.seed_model or (data_dir / "Cyrillic.traineddata")
    if not seed_td.exists() and not args.dry_run:
        print(f"  downloading {CYRILLIC_URL}", file=sys.stderr)
        urllib.request.urlretrieve(CYRILLIC_URL, seed_td)
    seed_lstm = data_dir / "Cyrillic.lstm"
    if not args.dry_run:
        # -e extracts the named component out of the traineddata
        run(["combine_tessdata", "-e", seed_td, seed_lstm])

    # ---- 3. train: continue_from seed, REMAPPED to the clean CU charset ----
    print("\n[3/4] lstmtraining: continue from Cyrillic.lstm, output layer "
          "remapped to the CU unicharset (--old_traineddata)", file=sys.stderr)
    lr = args.learning_rate
    attempt, status = 0, None
    while attempt <= args.retries:
        attempt += 1
        cmd = ["lstmtraining",
               "--traineddata", out / f"{args.model_name}.traineddata",  # proto (CU charset)
               "--old_traineddata", seed_td,
               "--continue_from", seed_lstm,
               "--model_output", ckpt / args.model_name,
               "--train_listfile", out / "list.train",
               "--eval_listfile", out / "list.eval",
               "--learning_rate", f"{lr}",
               "--max_iterations", str(args.max_iterations),
               "--debug_interval", "0"]
        if args.dry_run:
            print("  (dry-run)", " ".join(map(str, cmd)), file=sys.stderr)
            status = "ok"
            break
        print(f"  attempt {attempt} (learning_rate={lr})", file=sys.stderr)
        status = launch_and_watch(cmd, log_path, args.probation, args.ceiling)
        if status != "collapse":
            break
        # collapse: clear checkpoints so the retry restarts from the seed
        for f in ckpt.glob("*"):
            f.unlink()
        lr *= 2
        print(f"  retrying with learning_rate={lr}", file=sys.stderr)

    if status == "collapse":
        sys.exit("ERROR: collapsed on every attempt. With a seeded start this "
                 "is unusual -- check the rendered images (review_samples.py) "
                 "and the .lstmf freshness before anything else.")
    if status == "error":
        sys.exit(f"ERROR: lstmtraining failed -- see {log_path}")

    # ---- 4. finalize best checkpoint -> traineddata ------------------------
    print("\n[4/4] packaging best checkpoint", file=sys.stderr)
    final = data_dir / f"{args.model_name}.traineddata"
    if not args.dry_run:
        best = ckpt / f"{args.model_name}_checkpoint"
        if not best.exists():
            cands = sorted(ckpt.glob(f"{args.model_name}*checkpoint*"))
            if not cands:
                sys.exit("ERROR: no checkpoint found to package.")
            best = cands[-1]
        run(["lstmtraining", "--stop_training",
             "--continue_from", best,
             "--traineddata", out / f"{args.model_name}.traineddata",
             "--model_output", final])
        print(f"\nDone: {final}", file=sys.stderr)
        print("Sanity: decode one training line --", file=sys.stderr)
        print(f"  f=$(find {gt_dir} -name '*.png' | head -1); "
              f"tesseract $f stdout --psm 13 -l {args.model_name} "
              f"--tessdata-dir {data_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
