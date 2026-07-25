#!/usr/bin/env python3
"""update_source_counts.py -- keep data/real-lines/README.md's line-count
table in sync with what's actually in finetune/ and eval/.

Parses the existing table to preserve each stem's citation, recomputes the
finetune/eval counts (and total) by counting .gt.txt files per stem in the
real directories, and adds a placeholder row for any stem present on disk
but missing from the table. Run via `make review-staging` (after the
interactive session ends) so the table never drifts from reality.
"""

import re
import sys
from pathlib import Path

README = Path("data/real-lines/README.md")
FINETUNE = Path("data/real-lines/finetune")
EVAL = Path("data/real-lines/eval")

ROW_RE = re.compile(
    r"^\|\s*`([^`]+)`\s*\|\s*(.*?)\s*\|\s*\*{0,2}\d+\*{0,2}\s*\|\s*\*{0,2}\d+\*{0,2}\s*\|\s*$")


def stems_with_counts():
    """{stem: (finetune_count, eval_count)} by stripping _p####_l###.gt.txt."""
    counts = {}
    stem_re = re.compile(r"^(.*)_p\d+_l\d+\.gt\.txt$")
    for d, idx in ((FINETUNE, 0), (EVAL, 1)):
        for f in d.glob("*.gt.txt"):
            m = stem_re.match(f.name)
            if not m:
                continue
            stem = m.group(1)
            fc, ec = counts.get(stem, (0, 0))
            counts[stem] = (fc + 1, ec + 1) if idx == 1 else (fc + 1, ec)
    return counts


def main():
    if not README.exists():
        sys.exit(f"ERROR: {README} not found")

    counts = stems_with_counts()
    lines = README.read_text(encoding="utf-8").splitlines()

    citations = {}      # stem -> citation text, from the existing table
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("| stem "):
            header_idx = i
            continue
        m = ROW_RE.match(line)
        if m and m.group(1) != "total":
            citations[m.group(1)] = m.group(2)

    if header_idx is None:
        sys.exit(f"ERROR: no '| stem | citation | ...' table found in {README}")

    for stem in counts:
        citations.setdefault(stem, "[TODO: add citation]")

    new_rows = ["| stem | citation | finetune | eval |", "|---|---|---|---|"]
    total_ft = total_ev = 0
    for stem in sorted(citations):
        ft, ev = counts.get(stem, (0, 0))
        total_ft += ft
        total_ev += ev
        new_rows.append(f"| `{stem}` | {citations[stem]} | {ft} | {ev} |")
    new_rows.append(f"| **total** | | **{total_ft}** | **{total_ev}** |")

    # replace the old table block: header line through the line before the
    # next blank line (or end of file)
    end = header_idx + 1
    while end < len(lines) and lines[end].strip():
        end += 1
    lines[header_idx:end] = new_rows

    README.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  {README}: {len(citations)} stems, "
          f"finetune={total_ft} eval={total_ev}", file=sys.stderr)


if __name__ == "__main__":
    main()
