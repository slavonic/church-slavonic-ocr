# Sources

The book each line crop in `eval/` and `finetune/` was extracted from, keyed by
the filename stem (`<stem>_p<pg>_l<ln>.png`/`.gt.txt` — see `docs/evaluation.md`
for how these are produced).

| stem | citation | finetune | eval |
|---|---|---|---|
| `Trebnik_CSL_SPC` | Требник. — Београд, 2008. | 340 | 36 |
| `menaion1` | Минея дополнительная: в 2-х ч. Часть 1: Сентябрь-февраль. — М.: Издательство Московской Патриархии, 2018. | 100 | 14 |
| `triodion2002` | Триодь цветная. — М.: Издательский совет РПЦ, 2002. | 126 | 22 |
| **total** | | **566** | **72** |

Counts are `.gt.txt` files per stem. `make review-staging` refreshes this
table automatically after each review session; run it standalone with:

```bash
python3 scripts/update_source_counts.py
```

New stems found on disk get a `[TODO: add citation]` placeholder row —
existing citations are preserved, only the counts are recomputed.
