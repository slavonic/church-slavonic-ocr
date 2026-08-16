# Sources

The book each line crop in `eval/` and `finetune/` was extracted from, keyed by
the filename stem (`<stem>_p<pg>_l<ln>.png`/`.gt.txt` — see `docs/evaluation.md`
for how these are produced).

| stem | citation | finetune | eval |
|---|---|---|---|
| `Trebnik_CSL_SPC` | Требник. — Београд, 2008. | 495 | 51 |
| `menaion1` | Минея дополнительная: в 2-х ч. Часть 1: Сентябрь-февраль. — М.: Издательство Московской Патриархии, 2018. | 473 | 50 |
| `newmartyrs` | Служба святым новомучеником и исповедником российским. — Джорданвилль: Типография прп. Иова Почаевского, 1983. | 94 | 9 |
| `triodion2002` | Триодь цветная. — М.: Издательский совет РПЦ, 2002. | 126 | 22 |
| `xenia` | Служба святей, блаженней во Христе, Ксении, бездомней страннице Петрова града. — Jordanville: Holy Trinity Monastery, б.д. | 46 | 5 |
| **total** | | **1234** | **137** |

Counts are `.gt.txt` files per stem. `make review-staging` refreshes this
table automatically after each review session; run it standalone with:

```bash
python3 scripts/update_source_counts.py
```

New stems found on disk get a `[TODO: add citation]` placeholder row —
existing citations are preserved, only the counts are recomputed.
