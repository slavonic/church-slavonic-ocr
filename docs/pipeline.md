# Pipeline

The whole path from corpus to a model that reads real scans, and where each
script fits.

```
corpus/ (text) + fonts/ ──▶ cu_make_training_data.py ──▶ data/cu-ground-truth/ (synthetic pairs)
                                                                    │
                                                                    ▼
                                                        tesstrain  ──▶ model/cu.traineddata
                                                                    │
scanned books (PDF/DJVU) ──▶ extract_lines.py ──▶ line crops + OCR guess
                                                     │ (you correct them)
                                                     ▼
                                        data/real-lines/{eval,finetune}/
                                                     │
                              eval ──▶ cu_eval.py ──▶ CER/WER + report.html
                              adapt ─▶ fine-tune from cu on finetune/ ──▶ better model
```

## Stages

1. **Generate synthetic data** — `make dataset` renders the cleaned corpus text
   in five faces with degradation and hyphenation. Details: `docs/data-generation.md`.

2. **Train** — from scratch with tesstrain, reading `data/cu-ground-truth/`.
   Details and the exact commands: `docs/training.md`.

3. **Get real lines** — `extract_lines.py` turns book pages (PDF/DJVU) into line
   crops with a first-pass transcription you then correct. Details: `docs/evaluation.md`.

4. **Evaluate** — `cu_eval.py` scores the model on the held-out real lines, full
   and mark-stripped, with a per-line report. Details: `docs/evaluation.md`.

5. **Adapt** — fine-tune the model on the corrected real lines to close the gap
   between clean synthetic training data and real scans. This is the step that
   moves real-world CER the most once the pipeline is otherwise clean.

## Iterating

The loop is: eval → read the failure modes → fix the highest-leverage one →
retrain or fine-tune → eval again. `docs/troubleshooting.md` maps common symptoms
to the right fix so you change one thing at a time.

## Monitoring

During a training run, watch progress live:
```bash
make train 2>&1 | tee training.log &
python3 scripts/watch_training.py training.log        # or --headless --png live.png
```
