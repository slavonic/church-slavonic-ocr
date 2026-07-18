# Church Slavonic OCR

A training pipeline and a distributable [Tesseract](https://github.com/tesseract-ocr/tesseract)
LSTM model for **Church Slavonic** printed liturgical texts — the accented,
titlo-bearing, superscript-lettered orthography that general Cyrillic OCR models
cannot read.

The model is trained on synthetic line images rendered from a large Church
Slavonic text corpus in several historically-grounded typefaces, then adapted to
real scans with a small hand-corrected fine-tuning set.

## What's here

| Path | Contents |
|------|----------|
| `scripts/` | the Python tooling (data generation, extraction, evaluation, monitoring, review) |
| `corpus/` | the source text corpus — git submodule of [`slavonic/cu-md-sandbox`](https://github.com/slavonic/cu-md-sandbox) |
| `fonts/`  | the rendering typefaces — git submodules (Ponomar, Triodion, Pochaevsk, Acathist, Monomakh) |
| `data/cu-ground-truth/` | the **synthetic** training pairs — *generated*, not committed (regenerate with `make dataset`) |
| `data/real-lines/` | **hand-corrected** scan lines — committed; `eval/` (held out) and `finetune/` |
| `model/` | the [model card](model/MODEL_CARD.md) and evaluation reports (the `cu.traineddata` binary itself ships via [Releases](../../releases), not in the repo) |
| `training/` | tesstrain working state (checkpoints, unicharset) — not committed |
| `docs/` | the full workflow, per-stage guides, and troubleshooting |

## Just want to OCR?

No clone needed — download `cu.traineddata` from the [**Releases**](../../releases)
page, then:

```bash
tesseract your_line.png stdout --psm 13 -l cu --tessdata-dir /path/to/download-dir
# (or copy it into your tessdata/ dir to use -l cu directly — see model/README.md)
```

## Quickstart (reproduce / retrain)

```bash
# 1. clone with submodules (corpus + fonts) and install python deps
git clone --recurse-submodules <this-repo> && cd church-slavonic-ocr
python3 -m venv .venv && source .venv/bin/activate
make setup         # git submodule update --init --recursive + pip install -r requirements.txt

# 2. see docs/pipeline.md for the full workflow
make dataset       # regenerate synthetic ground truth
make train         # from-scratch tesstrain run (needs a tesstrain checkout)
make eval          # score against data/real-lines/eval
```

## Results

Download the trained model from the [**Releases**](../../releases) page. See the
[model card](model/MODEL_CARD.md) for current metrics and
[`model/eval/`](model/eval) for the per-line report. Character error rate is
reported two ways — full, and with combining marks stripped — so letterform
accuracy is separable from diacritic placement (see `docs/evaluation.md`).

## Documentation

- [`docs/pipeline.md`](docs/pipeline.md) — the end-to-end workflow, corpus → model
- [`docs/data-generation.md`](docs/data-generation.md) — `cu_make_training_data.py`: fonts, charset, hyphenation, degradation
- [`docs/training.md`](docs/training.md) — tesstrain setup, from-scratch vs fine-tune, resuming, parallelism
- [`docs/evaluation.md`](docs/evaluation.md) — extracting real lines and measuring CER/WER
- [`docs/troubleshooting.md`](docs/troubleshooting.md) — the failure modes we actually hit and how to read them

## License & attribution

This repository combines works under different licenses; please respect each:

- **Code** (`scripts/`) — see [`LICENSE`](LICENSE) (MIT License).
- **Corpus** (`corpus/`, submodule) — retains the license of `slavonic/cu-md-sandbox` (MIT License).
- **Fonts** (`fonts/`, submodules) — each retains its own license (SIL Open Font License).
- **Model** (`model/cu.traineddata`) — a derivative work of the corpus and fonts;
  its provenance and intended use are documented in the [model card](model/MODEL_CARD.md).
