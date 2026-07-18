# Convenience wrapper. Real work lives in scripts/ and (for training) in tesstrain.
# Override on the command line, e.g.  make train MAX_ITERATIONS=120000

MODEL_NAME     ?= cu
LIMIT          ?= 40000
MAX_ITERATIONS ?= 100000
TESSTRAIN      ?= ../tesstrain        # path to a tesseract-ocr/tesstrain checkout
JOBS           ?= $(shell nproc)

.PHONY: setup dataset train eval review review-staging release clean-train

setup:            ## fetch corpus + font submodules and install python deps
	git submodule update --init --recursive
	pip install -r requirements.txt

dataset:          ## (re)generate the synthetic ground truth into data/cu-ground-truth
	LIMIT=$(LIMIT) scripts/build_dataset.sh

train:            ## run tesstrain from scratch, reading data/, writing training/
	$(MAKE) -C $(TESSTRAIN) training -j$(JOBS) \
	  MODEL_NAME=$(MODEL_NAME) MAX_ITERATIONS=$(MAX_ITERATIONS) \
	  DATA_DIR=$(CURDIR)/training \
	  GROUND_TRUTH_DIR=$(CURDIR)/data/cu-ground-truth
	cp training/$(MODEL_NAME).traineddata model/

eval:             ## score model/ against the held-out real lines
	python3 scripts/cu_eval.py data/real-lines/eval --model $(MODEL_NAME) \
	  --tessdata-dir model --report model/eval/report.html --tsv model/eval/metrics.tsv

review:           ## montage a few hyphenated training samples
	python3 scripts/review_samples.py data/cu-ground-truth --n 8 --out model/eval/review.png

review-staging:   ## web UI to correct/delete extract_lines.py's staging output
	python3 scripts/review_staging.py --dir data/real-lines/staging

release:          ## publish model/cu.traineddata as a GitHub Release asset — needs gh CLI, e.g. make release VERSION=v1.0
	@test -n "$(VERSION)" || { echo "set VERSION=vX.Y (e.g. make release VERSION=v1.0)"; exit 1; }
	@test -f model/$(MODEL_NAME).traineddata || { echo "model/$(MODEL_NAME).traineddata not found — run 'make train' first"; exit 1; }
	gh release create $(VERSION) model/$(MODEL_NAME).traineddata \
	  --title "$(MODEL_NAME) $(VERSION)" \
	  --notes "Church Slavonic Tesseract model. Install per model/README.md; details in model/MODEL_CARD.md."

clean-train:      ## wipe tesstrain scratch (keeps ground truth + model)
	rm -rf training/* && touch training/.gitkeep
