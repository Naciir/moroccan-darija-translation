# Every target runs from the repository root (data/, checkpoints/, outputs/, results*.jsonl are relative to it).
PY ?= python

.PHONY: install data tokenizers test smoke grid analyze figures docs app

install:
	$(PY) -m pip install -e ".[dev,docs,hub,app]"

data:
	$(PY) scripts/prepare_data.py && $(PY) scripts/build_extra_data.py

tokenizers:
	$(PY) scripts/retrain_tokenizers.py

test:
	$(PY) -m pytest -q

smoke:
	$(PY) -m darija_nmt.train --model vanilla --attn --epochs 1 --limit_train 200 --tag smoke --runs_dir smoke_ckpt --results smoke.jsonl && $(PY) -m darija_nmt.evaluate --runs vanilla_smoke_s0 --runs_dir smoke_ckpt --limit 20 --show 2 --out_dir smoke_out && $(PY) -m darija_nmt.translate "where are you?" --run vanilla_smoke_s0 --runs_dir smoke_ckpt

grid:
	$(PY) scripts/run_experiments.py --jobs 3

analyze:
	$(PY) scripts/analyze.py && $(PY) scripts/extras.py

figures:
	$(PY) scripts/figures.py

docs:
	$(PY) scripts/build_docs.py docs

app:
	$(PY) app/app.py
