# Continue the project on your machine

This folder is the **complete working state** of the project as of 2026-10-26: the code (an installable package),
every dataset file, all 89 trained checkpoints, every result record, the experiment logs and the scheduler that
runs what is left. Nothing has to be retrained: the scripts skip every finished run and only do the missing ones.

## 1. Setup (once)

Python 3.11 or newer, ideally an NVIDIA GPU (the study used a laptop RTX 3060, 6 GB).

```bash
cd darija-nmt
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux / macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126      # GPU build (or .../whl/cpu for CPU only)
pip install -e ".[dev,docs,app]"
python scripts/restore_mtimes.py      # ONCE, right after unzipping (see below)
```

`restore_mtimes.py` puts back the original modification times of the checkpoints. Each evaluation record stores the
time of the checkpoint it scored, so that a retrained model is re-evaluated; some unzip tools reset every file time,
which would make all ~100 finished evaluations look stale and redo them. It deletes its manifest after use.

Check that everything works (about 1-2 minutes):

```bash
pytest -q                                                    # expect: 31 passed
python -m darija_nmt.translate "where are you going?" --run vanilla_c42_s0     # -> fin ghatmchi?
```

## 2. Where the study stands

Done (test chrF, greedy decoding; ± = std over 3 seeds; full tables in `RESULTS.md`):

| | Vanilla | CIFG | LSTMP | WMC |
|---|---|---|---|---|
| 10k pairs, no attention | 19.35 ± 0.34 | 18.56 ± 0.34 | 19.11 ± 0.28 | 19.48 ± 0.28 |
| 10k pairs, dot attention | 22.64 ± 0.22 | 22.06 ± 0.24 | 17.40 ± 0.12 | 21.92 ± 0.23 |
| 10k pairs, additive attention | 23.18 ± 0.34 | 23.30 ± 0.22 | 20.86 ± 0.38 | 23.07 ± 0.20 |
| 42k pairs, additive attention (1 seed so far) | **26.31** | 25.71 | 23.00 | not run yet |

Also done: pipeline ablation, hyperparameter study (12 settings), scaled/general attention, attention saturation
statistics, tokenizer study on the 42k corpus (6k BPE chosen on validation), Transformer seed 0 on 10k (18.21),
the Darija→English model (chrF 26.57) and the back-translated corpus `data/train_bt.csv` (41,702 real + 42,310
synthetic pairs). Reference points: original model 17.10, retrieval baseline 19.01.

Still to run (≈ 20-25 GPU-hours on a laptop RTX 3060):

1. `c42`: the 42k cell comparison — WMC seed 0, then seeds 1 and 2 for all four cells (9 runs).
2. One-change ablations of the best system (vanilla, 42k): bidirectional encoder, label smoothing 0.1,
   BPE-dropout 0.1, learning rate 3e-3, batch 32 (5 runs), and the back-translation run `bt` (1 longer run).
3. `n42`: the four cells without attention on 42k (4 runs); the Transformer seeds 1-2 (10k) and `trf42` (42k).
4. Then automatically: the final recipe = base + every change that beat it by ≥ 0.3 validation chrF → `fin`
   runs (vanilla, CIFG × 3 seeds, then WMC, LSTMP), beam-search alpha tuned on validation, and all tables/figures.

## 3. Run it

```bash
python scripts/overnight.py                       # 2 runs in parallel; restartable at any time
NIGHT_JOBS=3 python scripts/overnight.py          # 3 in parallel if you have >= 16 GB RAM free and a pagefile/swap
NIGHT_HOURS=8 python scripts/overnight.py         # stop starting new runs after 8 hours
```

- Progress: `night.log` (one line per started / finished job and every decision), per-run logs in `logs/`.
- It is **resumable**: stop it (Ctrl+C, or kill the python processes) and start the same command again; finished
  runs and evaluations are skipped. The tokenizer decision is already stored in `night_decisions.json`.
- Keep the PC plugged in and awake. On the last machine some epochs stalled for hours with no sleep event; the PC
  had **no pagefile**, so memory ran out with 3 jobs. With a normal pagefile/swap, or `NIGHT_JOBS=2`, it is fine.
- Single runs by hand: `python scripts/run_experiments.py --only c42 --seeds 1 --cells cifg --jobs 1`
  (tags created at run time are in `grid_extra.json`).

## 4. When the runs are done

```bash
python scripts/analyze.py && python scripts/extras.py && python scripts/figures.py   # RESULTS.md, EXTRAS.md, figures/
python scripts/attention_stats.py                                                     # attention_stats.json
python scripts/human_eval.py make --systems retrieval=outputs/baseline_retrieval_test.txt \
    lstm=outputs/vanilla_attn_additive_s0_test_beam1.txt best=outputs/vanilla_c42_s0_test_beam1.txt
```

`human_eval.py` makes a blind rating sheet (`human_eval/sheet.csv`) for a Darija speaker: 50 test sentences,
adequacy and fluency 1-5. Fill it in, then `python scripts/human_eval.py score`.

Then update the numbers in `README.md` (the 42k row and the final-recipe section) and publish:

```bash
git init && git add . && git commit -m "English-Darija translation with hand-written LSTM cells"
gh repo create darija-nmt-lstm-cells --public --source . --push      # or create the repo on github.com and push
```

`.gitignore` keeps the data files, checkpoints and logs out of git. **Do not commit the data**: the extra corpus
(`atlasia/darija_english`) is CC BY-NC 4.0 and the main corpus licence is unspecified; the repository ships the
scripts that rebuild them (`scripts/prepare_data.py`, `scripts/build_extra_data.py`). To share trained models,
upload a few checkpoints to the Hugging Face Hub (with `huggingface_hub`) and point the demo at them:
`DARIJA_NMT_HUB_REPO=<user>/<repo> python app/app.py` (Gradio; also deployable as an HF Space with
`app/requirements.txt`).

## 5. Map of the state files

| file | content |
|---|---|
| `results.jsonl` | one record per training run (config, per-epoch losses, best epoch, minutes) |
| `results_test.jsonl` / `results_val.jsonl` | test / validation scores per run and decoding setting |
| `night_decisions.json` | automatic decisions (tokenizer; the final recipe once ablations exist) |
| `grid_extra.json` | experiment tags created by the scheduler (c42, n42, abl_*, bt, trf42, fin) |
| `attention_stats.json` | attention sharpness per attention run |
| `checkpoints/` | `<run>.pt` weights + `<run>.json` config for every run |
| `outputs/` | test translations of every run (used by the significance tests) |
| `logs/` | training logs; `logs/night_2026-10-26.log` is the previous scheduler log |
| `docs/` | `bug_hunt.md` (what was wrong in the first version), `data_card.md`, the course report + slides |
