# data/

Only the SentencePiece tokenizers are versioned here. The parallel data is **not redistributed**: rebuild it from
Hugging Face with the scripts below (run from the repository root).

| path | what | rebuilt by |
|---|---|---|
| `spm_en.model`, `spm_dar.model` (+ `.vocab`) | BPE 3,000, `split_by_number=False`; EN lowercased (`nmt_nfkc_cf`), DAR `identity` (capitals are emphatic consonants in Arabizi). Used by every main run. | `python scripts/retrain_tokenizers.py` |
| `tok_orig/` | the original tokenizers (`split_by_number=True`, EN `identity`): Arabizi words are cut at their digits (`3la` -> `▁3` `la`). Kept for the ablation and the legacy checkpoints. | `python scripts/retrain_tokenizers.py --out_dir data/tok_orig --split_by_number 1 --en_norm identity` |
| `tok_plus3k/`, `tok_plus6k/`, `tok_plus10k/` | BPE 3k / 6k / 10k trained on `train_plus.csv` (tokenizer study on the 42k corpus) | `python scripts/retrain_tokenizers.py --train_csv data/train_plus.csv --vocab 6000 --out_dir data/tok_plus6k` (etc.) |

Retraining reproduces the same vocabularies and scores (the `.model` bytes differ only by the training paths stored
inside them).

## Rebuilding the splits

```bash
python scripts/prepare_data.py          # -> data/train.csv, val.csv, test.csv, train.en.txt, train.dar.txt
python scripts/build_extra_data.py      # -> data/extra/doda.parquet, data/train_plus.csv  (data-scaling runs)
python scripts/fetch_mono_darija.py     # -> data/mono_dar.txt  (optional: tok_mono ablation, back-translation)
python scripts/retrain_tokenizers.py --out_dir data/tok_mono --dar_extra data/mono_dar.txt   # optional
```

`prepare_data.py` is the download / cleaning / grouped-split logic of `notebooks/darija_data_prep.ipynb`
(HF `imomayiz/darija-english`, config `sentences`): 87,785 rows -> 12,650 clean Latin-script pairs -> a split grouped on
the lowercased English sentence (seed 42) into 10,131 / 1,251 / 1,268 train / val / test pairs. It then checks the
sha256 of every written file against the splits all reported numbers were computed on, and exits with an error if
any file differs (e.g. if the upstream dataset changes). Lines end in CRLF on every OS so the bytes match.

`build_extra_data.py` adds pairs from HF `atlasia/darija_english` (config `doda`) after removing every pair whose
English or Darija side matches or nearly matches a val/test sentence; `val.csv` / `test.csv` never change.

## Licences

- `imomayiz/darija-english`: the dataset card states a "cc" licence without a version.
- `atlasia/darija_english` (DODa): CC BY-NC 4.0 (non-commercial).

Check the dataset cards before reusing the data. The tokenizers here are models trained on these corpora.

## Test fixtures

`tests/fixtures/mini_{train,val,test}.csv` are the first 40 rows of each split (a tiny excerpt, used only so that the
test suite and the CI smoke run work without downloading the data). See `tests/conftest.py`.
