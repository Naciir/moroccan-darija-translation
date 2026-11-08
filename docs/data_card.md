# Data card

## Main corpus: `imomayiz/darija-english` (Hugging Face), "sentences" subset

| | |
|---|---|
| Source | [huggingface.co/datasets/imomayiz/darija-english](https://huggingface.co/datasets/imomayiz/darija-english) |
| Content | Moroccan Darija sentences written in Latin script (Arabizi); 87,785 rows, 12,650 with an English translation |
| Licence | listed as "cc" on the dataset card (no version given). The CSV splits are **not redistributed** in this repository; `scripts/prepare_data.py` rebuilds them from the source |
| Cleaning | drop rows without English (87,785 → 12,743), duplicate rows (→ 12,669) and identical EN/DAR pairs (→ 12,650) |
| Split | 80 / 10 / 10, **grouped by the lowercased English sentence** (a sentence and its translations never straddle two splits): train 10,131 / validation 1,251 / test 1,268 pairs |
| Reproducibility | `prepare_data.py` checks the sha256 of every file it writes against the hashes of the files used for all experiments; a mismatch stops the script |

**Properties that matter for modelling**

- Short sentences: 5.9 English and 4.4 Darija words on average (95th percentile 13 / 10).
- No standard spelling: *dyal / dial*, *ch7al / chhal*, *o / ou / w*. 72% of Darija word types in the training
  split occur once, and 26% of the Darija words in the test references never occur in training.
- Digits are letters: 29.5% of Darija words contain one of 3 = ع, 7 = ح, 9 = ق, 2 = ء, 5 = خ, 8 = ه.
  Capital S/D/T are different letters (ص/ض/ط), so the Darija side is never lowercased.
- One reference per sentence, crowd-sourced.

**Known issue.** 3.9% of the test English sentences (49) equal a training sentence once punctuation is removed
(`anyway` / `anyway.`), because the grouping used lowercasing only. It is kept to stay comparable with the original
study; it affects every system in the same way and mostly helps the retrieval baseline.

## Extra parallel data (data-scaling experiments only): `atlasia/darija_english`, config `doda`

| | |
|---|---|
| Source | [huggingface.co/datasets/atlasia/darija_english](https://huggingface.co/datasets/atlasia/darija_english) (Darija Open Dataset contributions, curated by AtlasIA) |
| Licence | **CC BY-NC 4.0** — non-commercial. Not redistributed here; `scripts/build_extra_data.py` downloads and filters it |
| Content | 45,103 Latin-script English/Darija pairs, a superset of the main corpus (99.4% of its pairs appear in it) |
| Filters | every row whose English **or** Darija side equals (lowercased, punctuation removed) **or** nearly equals (Jaccard ≥ 0.8 on English words or on Darija character 5-grams) any validation/test sentence is removed; mojibake / Arabic-script rows, sides longer than 40 words, duplicate pairs and pairs already in the training split are removed |
| Result | 31,571 new pairs → `train_plus.csv` = 41,702 training pairs; validation and test sets unchanged |
| Leakage control | the retrieval baseline over the 41,702 pairs scores **lower** on test than over the original 10,131 (chrF 18.46 vs 19.01): the extra data contains no copies of test sentences |

## Monolingual Darija (back-translation)

75,020 Darija-only sentences from the same corpus (`scripts/fetch_mono_darija.py`), with every validation/test
sentence and near-duplicate removed. A Darija → English model trained on `train_plus.csv` translates them to
create synthetic pairs (`scripts/backtranslate.py`), which are filtered again against validation/test.

## Tokenizers (shipped in `data/`)

SentencePiece BPE, one model per language, trained on the training split only:
`data/spm_{en,dar}.model` (3,000 pieces, `split_by_number=False`, English lowercased), `data/tok_orig/` (the
original, buggy tokenizers, kept for the ablation), `data/tok_plus{3,6,10}k/` (trained on `train_plus.csv`).
