# Bug hunt: what was wrong in the first version, and how each fix was proven

The first version of this project (the course submission) trained three LSTM cells for 20 epochs with one seed and
reported BLEU ≈ 1.3. Before comparing cells, every part of the pipeline was audited. Each finding below comes with
the measurement that proves it and the measured effect of the fix (test chrF, 3 seeds unless stated otherwise).
The audit and the regression tests that lock each fix in place are in [`tests/test_fixes.py`](../tests/test_fixes.py).

## 1. The tokenizer cut every Arabizi word at its digits (fixed: +0.9 chrF, p = 0.001)

In Darija written in Latin script, digits are letters: 3 = ع, 7 = ح, 9 = ق, 2 = ء, 5 = خ, 8 = ه. SentencePiece's
default `split_by_number=True` treats them as numbers and always splits there:

| word | meaning | original tokenizer | fixed tokenizer |
|---|---|---|---|
| `3la` | on, about | `▁3` `la` | `▁3la` |
| `ch7al` | how much | `▁ch` `7` `al` | `▁ch7al` |
| `t7kem` | (verb) | `▁t` `7` `k` `em` | `▁t7` `k` `em` |

- **Evidence.** 29.5% of the Darija words in the test references contain a digit-letter. Not one of the 3,000 Darija
  BPE pieces combined a digit and a letter, 14.7% of all test pieces were bare digits, and no digit-word was ever a
  single piece.
- **Fix.** `split_by_number=False`, same vocabulary size: 41% of digit-words become one piece, bare digits fall to
  2.8%, sequences are 12% shorter. English is lowercased (`nmt_nfkc_cf`); Darija is *not*, because capital S/D/T are
  different letters (ص/ض/ط).
- **Effect.** 18.70 → 19.60 chrF (+0.90, paired bootstrap p = 0.001). The models trained on the old tokenizer also
  over-produced digit-words (≈50% of output words vs 29.5% in the references); after the fix ≈25%.
- **Check.** `retrain_tokenizers.py --en_norm identity --split_by_number 1` rebuilds the original vocabulary byte for
  byte, so the comparison isolates exactly this setting.

## 2. Padding ran through the encoder (fixed: no measurable effect — a negative result)

The batches were right-padded and the source was then reversed, so every short sentence started with up to ~30 pad
steps that went through the LSTM without a mask; 65% of all encoder steps in training were pads. At test time
sentences were translated one by one, without pads: a train/test mismatch.

- **Fix.** Masked recurrence: on a pad position the cell keeps its previous state. A test checks that a padded batch
  gives exactly the same outputs as each sentence encoded alone.
- **Effect.** 18.70 → 18.67 chrF (p = 0.82): none. The pad embedding is zero, and the network had learned to absorb
  the leading pads. The fix is kept because it is correct and because it makes length-bucketed batches possible,
  which train ≈2.5× faster.

## 3. Undertrained, single seed, no baseline

- All three original models had their best validation loss at their **last** epoch (20/20). With early stopping
  (patience 5, ≤ 40 epochs) the same setup reaches 18.70 instead of 17.10 chrF.
- The original ranking of the cells rested on validation-loss gaps of 0.01–0.03 from one seed. Over 3 seeds the
  standard deviation of test chrF is 0.3–0.5, as large as the gaps: every comparison here uses 3 seeds and a paired
  bootstrap on corpus chrF.
- A non-neural **retrieval baseline** (copy the Darija side of the training pair whose English words overlap most
  with the input) scores 19.01 chrF — above all three original models. A model that does not beat it has not learned
  to translate.

## 4. Validation loss depended on how the batches were formed

The original validation loss was the mean of per-batch means, so the same model scored 0.14 nats differently with
bucketed and with random batches. It is now token-weighted (sum over all target tokens / their count).

## 5. Dot-product attention saturates with LSTMP

Adding Luong dot attention gives +3.3 chrF to the vanilla LSTM but makes LSTMP *worse* (19.11 → 17.40). LSTMP's
output h = W_p m is a linear projection, not squashed by a tanh, so its vectors are larger (mean norm 8.5 vs 6.4)
and the dot-product scores saturate the softmax: the decoder puts 95% of its attention on a single source position
(normalised entropy 0.07 vs 0.40 for the vanilla cell). Scaling the scores by 1/√d over-corrects (entropy 0.81:
almost uniform). Score functions with a learned map (`general`, `additive`) are compared in the results.
Reproduce with `python scripts/attention_stats.py`.

## 6. Beam search made outputs 30–40% shorter

With the standard GNMT length normalisation (α = 0.6), beam-5 outputs are 30–40% shorter than the references (some
are empty: the end-of-sentence token at step 1 beats every normalised longer hypothesis) and chrF drops. The beam
search itself is correct — it never scores below greedy decoding on its own objective (tested) — so greedy decoding
is the main setting and α is tuned on the validation set.

## 7. The split leaks a little

The split was grouped by the lowercased English sentence. 3.9% of test English sentences still equal a training
sentence once punctuation is ignored (`anyway` / `anyway.`). This is kept as is (changing the split would invalidate
the comparison with the original), affects every system equally and mostly helps the retrieval baseline. The extra
training data used for the data-scaling experiments is filtered much more strictly (exact and near duplicates of any
validation/test sentence removed).

## 8. Speed

Each cell originally computed its 8 gate projections with 8 small `nn.Linear` calls. They are fused into two
(`W_x`, `W_h`) and split back into the named gates; a test proves the fused cell equals the original one and has the
same parameter count. Together with length bucketing, an epoch went from ~57 s to ~5 s.
