# Results

Test set = data/test.csv (1268 sentences). Scores are sacrebleu 2.6 corpus scores; mean ± std over seeds.
Main numbers are greedy decoding; beam-5 uses length normalisation alpha=0.6.
*val loss is teacher-forced cross-entropy averaged over all non-pad target BPE tokens of the val set (token-weighted,
so it does not depend on how val batches are formed). Only comparable between runs that use the SAME tokenizer; for
--no_mask runs it also reflects the pads in their (fixed-order, unbucketed) val batches, which is part of what the mask
fixes. The shipped val losses in table 1 use the ORIGINAL definition (mean of per-batch means, original tokenizer,
unmasked pads) and are not comparable with any other val loss here.
Beam-5 (GNMT alpha=0.6) shortens outputs sharply on these models (len ratio = BLEU sys_len / ref_len; some beam outputs
are empty), so a beam-5 drop vs greedy is a brevity effect of the objective, not a search bug and not a decoding
improvement. Beam-5 is a secondary check only (evaluate.py --alpha 1.0 penalises short outputs less).

## 1. Shipped numbers (original code, single seed)
| run | params | best epoch | val loss (orig. def.: per-batch mean; NOT comparable) | test chrF | test BLEU | seeds |
|---|---|---|---|---|---|---|
| vanilla / shipped | 2,327,480 | 20 | 4.260 | 17.10 | 1.27 | 1 |
| cifg / shipped | 2,130,360 | 20 | 4.269 | 17.13 | 1.11 | 1 |
| lstmp / shipped | 1,747,128 | 20 | 4.289 | 17.45 | 1.54 | 1 |

## 2. Ablation (vanilla cell): what each fix contributes
old/tok/mask use random batches like the original (~65% leading pads for unmasked runs); main adds length bucketing,
which with the mask on only changes batch composition. old->mask = mask effect, old->tok = tokenizer effect.
| run (model / tag) | params | best epoch | val loss* | test chrF | test chrF++ | test BLEU | seeds |
|---|---|---|---|---|---|---|---|
| old: orig tokenizer, no mask, random batches (= original setup) | 2,327,480 | 30.0 ± 2.0 | 4.228 ± 0.010 | 18.70 ± 0.32 | 17.00 ± 0.29 | 1.62 ± 0.18 | 3 |
| + new tokenizer (no mask, random batches) | 2,327,480 | 30.7 ± 2.5 | 4.692 ± 0.008 | 19.60 ± 0.46 | 17.99 ± 0.45 | 2.11 ± 0.35 | 3 |
| + mask (orig tokenizer, random batches) | 2,327,480 | 33.3 ± 5.0 | 4.215 ± 0.008 | 18.67 ± 0.07 | 16.97 ± 0.06 | 1.59 ± 0.23 | 3 |
| full fix: new tokenizer + mask + length bucketing | 2,327,480 | 34.3 ± 4.2 | 4.743 ± 0.009 | 19.35 ± 0.34 | 17.74 ± 0.33 | 1.98 ± 0.15 | 3 |

## 3. Main table: cell comparison (new tokenizer + masked encoder, no attention)
| run (model / tag) | params | best epoch | val loss* | test chrF | test chrF++ | test BLEU | chrF beam5 | BLEU beam5 | len ratio greedy / beam5 | seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| vanilla / main | 2,327,480 | 34.3 ± 4.2 | 4.743 ± 0.009 | 19.35 ± 0.34 | 17.74 ± 0.33 | 1.98 ± 0.15 | 16.01 ± 0.43 | 2.20 ± 0.40 | 1.01 ± 0.01 / 0.62 ± 0.01 | 3 |
| cifg / main | 2,130,360 | 36.0 ± 3.0 | 4.750 ± 0.008 | 18.56 ± 0.34 | 17.04 ± 0.32 | 1.70 ± 0.17 | 15.82 ± 0.20 | 1.66 ± 0.31 | 0.97 ± 0.03 / 0.66 ± 0.01 | 3 |
| lstmp / main | 1,747,128 | 35.7 ± 1.5 | 4.796 ± 0.003 | 19.11 ± 0.28 | 17.47 ± 0.27 | 1.77 ± 0.02 | 15.17 ± 0.13 | 1.91 ± 0.25 | 1.02 ± 0.01 / 0.61 ± 0.01 | 3 |

## 4. Controls and baselines
| run (model / tag) | params | best epoch | val loss* | test chrF | test chrF++ | test BLEU | seeds |
|---|---|---|---|---|---|---|---|
| vanilla / nosrc (LM control: no source) | 2,327,480 | 25.7 ± 6.4 | 5.343 ± 0.007 | 7.85 ± 0.87 | 6.75 ± 0.65 | 0.11 ± 0.00 | 3 |
| baseline_retrieval | - | - | - | 19.01 | 16.69 | 1.88 | - |
| baseline_mostfreq | - | - | - | 4.58 | 3.93 | 0.00 | - |

## 5. Secondary experiment: Luong (dot) attention
| run (model / tag) | params | best epoch | val loss* | test chrF | test chrF++ | test BLEU | chrF beam5 | BLEU beam5 | len ratio greedy / beam5 | seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| vanilla / attn | 2,458,552 | 37.0 ± 3.6 | 4.599 ± 0.011 | 22.64 ± 0.22 | 20.89 ± 0.23 | 3.07 ± 0.24 | 18.90 ± 0.18 | 3.55 ± 0.33 | 1.08 ± 0.00 / 0.69 ± 0.02 | 3 |
| cifg / attn | 2,261,432 | 36.7 ± 4.9 | 4.578 ± 0.017 | 22.06 ± 0.24 | 20.37 ± 0.24 | 2.98 ± 0.12 | 19.19 ± 0.48 | 3.65 ± 0.47 | 1.06 ± 0.03 / 0.70 ± 0.01 | 3 |
| lstmp / attn | 1,779,896 | 38.7 ± 0.6 | 4.892 ± 0.008 | 17.40 ± 0.12 | 15.87 ± 0.10 | 1.27 ± 0.08 | 12.72 ± 0.33 | 1.40 ± 0.32 | 1.08 ± 0.02 / 0.59 ± 0.02 | 3 |
| lstmp / attn, scaled dot product (score / sqrt(d)) | 1,779,896 | 39.7 ± 0.6 | 4.750 ± 0.018 | 19.96 ± 0.45 | 18.37 ± 0.45 | 2.20 ± 0.19 | 15.12 ± 0.41 | 2.09 ± 0.26 | 1.07 ± 0.03 / 0.59 ± 0.00 | 3 |
| vanilla / attn, scaled dot product (score / sqrt(d)) | 2,458,552 | 30.3 ± 1.5 | 4.691 ± 0.005 | 20.98 ± 0.19 | 19.33 ± 0.18 | 2.37 ± 0.13 | 17.78 ± 0.17 | 2.88 ± 0.36 | 1.07 ± 0.02 / 0.70 ± 0.00 | 3 |

## 6. Paired bootstrap significance (corpus chrF, greedy, 1000 resamples, seed 12345, pairing seed i with seed i)
| comparison (chrF of B - A) | per-seed delta mean ± std | seeds B>A sig / A>B sig / n.s. | pooled-seeds bootstrap delta (p) | per seed |
|---|---|---|---|---|
| vanilla/main -> cifg/main | -0.80 ± 0.53 | 0 / 2 / 1 | -0.80 (p=0.001) | s0: -0.71 (p=0.004), s1: -0.31 (p=0.177), s2: -1.36 (p=0.001) |
| vanilla/main -> lstmp/main | -0.24 ± 0.12 | 0 / 0 / 3 | -0.24 (p=0.082) | s0: -0.20 (p=0.391), s1: -0.15 (p=0.521), s2: -0.37 (p=0.127) |
| cifg/main -> lstmp/main | 0.56 ± 0.42 | 2 / 0 / 1 | +0.55 (p=0.001) | s0: +0.51 (p=0.032), s1: +0.16 (p=0.509), s2: +0.99 (p=0.001) |
| vanilla/old -> vanilla/main | 0.65 ± 0.29 | 1 / 0 / 2 | +0.65 (p=0.002) | s0: +0.49 (p=0.095), s1: +0.49 (p=0.072), s2: +0.99 (p=0.001) |
| vanilla/attn -> cifg/attn | -0.58 ± 0.28 | 0 / 2 / 1 | -0.58 (p=0.001) | s0: -0.26 (p=0.383), s1: -0.78 (p=0.010), s2: -0.70 (p=0.014) |
| vanilla/attn -> lstmp/attn | -5.24 ± 0.25 | 0 / 3 / 0 | -5.24 (p=0.001) | s0: -5.00 (p=0.001), s1: -5.51 (p=0.001), s2: -5.22 (p=0.001) |
| cifg/attn -> lstmp/attn | -4.66 ± 0.12 | 0 / 3 / 0 | -4.66 (p=0.001) | s0: -4.74 (p=0.001), s1: -4.72 (p=0.001), s2: -4.52 (p=0.001) |
| vanilla/main -> vanilla/attn | 3.29 ± 0.56 | 3 / 0 / 0 | +3.29 (p=0.001) | s0: +2.97 (p=0.001), s1: +3.93 (p=0.001), s2: +2.96 (p=0.001) |
| cifg/main -> cifg/attn | 3.50 ± 0.10 | 3 / 0 / 0 | +3.50 (p=0.001) | s0: +3.42 (p=0.001), s1: +3.46 (p=0.001), s2: +3.62 (p=0.001) |
| lstmp/main -> lstmp/attn | -1.72 ± 0.25 | 0 / 3 / 0 | -1.72 (p=0.001) | s0: -1.83 (p=0.001), s1: -1.42 (p=0.001), s2: -1.89 (p=0.001) |
| vanilla/old -> vanilla/tok | 0.90 ± 0.68 | 2 / 0 / 1 | +0.90 (p=0.001) | s0: +0.21 (p=0.442), s1: +0.93 (p=0.001), s2: +1.57 (p=0.001) |
| vanilla/old -> vanilla/mask | -0.02 ± 0.25 | 0 / 0 / 3 | -0.02 (p=0.823) | s0: -0.31 (p=0.171), s1: +0.16 (p=0.462), s2: +0.08 (p=0.678) |
| lstmp/attn -> lstmp/attn_scaled | 2.56 ± 0.57 | 3 / 0 / 0 | +2.56 (p=0.001) | s0: +1.91 (p=0.001), s1: +2.83 (p=0.001), s2: +2.94 (p=0.001) |
| vanilla/attn -> vanilla/attn_plus | +1.76 | 1 / 0 / 0 | +1.76 (p=0.001) | s0: +1.76 (p=0.001) |

p-value convention: two-sided shifted (null-centred) bootstrap, p = (#{|d_i - mean(d)| >= |delta|} + 1) / (n + 1).
The per-seed p-values (and the pooled one) measure TEST-SET resampling variance only, not training-seed variance. A seed
counts as significant only in the direction of its own delta (B>A sig / A>B sig / not sig). A cell difference is only
credible when the per-seed deltas agree in sign and their mean is large relative to their std over seeds. "Pooled" = one
bootstrap over the seed-summed sentence statistics (corpus chrF of all seeds' outputs together).
This is more conservative than `sacrebleu --paired-bs` (which compares against the mean of |d| with a strict '>'), so
sacrebleu reports smaller p-values for the same pair (e.g. shipped vanilla vs lstmp: 0.115 here vs 0.061 in sacrebleu).

## 7. Hyperparameter study (vanilla cell + attention, 20 epochs / patience 4, one factor at a time)
| setting (one change vs base) | best val loss | best epoch | test chrF | test BLEU | Δ chrF vs base | seeds |
|---|---|---|---|---|---|---|
| Adam, lr 1e-3, default init, batch 64, dropout 0.4 | 4.654 ± 0.016 | 20.0 ± 0.0 | 21.26 ± 0.25 | 2.50 ± 0.15 |  | 3 |
| learning rate 3e-4 | 5.179 | 20.0 | 15.76 | 0.80 | -5.51 | 1 |
| learning rate 3e-3 | 4.538 | 17.0 | 21.94 | 2.52 | +0.68 | 1 |
| optimizer AdamW | 4.663 | 20.0 | 21.14 | 2.50 | -0.13 (within noise) | 1 |
| optimizer RMSprop | 4.688 | 20.0 | 20.93 | 2.39 | -0.34 (within noise) | 1 |
| optimizer SGD (momentum 0.9, lr 0.05) | 5.364 | 20.0 | 15.25 | 0.58 | -6.01 | 1 |
| init Xavier-uniform | 4.648 | 20.0 | 21.40 | 1.93 | +0.14 (within noise) | 1 |
| init orthogonal | 4.701 | 20.0 | 20.15 | 2.20 | -1.12 | 1 |
| batch 32 | 4.581 | 19.0 | 22.04 | 2.45 | +0.77 | 1 |
| batch 128 | 4.786 | 20.0 | 20.22 | 1.93 | -1.04 | 1 |
| dropout 0.2 | 4.804 | 12.0 | 20.44 | 2.21 | -0.82 | 1 |
| dropout 0.5 | 4.697 | 20.0 | 20.04 | 2.39 | -1.22 | 1 |

Selected on validation loss, reported on test. "within noise" = |Δ| < 2 std of the base seeds. The 20-epoch cap
favours fast-converging settings (SGD in particular is still far from converged).

## 8. Data scaling (+ 31,571 leakage-filtered DODa pairs; same val/test and tokenizers; 25 epochs / patience 4)
| system | train pairs | best val loss | test chrF | test BLEU | chrF beam5 | seeds |
|---|---|---|---|---|---|---|
| vanilla + attention | 10,131 | 4.599 ± 0.011 | 22.64 ± 0.22 | 3.07 ± 0.24 | 18.90 ± 0.18 | 3 |
| vanilla + attention | 41,702 | 4.050 | 24.29 | 3.39 | 17.44 | 1 |
| retrieval baseline | 10,131 | - | 19.01 | 1.88 | - | - |
| retrieval baseline | 41,702 | - | 18.46 | 1.64 | - | - |

Val loss is comparable here (same tokenizers, same val set). The retrieval baseline over the larger corpus is the
leakage control: it does not improve, so the extra data contains no copies of test sentences.

## 9. Attention score functions × cells (10k pairs, test chrF, greedy, 3 seeds)
| cell | no attention | dot | scaled dot | general | additive |
|---|---|---|---|---|---|
| vanilla | 19.35 ± 0.34 | 22.64 ± 0.22 | 20.98 ± 0.19 | 22.98 ± 0.80 | 23.18 ± 0.34 |
| cifg | 18.56 ± 0.34 | 22.06 ± 0.24 | - | 22.89 ± 0.48 | 23.30 ± 0.22 |
| lstmp | 19.11 ± 0.28 | 17.40 ± 0.12 | 19.96 ± 0.45 | 18.13 ± 1.47 | 20.86 ± 0.38 |
| wmc | 19.48 ± 0.28 | 21.92 ± 0.23 | - | 21.20 ± 1.99 | 23.07 ± 0.20 |

Attention saturation (mean max attention weight / normalised entropy over 300 test sentences, seed 0;
1 / 0 = all weight on one source position, small / 1 = uniform):
| cell | dot | scaled dot | general | additive |
|---|---|---|---|---|
| vanilla | 0.65 / 0.40 | 0.24 / 0.91 | 0.72 / 0.31 | 0.44 / 0.66 |
| cifg | 0.62 / 0.45 | - | 0.68 / 0.37 | 0.44 / 0.65 |
| lstmp | 0.95 / 0.07 | 0.34 / 0.81 | 1.00 / 0.00 | 0.39 / 0.73 |
| wmc | 0.65 / 0.41 | - | 1.00 / 0.00 | 0.48 / 0.61 |

## 10. Four times more data (41,702 pairs; tokenizer chosen on validation; 25 epochs / patience 4)
Tokenizer study (vanilla + dot attention, seed 0; chosen on VALIDATION chrF):
| setting | params | best epoch | val chrF | test chrF | test BLEU |
|---|---|---|---|---|---|
| BPE 3k trained on 10k pairs (data/) | 2,458,552 | 25 | 24.61 | 24.29 | 3.39 |
| BPE 3k trained on 42k | 2,458,552 | 25 | 24.73 | 24.64 | 2.92 |
| BPE 6k trained on 42k | 3,997,552 | 25 | 25.27 | 24.55 | 3.71 |
| BPE 10k trained on 42k | 6,049,552 | 25 | 24.55 | 24.83 | 3.77 |

Cells on the 42k corpus (test chrF, greedy):
| cell | no attention | additive attention |
|---|---|---|
| vanilla | - | 26.31 (n=1) |
| cifg | - | 25.71 (n=1) |
| lstmp | - | 23.00 (n=1) |
| wmc | - | - |

## 11. Best-system ablations (vanilla + additive attention, 42k, seed 0; one change at a time)
| setting | params | best epoch | val chrF | test chrF | test BLEU |
|---|---|---|---|---|---|
| base (c42 vanilla) | 4,129,136 | 25 | 26.68 | 26.31 | 3.61 |

## 12. Final recipe
| cell | final recipe |
|---|---|
| vanilla | - |
| cifg | - |
| lstmp | - |
| wmc | - |

## 13. Context: a small Transformer (nn.Transformer, 3+3 layers, d 256)
| cell | 10k pairs | 42k pairs |
|---|---|---|
| transformer | 18.21 (n=1) | - |

## 14. Significance of the cell comparisons in the extended study
| comparison (chrF of B - A) | per-seed delta mean ± std | seeds B>A sig / A>B sig / n.s. | pooled-seeds bootstrap delta (p) | per seed |
|---|---|---|---|---|
| vanilla/attn_additive -> cifg/attn_additive | 0.11 ± 0.14 | 0 / 0 / 3 | +0.11 (p=0.501) | s0: +0.28 (p=0.350), s1: +0.02 (p=0.961), s2: +0.05 (p=0.876) |
| vanilla/attn_additive -> lstmp/attn_additive | -2.33 ± 0.69 | 0 / 3 / 0 | -2.33 (p=0.001) | s0: -1.69 (p=0.001), s1: -2.24 (p=0.001), s2: -3.05 (p=0.001) |
| vanilla/attn_additive -> wmc/attn_additive | -0.11 ± 0.29 | 0 / 0 / 3 | -0.11 (p=0.498) | s0: +0.21 (p=0.457), s1: -0.37 (p=0.216), s2: -0.18 (p=0.568) |
| vanilla/c42 -> cifg/c42 | -0.60 | 0 / 0 / 1 | -0.60 (p=0.068) | s0: -0.60 (p=0.068) |
| vanilla/c42 -> lstmp/c42 | -3.32 | 0 / 1 / 0 | -3.32 (p=0.001) | s0: -3.32 (p=0.001) |
| vanilla/attn -> vanilla/attn_additive | 0.55 ± 0.38 | 1 / 0 / 2 | +0.55 (p=0.004) | s0: +0.28 (p=0.381), s1: +0.38 (p=0.229), s2: +0.98 (p=0.001) |
| lstmp/attn -> lstmp/attn_additive | 3.46 ± 0.28 | 3 / 0 / 0 | +3.46 (p=0.001) | s0: +3.59 (p=0.001), s1: +3.65 (p=0.001), s2: +3.14 (p=0.001) |
| vanilla/attn_additive -> vanilla/c42 | +3.51 | 1 / 0 / 0 | +3.51 (p=0.001) | s0: +3.51 (p=0.001) |

Val-loss curves: outputs/curves.png
