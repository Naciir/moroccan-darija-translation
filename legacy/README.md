# legacy/

`cells_original.py` is the original implementation of the three cells, kept verbatim: one `nn.Linear` per gate
(`W_xi`, `W_hi`, ...). `src/darija_nmt/cells.py` fuses the gates into two matrices per cell; the tests check that both
give the same outputs and the same parameter counts (`test_01`, `test_02`), and that
`cells.convert_legacy_state_dict` maps the old weights onto the fused layout.

The original checkpoints and their outputs are not in git. To run the tests marked `needs_checkpoints`, put them here:

```
legacy/checkpoints/{vanilla,cifg,lstmp}_base_s0.pt       # original per-gate weights (no .json config)
legacy/outputs/vanilla_base_s0_pen0.0.csv                # the shipped test translations (column "pred")
```

`darija_nmt.evaluate.load_run` loads them as legacy runs: without a `<run>.json` it assumes the original setup
(`data/tok_orig` tokenizers, no attention) and converts the per-gate weights, e.g.

```bash
python -m darija_nmt.evaluate --runs vanilla_base_s0 --runs_dir legacy/checkpoints --out_dir outputs/legacy
```
