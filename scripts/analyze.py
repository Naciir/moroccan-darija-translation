"""Aggregate results.jsonl + results_test.jsonl (+ outputs/*.txt) into RESULTS.md and outputs/curves.png.
Runs are grouped by (model, tag) across seeds -> mean +- std. Significance: paired bootstrap on corpus
chrF (resampled sentence-level sufficient statistics, NOT averaged sentence scores)."""
import argparse, sys, json, os
from collections import defaultdict
import numpy as np
import pandas as pd
from sacrebleu.metrics import CHRF

LEGACY = [  # shipped single-seed numbers (original code, 20 epochs, test greedy)
    ("vanilla", 2327480, 20, 4.2595, 17.10, 1.27), ("cifg", 2130360, 20, 4.2693, 17.13, 1.11),
    ("lstmp", 1747128, 20, 4.2893, 17.45, 1.54)]
CELLS = ["vanilla", "cifg", "lstmp"]

def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def ms(x, nd=2):
    x = [v for v in x if v is not None]
    if not x:
        return "-"
    return f"{np.mean(x):.{nd}f} ± {np.std(x, ddof=1):.{nd}f}" if len(x) > 1 else f"{x[0]:.{nd}f}"

class Results:
    def __init__(self, a):
        self.a = a
        self.train = {r["run"]: r for r in read_jsonl(a.train_results)}          # last record wins
        self.test = {(r["run"], r["split"], r["beam"]): r for r in read_jsonl(a.test_results)
                     if r.get("alpha") in (None, 0.6)}                           # --alpha ablations: not tabled
        self.groups = defaultdict(dict)                                          # (model, tag) -> {seed: run}
        for run, r in self.train.items():
            self.groups[(r["model"], r["tag"])][r["seed"]] = run
        keys = ("tok_dir", "mask", "attn", "no_src", "epochs", "patience", "bucket")
        for g, seeds in self.groups.items():                                     # seeds of a group must share a config
            cfgs = {tuple(self.train[r].get(k) for k in keys) for r in seeds.values()}
            if len(cfgs) > 1:
                print(f"WARNING: {g} mixes configs across seeds {sorted(seeds)}: {cfgs}")
        self.refs = [str(x) for x in pd.read_csv("data/test.csv")["dar"]]
        self.valres = {r["run"]: r["chrf"] for r in read_jsonl("results_val.jsonl")
                       if r.get("split") == "val" and r.get("beam") == 1}              # last record wins
        self._stats = {}

    def row(self, model, tag, label=None, beam5=True):
        runs = [self.groups[(model, tag)][s] for s in sorted(self.groups.get((model, tag), {}))]
        tr = [self.train[r] for r in runs]
        g = [self.test.get((r, "test", 1)) for r in runs]
        b = [self.test.get((r, "test", 5)) for r in runs]
        n = len(runs)
        get = lambda lst, k: [x[k] for x in lst if x]
        msn = lambda x, nd=2: ms(x, nd) + (f" (n={len(x)})" if x and len(x) != n else "")   # flag missing evals
        cells = [label or f"{model} / {tag}", f"{tr[0]['params']:,}" if tr else "-",
                 ms(get(tr, "best_epoch"), 1), ms(get(tr, "best_val_loss"), 3),
                 msn(get(g, "chrf")), msn(get(g, "chrfpp")), msn(get(g, "bleu"))]
        if beam5:
            cells += [msn(get(b, "chrf")), msn(get(b, "bleu")),
                      f"{ms(get(g, 'len_ratio'))} / {ms(get(b, 'len_ratio'))}"]
        return cells + [str(len(runs))]

    def table(self, rows, beam5=True):
        hdr = ["run (model / tag)", "params", "best epoch", "val loss*", "test chrF", "test chrF++", "test BLEU"]
        hdr += ["chrF beam5", "BLEU beam5", "len ratio greedy / beam5"] if beam5 else []
        hdr += ["seeds"]
        lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])

    # ---- paired bootstrap on corpus chrF -------------------------------------------------------
    def stats(self, run):
        if run not in self._stats:
            p = f"{self.a.out_dir}/{run}_test_beam1.txt"
            if not os.path.exists(p):
                return None
            with open(p, encoding="utf-8") as f:
                hyps = f.read().split("\n")
            if hyps and hyps[-1] == "":
                hyps = hyps[:-1]                                                 # trailing newline only
            if len(hyps) != len(self.refs):
                print(f"WARNING: {p} has {len(hyps)} lines, expected {len(self.refs)} -> ignored"); return None
            self._stats[run] = np.array(CHRF()._extract_corpus_statistics(hyps, [self.refs]), dtype=np.float64)
        return self._stats[run]

    def bootstrap(self, run_a, run_b, n=1000, seed=12345):
        """delta = chrF(b) - chrF(a) on the full test set. Both systems are resampled with the SAME indices;
        corpus chrF of each resample is recomputed from the summed sentence statistics. Two-sided p-value of
        the shifted (null-centred) bootstrap: p = (#{|d_i - mean(d)| >= |delta|} + 1) / (n + 1).
        run_a / run_b may be LISTS of runs (seeds): their sentence statistics are summed per sentence, i.e.
        corpus chrF of all seeds' outputs together ("pooled"), resampling sentences only."""
        get = lambda r: self.stats(r) if isinstance(r, str) else (
            None if any(self.stats(x) is None for x in r) else sum(self.stats(x) for x in r))
        sa, sb = get(run_a), get(run_b)
        if sa is None or sb is None:
            return None
        return boot_stats(sa, sb, n, seed)

    def sig_rows(self, pairs):
        out = []
        for (ma, ta), (mb, tb) in pairs:
            ga, gb = self.groups.get((ma, ta), {}), self.groups.get((mb, tb), {})
            res = [(s, self.bootstrap(ga[s], gb[s])) for s in sorted(set(ga) & set(gb))]
            res = [(s, r) for s, r in res if r]
            if not res:
                out.append([f"{ma}/{ta} -> {mb}/{tb}"] + ["-"] * 4); continue
            pooled = self.bootstrap([ga[s] for s, _ in res], [gb[s] for s, _ in res])
            out.append([f"{ma}/{ta} -> {mb}/{tb}"] + sig_cells(res, pooled))
        hdr = ("| comparison (chrF of B - A) | per-seed delta mean ± std | seeds B>A sig / A>B sig / n.s. "
               "| pooled-seeds bootstrap delta (p) | per seed |\n|---|---|---|---|---|\n")
        return hdr + "\n".join("| " + " | ".join(r) + " |" for r in out)

    def hp_table(self):
        """One-factor HP study (vanilla + attention, 20 epochs). Base = mean ± std over its seeds (the noise bar)."""
        HP = [("base", "Adam, lr 1e-3, default init, batch 64, dropout 0.4"),
              ("lr3e-4", "learning rate 3e-4"), ("lr3e-3", "learning rate 3e-3"), ("adamw", "optimizer AdamW"),
              ("rmsprop", "optimizer RMSprop"), ("sgd", "optimizer SGD (momentum 0.9, lr 0.05)"),
              ("xavier", "init Xavier-uniform"), ("orthogonal", "init orthogonal"), ("bs32", "batch 32"),
              ("bs128", "batch 128"), ("do0.2", "dropout 0.2"), ("do0.5", "dropout 0.5")]
        base = self.groups.get(("vanilla", "hp_base"), {})
        bc = [self.test[(r, "test", 1)]["chrf"] for r in base.values() if (r, "test", 1) in self.test]
        bmean, bstd = (np.mean(bc), np.std(bc, ddof=1) if len(bc) > 1 else 0.0) if bc else (None, None)
        rows = []
        for k, label in HP:
            g = self.groups.get(("vanilla", f"hp_{k}"), {})
            if not g:
                continue
            runs = [g[s] for s in sorted(g)]
            tr = [self.train[r] for r in runs]
            te = [self.test.get((r, "test", 1)) for r in runs]
            ch = [x["chrf"] for x in te if x]
            d = "" if k == "base" or not ch or bmean is None else f"{np.mean(ch) - bmean:+.2f}" + (
                " (within noise)" if bstd and abs(np.mean(ch) - bmean) < 2 * bstd else "")
            rows.append([label, ms([x["best_val_loss"] for x in tr], 3), ms([x["best_epoch"] for x in tr], 1),
                         ms(ch), ms([x["bleu"] for x in te if x]), d, str(len(runs))])
        hdr = ["setting (one change vs base)", "best val loss", "best epoch", "test chrF", "test BLEU",
               "Δ chrF vs base", "seeds"]
        return "\n".join(["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)] +
                         ["| " + " | ".join(r) + " |" for r in rows])

    def scaling_table(self):
        rows = []
        for m, tag, label, n in [("vanilla", "main", "vanilla, no attention", "10,131"),
                                 ("vanilla", "plus", "vanilla, no attention", "41,702"),
                                 ("vanilla", "attn", "vanilla + attention", "10,131"),
                                 ("vanilla", "attn_plus", "vanilla + attention", "41,702"),
                                 ("cifg", "attn", "CIFG + attention", "10,131"),
                                 ("cifg", "attn_plus", "CIFG + attention", "41,702"),
                                 ("lstmp", "attn", "LSTMP + attention", "10,131"),
                                 ("lstmp", "attn_plus", "LSTMP + attention", "41,702")]:
            g = self.groups.get((m, tag), {})
            partner = tag[:-5] if tag.endswith("_plus") else ("main" if tag == "plus" else
                      ("plus" if tag == "main" else tag + "_plus"))
            if not g or not self.groups.get((m, partner)):
                continue                                     # only systems trained on both data sizes
            runs = [g[s] for s in sorted(g)]
            tr = [self.train[r] for r in runs]
            get = lambda b, k: [self.test[(r, "test", b)][k] for r in runs if (r, "test", b) in self.test]
            rows.append([label, n, ms([x["best_val_loss"] for x in tr], 3), ms(get(1, "chrf")), ms(get(1, "bleu")),
                         ms(get(5, "chrf")) if get(5, "chrf") else "-", str(len(runs))])
        for name, n in [("retrieval", "10,131"), ("retrieval_plus", "41,702")]:
            r = self.test.get((f"baseline_{name}", "test", 0))
            if r:
                rows.append(["retrieval baseline", n, "-", f"{r['chrf']:.2f}", f"{r['bleu']:.2f}", "-", "-"])
        hdr = ["system", "train pairs", "best val loss", "test chrF", "test BLEU", "chrF beam5", "seeds"]
        return "\n".join(["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)] +
                         ["| " + " | ".join(r) + " |" for r in rows])

    # ---- extended study ----------------------------------------------------------------------------------------
    def chrf(self, model, tag, beam=1, key="chrf"):
        g = self.groups.get((model, tag), {})
        return [self.test[(g[s], "test", beam)][key] for s in sorted(g) if (g[s], "test", beam) in self.test]

    def compact(self, cells, cols, key="chrf"):
        """rows = cells, columns = (tag, label): test score mean ± std over seeds, (n=k) when fewer than 3 seeds."""
        out = ["| cell | " + " | ".join(lab for _, lab in cols) + " |", "|---|" + "---|" * len(cols)]
        for m in cells:
            vals = []
            for tag, _ in cols:
                v = self.chrf(m, tag, key=key)
                vals.append(ms(v) + (f" (n={len(v)})" if 0 < len(v) < 3 else ""))
            out.append(f"| {m} | " + " | ".join(vals) + " |")
        return "\n".join(out)

    def runs_table(self, rows):
        """rows = [(label, run)], one seed each: params, best epoch, val chrF, test chrF / BLEU."""
        out = ["| setting | params | best epoch | val chrF | test chrF | test BLEU |", "|---|---|---|---|---|---|"]
        fmt = lambda v: "-" if v is None else f"{v:.2f}"
        for label, run in rows:
            tr, te = self.train.get(run), self.test.get((run, "test", 1)) or {}
            if tr:
                out.append(f"| {label} | {tr['params']:,} | {tr['best_epoch']} | {fmt(self.valres.get(run))} | "
                           f"{fmt(te.get('chrf'))} | {fmt(te.get('bleu'))} |")
        return "\n".join(out)

def boot_stats(sa, sb, n=1000, seed=12345):
    """Paired bootstrap of corpus chrF(b) - chrF(a) from (sentences x stats) arrays -> (delta, p)."""
    assert sa.shape == sb.shape, f"paired bootstrap needs equal-shaped stats, got {sa.shape} vs {sb.shape}"
    m = CHRF()
    f = lambda s: m._compute_f_score(list(s))
    idx = np.random.RandomState(seed).randint(0, len(sa), size=(n, len(sa)))
    da = np.array([f(sa[i].sum(0)) for i in idx]); db = np.array([f(sb[i].sum(0)) for i in idx])
    obs = f(sb.sum(0)) - f(sa.sum(0))
    deltas = db - da
    p = ((np.abs(deltas - deltas.mean()) >= abs(obs)).sum() + 1) / (n + 1)
    return obs, p

def sig_cells(res, pooled, alpha=0.05):
    """res = [(seed, (delta, p))]. A seed is significant only in the direction of ITS OWN delta (opposite-sign
    seeds never add up to '2/2'). The per-seed bootstrap only measures test-set resampling variance; the std of
    the per-seed deltas is the training-seed variation it misses."""
    d = [x for _, (x, _) in res]
    up = sum(x > 0 and p < alpha for _, (x, p) in res)
    down = sum(x < 0 and p < alpha for _, (x, p) in res)
    return [ms(d) if len(d) > 1 else f"{d[0]:+.2f}", f"{up} / {down} / {len(res) - up - down}",
            f"{pooled[0]:+.2f} (p={pooled[1]:.3f})" if pooled else "-",
            ", ".join(f"s{s}: {x:+.2f} (p={p:.3f})" for s, (x, p) in res)]

def curves(R, path):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib unavailable, skipping curves:", e); return
    panels = [("main + attn (same tokenizer)", [(m, t) for t in ["main", "attn"] for m in CELLS]),
              ("vanilla ablation (val loss NOT comparable across tokenizers)",
               [("vanilla", t) for t in ["old", "mask", "tok", "main", "nosrc", "mono"]])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (title, keys) in zip(axes, panels):
        for k, key in enumerate(keys):
            for j, s in enumerate(sorted(R.groups.get(key, {}))):
                vl = R.train[R.groups[key][s]]["val_losses"]
                ax.plot(range(1, len(vl) + 1), vl, color=f"C{k}", alpha=0.8, lw=1.2,
                        ls="-" if key[1] != "attn" else "--", label="/".join(key) if j == 0 else None)
        ax.set_title(title, fontsize=10); ax.set_xlabel("epoch"); ax.set_ylabel("val loss (teacher forced)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()
    print("wrote", path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_results", default="results.jsonl")
    ap.add_argument("--test_results", default="results_test.jsonl")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--md", default="RESULTS.md")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")            # Windows: "▁" etc. must survive redirected stdout
    R = Results(a)

    legacy = ("| run | params | best epoch | val loss (orig. def.: per-batch mean; NOT comparable) | test chrF "
              "| test BLEU | seeds |\n|---|---|---|---|---|---|---|\n") + \
        "\n".join(f"| {m} / shipped | {p:,} | {e} | {v:.3f} | {c:.2f} | {b:.2f} | 1 |" for m, p, e, v, c, b in LEGACY)
    abl = R.table([R.row("vanilla", "old", "old: orig tokenizer, no mask, random batches (= original setup)",
                         beam5=False),
                   R.row("vanilla", "tok", "+ new tokenizer (no mask, random batches)", beam5=False),
                   R.row("vanilla", "mask", "+ mask (orig tokenizer, random batches)", beam5=False),
                   R.row("vanilla", "main", "full fix: new tokenizer + mask + length bucketing", beam5=False)],
                  beam5=False)
    main_t = R.table([R.row(m, "main") for m in CELLS])
    ctrl = [R.row("vanilla", "nosrc", "vanilla / nosrc (LM control: no source)", beam5=False)]
    if ("vanilla", "mono") in R.groups:
        ctrl.append(R.row("vanilla", "mono", "vanilla / mono (DAR tokenizer + mono text)", beam5=False))
    for name in ["retrieval", "mostfreq"]:
        r = R.test.get((f"baseline_{name}", "test", 0))
        ctrl.append([f"baseline_{name}", "-", "-", "-"] +
                    ([f"{r['chrf']:.2f}", f"{r['chrfpp']:.2f}", f"{r['bleu']:.2f}"]
                    if r else ["-"] * 3) + ["-"])
    attn = R.table([R.row(m, "attn") for m in CELLS] +
                   [R.row(m, "attn_scaled", f"{m} / attn, scaled dot product (score / sqrt(d))")
                    for m in ["lstmp", "vanilla"] if (m, "attn_scaled") in R.groups])
    sig = R.sig_rows([(("vanilla", "main"), ("cifg", "main")), (("vanilla", "main"), ("lstmp", "main")),
                      (("cifg", "main"), ("lstmp", "main")), (("vanilla", "old"), ("vanilla", "main")),
                      (("vanilla", "attn"), ("cifg", "attn")), (("vanilla", "attn"), ("lstmp", "attn")),
                      (("cifg", "attn"), ("lstmp", "attn")), (("vanilla", "main"), ("vanilla", "attn")),
                      (("cifg", "main"), ("cifg", "attn")), (("lstmp", "main"), ("lstmp", "attn")),
                      (("vanilla", "old"), ("vanilla", "tok")), (("vanilla", "old"), ("vanilla", "mask")),
                      (("lstmp", "attn"), ("lstmp", "attn_scaled")), (("vanilla", "attn"), ("vanilla", "attn_plus"))])

    CELLS4 = ["vanilla", "cifg", "lstmp", "wmc"]
    att = json.load(open("attention_stats.json")) if os.path.exists("attention_stats.json") else {}
    att_rows = ["| cell | dot | scaled dot | general | additive |", "|---|---|---|---|---|"]
    for m in CELLS4:
        cells = []
        for tag in ["attn", "attn_scaled", "attn_general", "attn_additive"]:
            st = att.get(f"{m}_{tag}_s0")
            cells.append("-" if not st else f"{st['max_weight']:.2f} / {st['entropy']:.2f}")
        att_rows.append(f"| {m} | " + " | ".join(cells) + " |")
    att_md = "\n".join(att_rows)
    tok_md = R.runs_table([("BPE 3k trained on 10k pairs (data/)", "vanilla_attn_plus_s0"),
                           ("BPE 3k trained on 42k", "vanilla_tokp3k_s0"),
                           ("BPE 6k trained on 42k", "vanilla_tokp6k_s0"),
                           ("BPE 10k trained on 42k", "vanilla_tokp10k_s0")])
    abl_md = R.runs_table([("base (c42 vanilla)", "vanilla_c42_s0"),
                           ("+ bidirectional encoder", "vanilla_abl_bidir_s0"),
                           ("+ label smoothing 0.1", "vanilla_abl_ls_s0"), ("+ BPE-dropout 0.1", "vanilla_abl_bpe_s0"),
                           ("learning rate 3e-3", "vanilla_abl_lr_s0"), ("batch 32", "vanilla_abl_bs32_s0"),
                           ("+ back-translated pairs", "vanilla_bt_s0")])
    dec = json.load(open("night_decisions.json")) if os.path.exists("night_decisions.json") else {}
    final_tag = dec.get("final_tag", "fin")
    fin_md = ""
    if dec.get("final_args"):
        fin_md = ("Recipe chosen on validation (a change is adopted when it beats the base by >= 0.3 val chrF): `"
                  + " ".join(dec["final_args"]) + "`\n\n")
    fin_md += R.compact(CELLS4, [(final_tag, "final recipe")])
    lines = []
    for r in read_jsonl("results_alpha.jsonl"):
        how = "greedy" if r["selected_beam"] == 1 else "beam %d, alpha %g" % (r["selected_beam"], r["selected_alpha"])
        lines.append("- %s: %s -> test chrF %.2f" % (r["run"], how, r["test_chrf"]))
    if lines:
        fin_md += "\n\nDecoding chosen on validation (greedy vs beam 5 with alpha in 0..1.5):\n" + "\n".join(lines)
    pairs2 = []
    for tag in ["attn_additive", "c42", final_tag]:
        pairs2 += [(("vanilla", tag), (m, tag)) for m in ["cifg", "lstmp", "wmc"]]
    pairs2 += [(("vanilla", "attn"), ("vanilla", "attn_additive")), (("lstmp", "attn"), ("lstmp", "attn_additive")),
               (("vanilla", "attn_additive"), ("vanilla", "c42"))]
    sig2 = R.sig_rows([q for q in dict.fromkeys(pairs2) if q[0] in R.groups and q[1] in R.groups])
    md = f"""# Results

Test set = data/test.csv ({len(R.refs)} sentences). Scores are sacrebleu 2.6 corpus scores; mean ± std over seeds.
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
{legacy}

## 2. Ablation (vanilla cell): what each fix contributes
old/tok/mask use random batches like the original (~65% leading pads for unmasked runs); main adds length bucketing,
which with the mask on only changes batch composition. old->mask = mask effect, old->tok = tokenizer effect.
{abl}

## 3. Main table: cell comparison (new tokenizer + masked encoder, no attention)
{main_t}

## 4. Controls and baselines
{R.table(ctrl, beam5=False)}

## 5. Secondary experiment: Luong (dot) attention
{attn}

## 6. Paired bootstrap significance (corpus chrF, greedy, 1000 resamples, seed 12345, pairing seed i with seed i)
{sig}

p-value convention: two-sided shifted (null-centred) bootstrap, p = (#{{|d_i - mean(d)| >= |delta|}} + 1) / (n + 1).
The per-seed p-values (and the pooled one) measure TEST-SET resampling variance only, not training-seed variance. A seed
counts as significant only in the direction of its own delta (B>A sig / A>B sig / not sig). A cell difference is only
credible when the per-seed deltas agree in sign and their mean is large relative to their std over seeds. "Pooled" = one
bootstrap over the seed-summed sentence statistics (corpus chrF of all seeds' outputs together).
This is more conservative than `sacrebleu --paired-bs` (which compares against the mean of |d| with a strict '>'), so
sacrebleu reports smaller p-values for the same pair (e.g. shipped vanilla vs lstmp: 0.115 here vs 0.061 in sacrebleu).

## 7. Hyperparameter study (vanilla cell + attention, 20 epochs / patience 4, one factor at a time)
{R.hp_table()}

Selected on validation loss, reported on test. "within noise" = |Δ| < 2 std of the base seeds. The 20-epoch cap
favours fast-converging settings (SGD in particular is still far from converged).

## 8. Data scaling (+ 31,571 leakage-filtered DODa pairs; same val/test and tokenizers; 25 epochs / patience 4)
{R.scaling_table()}

Val loss is comparable here (same tokenizers, same val set). The retrieval baseline over the larger corpus is the
leakage control: it does not improve, so the extra data contains no copies of test sentences.

## 9. Attention score functions × cells (10k pairs, test chrF, greedy, 3 seeds)
{R.compact(CELLS4, [("main", "no attention"), ("attn", "dot"), ("attn_scaled", "scaled dot"),
                    ("attn_general", "general"), ("attn_additive", "additive")])}

Attention saturation (mean max attention weight / normalised entropy over 300 test sentences, seed 0;
1 / 0 = all weight on one source position, small / 1 = uniform):
{att_md}

## 10. Four times more data (41,702 pairs; tokenizer chosen on validation; 25 epochs / patience 4)
Tokenizer study (vanilla + dot attention, seed 0; chosen on VALIDATION chrF):
{tok_md}

Cells on the 42k corpus (test chrF, greedy):
{R.compact(CELLS4, [("n42", "no attention"), ("c42", "additive attention")])}

## 11. Best-system ablations (vanilla + additive attention, 42k, seed 0; one change at a time)
{abl_md}

## 12. Final recipe
{fin_md}

## 13. Context: a small Transformer (nn.Transformer, 3+3 layers, d 256)
{R.compact(["transformer"], [("trf", "10k pairs"), ("trf42", "42k pairs")])}

## 14. Significance of the cell comparisons in the extended study
{sig2}

Val-loss curves: outputs/curves.png
"""
    with open(a.md, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    os.makedirs(a.out_dir, exist_ok=True); curves(R, f"{a.out_dir}/curves.png")

if __name__ == "__main__":
    main()
