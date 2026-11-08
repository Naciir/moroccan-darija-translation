"""Secondary analyses on top of analyze.py -> EXTRAS.md + outputs/attn_*.png, outputs/len_buckets.png.
Run after the grid (and after evaluate.py on the legacy checkpoints into outputs/legacy/, see docs/bug_hunt.md).

  1. tokenizer statistics (orig / new / mono) on the test references
  2. Arabizi digit-word recall: how many reference words that contain a digit-letter (3la, ch7al) the
     system reproduces exactly -- the quantity the tokenizer fix targets directly
  3. degeneration: repeated-word outputs, distinct-n, output length
  4. corpus chrF by source length bucket (pooled seeds)
  5. beam length-penalty (alpha) sweep, if evaluate.py --alpha records exist
  6. qualitative examples (fixed random sample)
  7. attention heatmaps for the attention models
All word-level numbers are on whitespace tokens, case-sensitive (capitals are phonemic in Arabizi)."""
import argparse, json, os, re, sys
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import sentencepiece as spm
from sacrebleu.metrics import CHRF

DIGIT_WORD = re.compile(r"[0-9]")
LETTER = re.compile(r"[A-Za-z]")

def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def read_hyps(path, n):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        h = f.read().split("\n")
    if h and h[-1] == "":
        h = h[:-1]
    return h if len(h) == n else None

def ms(x, nd=2):
    x = [v for v in x if v is not None]
    if not x:
        return "-"
    return f"{np.mean(x):.{nd}f} ± {np.std(x, ddof=1):.{nd}f}" if len(x) > 1 else f"{x[0]:.{nd}f}"

def md_table(hdr, rows):
    return "\n".join(["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)] +
                     ["| " + " | ".join(map(str, r)) + " |" for r in rows])

# ---- 1. tokenizers ----------------------------------------------------------------------------------
def tok_stats(refs, tok_dirs):
    rows = []
    words = sum(len(r.split()) for r in refs)
    dwords = [w for r in refs for w in r.split() if DIGIT_WORD.search(w) and LETTER.search(w)]
    for name, d in tok_dirs:
        p = f"{d}/spm_dar.model"
        if not os.path.exists(p):
            continue
        sp = spm.SentencePieceProcessor(model_file=p)
        toks = [sp.encode(r, out_type=str) for r in refs]
        n = sum(map(len, toks))
        dig = sum(1 for t in toks for x in t if re.fullmatch(r"▁?[0-9]+", x))
        whole = sum(len(sp.encode(w, out_type=str)) == 1 for w in dwords)
        frag = np.mean([len(sp.encode(w, out_type=str)) for w in dwords])
        ex = " ".join("/".join(sp.encode(w, out_type=str)) for w in ["3la", "ch7al", "t7kem", "nt9adou"])
        rows.append([name, d, sp.get_piece_size(), f"{n / words:.2f}", f"{dig / n:.1%}",
                     f"{whole / len(dwords):.1%}", f"{frag:.2f}", ex])
    return md_table(["tokenizer", "dir", "pieces", "DAR tok/word", "digit-only pieces",
                     "digit-words kept whole", "pieces per digit-word", "examples"], rows), len(dwords) / words

# ---- 2/3. word-level diagnostics --------------------------------------------------------------------
def word_diag(hyps, refs):
    """-> dict of word-level numbers for one system (clipped counts, corpus level)."""
    match_d = tot_d = match_all = tot_ref = tot_hyp = 0
    rep = 0
    uni, bi = Counter(), Counter()
    for h, r in zip(hyps, refs):
        hw, rw = h.split(), r.split()
        hc = Counter(hw)
        rc_d = Counter(w for w in rw if DIGIT_WORD.search(w) and LETTER.search(w))
        match_d += sum(min(c, hc[w]) for w, c in rc_d.items()); tot_d += sum(rc_d.values())
        rc = Counter(rw)
        match_all += sum(min(c, hc[w]) for w, c in rc.items()); tot_ref += len(rw); tot_hyp += len(hw)
        rep += any(a == b for a, b in zip(hw, hw[1:]))
        uni.update(hw); bi.update(zip(hw, hw[1:]))
    hyp_d = sum(c for w, c in uni.items() if DIGIT_WORD.search(w) and LETTER.search(w))
    return dict(digit_recall=100 * match_d / max(tot_d, 1), digit_share_hyp=100 * hyp_d / max(tot_hyp, 1),
                word_recall=100 * match_all / max(tot_ref, 1), word_prec=100 * match_all / max(tot_hyp, 1),
                repeat=100 * rep / len(hyps), distinct1=100 * len(uni) / max(tot_hyp, 1),
                distinct2=100 * len(bi) / max(sum(bi.values()), 1), len=tot_hyp / len(hyps))

# ---- 4. length buckets ------------------------------------------------------------------------------
def bucket_chrf(systems, refs, src):
    """systems: label -> list of hyp lists (seeds). Corpus chrF of the seed-pooled sufficient statistics."""
    m = CHRF()
    lens = np.array([len(s.split()) for s in src])
    edges = [(1, 4), (5, 7), (8, 10), (11, 15), (16, 999)]
    rows, plot = [], {}
    for label, seeds in systems.items():
        st = sum(np.array(m._extract_corpus_statistics(h, [refs]), dtype=np.float64) for h in seeds)
        vals = []
        for lo, hi in edges:
            idx = (lens >= lo) & (lens <= hi)
            vals.append(m._compute_f_score(list(st[idx].sum(0))) if idx.any() else float("nan"))
        rows.append([label, len(seeds)] + [f"{v:.2f}" for v in vals]); plot[label] = vals
    counts = [int(((lens >= lo) & (lens <= hi)).sum()) for lo, hi in edges]
    hdr = ["system", "seeds"] + [f"EN {lo}-{hi if hi < 999 else '+'} w (n={c})" for (lo, hi), c in zip(edges, counts)]
    return md_table(hdr, rows), plot, [f"{lo}-{hi if hi < 999 else '+'}" for lo, hi in edges]

def plot_buckets(plot, labels, path):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib unavailable:", e); return
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for k, (name, vals) in enumerate(plot.items()):
        ax.plot(labels, vals, marker="o", label=name, color=f"C{k}")
    ax.set_xlabel("source length (EN words)"); ax.set_ylabel("corpus chrF (seeds pooled)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8); plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()
    print("wrote", path)

# ---- 7. attention -----------------------------------------------------------------------------------
def attention_maps(run, sentences, out_prefix, runs_dir="checkpoints", max_len=30):
    import torch
    from darija_nmt.evaluate import load_run, left_pad, src_ids, BOS, EOS
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib unavailable:", e); return []
    model, cfg, sp_src, sp_tgt = load_run(run, runs_dir)
    if not cfg.get("attn"):
        return []
    dec, paths = model.decoder, []
    for n, sent in enumerate(sentences):
        ids = src_ids(sp_src, sent)                                  # EOS + reversed source pieces
        pieces = [sp_src.id_to_piece(i) for i in ids]
        with torch.no_grad():
            state, enc_out, mask = model.encode(left_pad([ids]))
            cur, out, att = torch.tensor([BOS], device=enc_out.device), [], []
            for _ in range(max_len):
                emb = dec.embedding(cur)
                h, c = dec.cell(emb, state); state = (h, c)
                score = torch.bmm(enc_out, h.unsqueeze(2)).squeeze(2)
                a = torch.softmax(score, dim=1)
                ctx = torch.bmm(a.unsqueeze(1), enc_out).squeeze(1)
                logits = dec.fc_out(torch.tanh(dec.W_c(torch.cat([ctx, h], dim=1))))
                cur = logits.argmax(1)
                if cur.item() == EOS:
                    break
                out.append(cur.item()); att.append(a[0].cpu().numpy())
        if not out:
            continue
        A = np.array(att)[:, ::-1]                                   # back to reading order
        src_lab = pieces[::-1]
        tgt_lab = [sp_tgt.id_to_piece(i) for i in out]
        fig, ax = plt.subplots(figsize=(max(4, 0.45 * len(src_lab) + 1.5), max(3, 0.35 * len(tgt_lab) + 1.2)))
        ax.imshow(A, cmap="Greys", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(src_lab))); ax.set_xticklabels(src_lab, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(tgt_lab))); ax.set_yticklabels(tgt_lab, fontsize=8)
        ax.set_title(f"{run}: {sent[:60]}", fontsize=8)
        plt.tight_layout(); p = f"{out_prefix}_{n}.png"; plt.savefig(p, dpi=120); plt.close(); paths.append(p)
    return paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_results", default="results.jsonl")
    ap.add_argument("--test_results", default="results_test.jsonl")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--md", default="EXTRAS.md")
    ap.add_argument("--n_examples", type=int, default=15)
    ap.add_argument("--no_attn_maps", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    df = pd.read_csv("data/test.csv")
    refs, src = [str(x) for x in df["dar"]], [str(x) for x in df["en"]]
    N = len(refs)
    train = {r["run"]: r for r in read_jsonl(a.train_results)}
    groups = defaultdict(dict)
    for run, r in train.items():
        groups[(r["model"], r["tag"])][r["seed"]] = run

    def hyps_of(run, beam=1):
        return read_hyps(f"{a.out_dir}/{run}_test_beam{beam}.txt", N)

    systems = {}                                                     # label -> [hyp lists]
    for m, s in [("vanilla", "shipped"), ("cifg", "shipped"), ("lstmp", "shipped")]:
        h = read_hyps(f"{a.out_dir}/legacy/{m}_base_s0_test_beam1.txt", N)
        if h:
            systems[f"{m} / shipped (orig code, 20 ep)"] = [h]
    order = [("vanilla", t) for t in ["old", "tok", "mask", "main", "nosrc", "mono"]] + \
            [(m, "main") for m in ["cifg", "lstmp"]] + [(m, "attn") for m in ["vanilla", "cifg", "lstmp"]]
    for key in order:
        hs = [hyps_of(groups[key][s]) for s in sorted(groups.get(key, {}))]
        hs = [h for h in hs if h]
        if hs:
            systems[f"{key[0]} / {key[1]}"] = hs
    for b in ["retrieval", "mostfreq"]:
        h = read_hyps(f"{a.out_dir}/baseline_{b}_test.txt", N)
        if h:
            systems[f"baseline_{b}"] = [h]
    systems["reference (upper bound for word stats)"] = [refs]

    # 1
    tok_md, dshare = tok_stats(refs, [("original (split_by_number=True)", "data/tok_orig"),
                                      ("fixed (split_by_number=False)", "data"),
                                      ("fixed + 75k monolingual Darija", "data/tok_mono")])
    # 2/3
    rows = []
    for label, hs in systems.items():
        ds = [word_diag(h, refs) for h in hs]
        rows.append([label, len(hs)] + [ms([d[k] for d in ds], 1) for k in
                    ["digit_recall", "digit_share_hyp", "word_recall", "word_prec", "repeat", "distinct1", "distinct2",
                     "len"]])
    word_md = md_table(["system", "seeds", "digit-word recall %", "digit-words in output %", "word recall %",
                        "word precision %", "outputs with repeated word %", "distinct-1 %", "distinct-2 %",
                        "avg words"], rows)
    # 4
    sel = {k: v for k, v in systems.items() if k.split(" / ")[-1].split(" ")[0] in ("shipped", "old", "main", "attn")
           and not k.startswith(("cifg / shipped", "lstmp / shipped"))}
    if "baseline_retrieval" in systems:
        sel["baseline_retrieval"] = systems["baseline_retrieval"]
    bucket_md, plot, labels = bucket_chrf(sel, refs, src)
    plot_buckets(plot, labels, f"{a.out_dir}/len_buckets.png")
    # 5
    sweep = [r for r in read_jsonl(a.test_results) if r.get("beam", 1) > 1 and r.get("split") == "test"]
    by = defaultdict(list)
    for r in sweep:
        by[(r["run"], r["beam"], r.get("alpha"))] = r                 # last record wins
    srows = sorted(by.values(), key=lambda r: (r["run"], r["beam"], r.get("alpha") or 0))
    alpha_md = md_table(["run", "beam", "alpha", "chrF", "BLEU", "len ratio", "empty"],
                        [[r["run"], r["beam"], r.get("alpha"), f"{r['chrf']:.2f}", f"{r['bleu']:.2f}",
                          f"{r['len_ratio']:.3f}", r.get("empty", "-")] for r in srows
                         if any(x["run"] == r["run"] and x.get("alpha") not in (None, 0.6) for x in sweep)])
    # 6
    rng = np.random.RandomState(7)
    pick = sorted(rng.choice(N, size=min(a.n_examples, N), replace=False))
    show = [k for k in systems if k.startswith(("vanilla / shipped", "vanilla / old", "vanilla / main",
                                                 "vanilla / attn", "baseline_retrieval"))]
    ex = []
    for i in pick:
        ex.append(f"**#{i}** EN: {src[i]}  \nGOLD: `{refs[i]}`  ")
        for k in show:
            ex.append(f"{k} (seed 0): `{systems[k][0][i]}`  ")
        ex.append("")
    # 7
    attn_md = ""
    if not a.no_attn_maps and ("vanilla", "attn") in groups and 0 in groups[("vanilla", "attn")]:
        sents = [src[i] for i in pick[:4]]
        paths = attention_maps(groups[("vanilla", "attn")][0], sents, f"{a.out_dir}/attn_vanilla")
        attn_md = "\n".join(f"![attention {p}]({p})" for p in paths)

    md = f"""# Extra analyses

Test set: {N} sentences. Seeds: mean ± std. Word-level numbers use whitespace tokens, case-sensitive.
A "digit-word" is a word containing both a digit and a Latin letter (Arabizi: 3=ع, 7=ح, 9=ق, ...); they are
{dshare:.1%} of reference words.

## 1. Tokenizers on the test references
{tok_md}

## 2. Word-level diagnostics (greedy)
Digit-word recall = clipped matches of reference digit-words / all reference digit-words: the error the
tokenizer fix targets. A model that cannot emit digit-words scores ~0 here whatever its chrF.
{word_md}

## 3. chrF by source length (greedy, seeds pooled)
![length buckets]({a.out_dir}/len_buckets.png)

{bucket_md}

## 4. Beam length-penalty sweep (alpha)
alpha=0.6 is the grid setting; larger alpha penalises short outputs less.
{alpha_md if srows else "(no --alpha sweep records)"}

## 5. Attention (vanilla / attn, seed 0)
Rows = generated target pieces, columns = source pieces in reading order (the encoder reads them reversed).
{attn_md or "(no attention model)"}

## 6. Examples (fixed random sample, numpy seed 7)
""" + "\n".join(ex) + "\n"
    with open(a.md, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)

if __name__ == "__main__":
    main()
