"""Report figures -> figures/*.png (run after the grid; used by build_docs.py).
    python scripts/figures.py"""
import json, os, re, sys
from collections import defaultdict
import numpy as np
import pandas as pd
import sentencepiece as spm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figures"
CELLS = ["vanilla", "cifg", "lstmp"]                 # the three cells of the original study (shipped figures)
CELLS4 = ["vanilla", "cifg", "lstmp", "wmc"]
CELL_LABEL = {"vanilla": "Vanilla LSTM", "cifg": "CIFG", "lstmp": "LSTMP", "wmc": "WMC"}
COLORS = {"vanilla": "#1f5fbf", "cifg": "#d9822b", "lstmp": "#2e8b57", "wmc": "#8e44ad"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})

def read_jsonl(p):
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]

def load():
    train = {r["run"]: r for r in read_jsonl("results.jsonl")}
    test = {(r["run"], r["split"], r["beam"]): r for r in read_jsonl("results_test.jsonl")
            if r.get("alpha") in (None, 0.6)}
    groups = defaultdict(dict)
    for run, r in train.items():
        groups[(r["model"], r["tag"])][r["seed"]] = run
    return train, test, groups

def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout(); fig.savefig(f"{OUT}/{name}.png", dpi=200); plt.close(fig)
    print("wrote", f"{OUT}/{name}.png")

def fig_eda():
    tr = pd.read_csv("data/train.csv")
    en = tr["en"].astype(str).str.split().str.len(); da = tr["dar"].astype(str).str.split().str.len()
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
    bins = np.arange(0, 31) + 0.5
    ax[0].hist(en.clip(upper=30), bins=bins, alpha=0.7, label=f"English (mean {en.mean():.1f})", color="#555")
    ax[0].hist(da.clip(upper=30), bins=bins, alpha=0.7, label=f"Darija (mean {da.mean():.1f})", color="#1f5fbf")
    ax[0].set_xlabel("words per sentence (train)"); ax[0].set_ylabel("sentences"); ax[0].legend(fontsize=7)
    ax[0].set_title("Sentence length", fontsize=9)
    # tokenizer comparison on the test references
    refs = [str(x) for x in pd.read_csv("data/test.csv")["dar"]]
    dwords = [w for r in refs for w in r.split() if re.search(r"\d", w) and re.search(r"[A-Za-z]", w)]
    names, tpw, whole = [], [], []
    for name, d in [("original\n(split_by_number=True)", "data/tok_orig"), ("fixed\n(split_by_number=False)", "data")]:
        sp = spm.SentencePieceProcessor(model_file=f"{d}/spm_dar.model")
        names.append(name)
        tpw.append(sum(len(sp.encode(r)) for r in refs) / sum(len(r.split()) for r in refs))
        whole.append(100 * np.mean([len(sp.encode(w)) == 1 for w in dwords]))
    x = np.arange(2)
    b = ax[1].bar(x - 0.2, tpw, 0.4, color="#888", label="BPE pieces per word")
    ax2 = ax[1].twinx(); ax2.grid(False)
    b2 = ax2.bar(x + 0.2, whole, 0.4, color="#1f5fbf", label="digit-words kept whole (%)")
    for rect, v in zip(b, tpw):
        ax[1].text(rect.get_x() + rect.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    for rect, v in zip(b2, whole):
        ax2.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=7)
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, fontsize=7); ax[1].set_ylabel("pieces / word")
    ax2.set_ylabel("% of digit-words"); ax2.set_ylim(0, 80); ax2.spines["right"].set_visible(True)
    ax[1].set_ylim(0, 2.8)
    ax[1].set_title("Darija tokenizer (test refs)", fontsize=9)
    ax[1].legend(handles=[b, b2], fontsize=6.5, loc="upper center", ncol=2, frameon=False)
    save(fig, "eda")

def seed_scores(test, groups, model, tag, metric="chrf", beam=1):
    return [test[(groups[(model, tag)][s], "test", beam)][metric] for s in sorted(groups.get((model, tag), {}))
            if (groups[(model, tag)][s], "test", beam) in test]

def fig_cells(train, test, groups, legacy=(17.10, 17.13, 17.45), retrieval=None):
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    settings = [("shipped", None), ("main", "no attention (fixed)"), ("attn", "+ Luong attention")]
    x0 = 0
    ticks, labels = [], []
    for k, (tag, lab) in enumerate(settings):
        for j, m in enumerate(CELLS):
            xs = x0 + j * 0.27
            if tag == "shipped":
                v = [legacy[j]]
            else:
                v = seed_scores(test, groups, m, tag)
            if not v:
                continue
            ax.bar(xs, np.mean(v), 0.25, color=COLORS[m], alpha=0.85, label=CELL_LABEL[m] if k == 1 else None)
            ax.scatter([xs] * len(v), v, color="k", s=6, zorder=3)
            ax.text(xs, np.mean(v) + 0.4, f"{np.mean(v):.1f}", ha="center", fontsize=6.5)
        ticks.append(x0 + 0.27); labels.append(lab or "shipped\n(1 seed, 20 ep)")
        x0 += 1.1
    if retrieval:
        ax.axhline(retrieval, ls="--", color="#c0392b", lw=1)
        ax.text(x0 - 0.35, retrieval + 0.3, f"retrieval baseline {retrieval:.1f}", color="#c0392b", fontsize=7,
                ha="right")
    ax.set_xticks(ticks); ax.set_xticklabels(labels); ax.set_ylabel("test chrF (greedy)")
    ax.set_ylim(0, None); ax.legend(fontsize=7, ncol=3, loc="upper left")
    save(fig, "cells")

def fig_curves(train, groups):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for a, tag, title in [(ax[0], "main", "no attention"), (ax[1], "attn", "Luong attention")]:
        for m in CELLS:
            for j, s in enumerate(sorted(groups.get((m, tag), {}))):
                r = train[groups[(m, tag)][s]]
                a.plot(range(1, len(r["val_losses"]) + 1), r["val_losses"], color=COLORS[m], lw=1, alpha=0.8,
                       label=CELL_LABEL[m] if j == 0 else None)
                if "train_losses" in r:
                    a.plot(range(1, len(r["train_losses"]) + 1), r["train_losses"], color=COLORS[m], lw=0.7,
                           ls=":", alpha=0.6)
        a.set_title(f"{title} (solid = val, dotted = train)", fontsize=8.5); a.set_xlabel("epoch")
    ax[0].set_ylabel("cross-entropy / BPE token"); ax[0].legend(fontsize=7)
    save(fig, "curves")

HP = [("lr3e-4", "lr 3e-4"), ("lr3e-3", "lr 3e-3"), ("adamw", "AdamW"), ("rmsprop", "RMSprop"),
      ("sgd", "SGD+mom. (lr 0.05)"), ("xavier", "Xavier init"), ("orthogonal", "orthogonal init"),
      ("bs32", "batch 32"), ("bs128", "batch 128"), ("do0.2", "dropout 0.2"), ("do0.5", "dropout 0.5")]

def fig_hp(train, test, groups):
    base = seed_scores(test, groups, "vanilla", "hp_base")
    if not base:
        return
    bm, bs = np.mean(base), (np.std(base, ddof=1) if len(base) > 1 else 0.0)
    labs, vals = [], []
    for k, lab in HP:
        v = seed_scores(test, groups, "vanilla", f"hp_{k}")
        if v:
            labs.append(lab); vals.append(np.mean(v))
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    x = np.arange(len(vals))
    ax.axhspan(bm - 2 * bs, bm + 2 * bs, color="#999", alpha=0.25, lw=0, label=f"base ± 2 std ({len(base)} seeds)")
    ax.axhline(bm, color="#555", lw=1)
    ax.bar(x, vals, 0.6, color=["#1f5fbf" if v > bm + 2 * bs else "#c0392b" if v < bm - 2 * bs else "#9aa7b8"
                                 for v in vals])
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.2, f"{v:.1f}", ha="center", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(labs, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("test chrF"); ax.set_ylim(max(0, min(vals + [bm]) - 4), max(vals + [bm]) + 1.5)
    ax.set_title(f"One change at a time vs base (Adam 1e-3, default init, batch 64, dropout 0.4): {bm:.1f} chrF",
                 fontsize=8.5)
    ax.legend(fontsize=7, loc="lower left")
    save(fig, "hp")

def fig_scaling(train, groups):
    pairs = [(("vanilla", "attn"), ("vanilla", "attn_plus"), "vanilla + attention"),
             (("vanilla", "main"), ("vanilla", "plus"), "vanilla, no attention")]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for k, (a, b, lab) in enumerate(pairs):
        for key, ls, n in [(a, "-", "10k"), (b, "--", "42k")]:
            g = groups.get(key, {})
            if 0 in g:
                vl = train[g[0]]["val_losses"]
                ax.plot(range(1, len(vl) + 1), vl, ls=ls, color=f"C{k}", label=f"{lab}, {n} pairs")
    ax.set_xlabel("epoch"); ax.set_ylabel("val loss (seed 0)"); ax.legend(fontsize=6.5)
    save(fig, "scaling")

def grouped(ax, test, groups, settings, retrieval=None, cells=CELLS4, w=0.19):
    """bars = mean over seeds, dots = seeds; settings = [(tag, label)]"""
    ticks = []
    for k, (tag, _) in enumerate(settings):
        x0 = k * (len(cells) * w + 0.35)
        for j, m in enumerate(cells):
            v = seed_scores(test, groups, m, tag)
            if not v:
                continue
            xs = x0 + j * w
            ax.bar(xs, np.mean(v), w * 0.92, color=COLORS[m], alpha=0.85, label=CELL_LABEL[m] if k == 0 else None)
            ax.scatter([xs] * len(v), v, color="k", s=5, zorder=3)
            ax.text(xs, max(v) + 0.25, f"{np.mean(v):.1f}", ha="center", fontsize=6)
        ticks.append(x0 + w * (len(cells) - 1) / 2)
    ax.set_xticks(ticks); ax.set_xticklabels([lab for _, lab in settings], fontsize=7.5)
    if retrieval:
        ax.axhline(retrieval, ls="--", color="#c0392b", lw=1)
        ax.text(ax.get_xlim()[1], retrieval + 0.2, f"retrieval baseline {retrieval:.1f}", color="#c0392b",
                fontsize=6.5, ha="right")
    ax.set_ylabel("test chrF (greedy)")

def fig_hero(train, test, groups, retrieval=None):
    """README headline: the four cells without attention, with dot / additive attention, and on 4x the data."""
    settings = [("main", "no attention"), ("attn", "dot attention"), ("attn_additive", "additive attention")]
    if any(seed_scores(test, groups, m, "c42") for m in CELLS4):
        settings.append(("c42", "additive attention\n+ 4x data (42k pairs)"))
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    grouped(ax, test, groups, settings, retrieval)
    lo = min(min(seed_scores(test, groups, m, t) or [99]) for m in CELLS4 for t, _ in settings)
    ax.set_ylim(max(0, lo - 3), None); ax.legend(fontsize=7, ncol=4, loc="upper left")
    ax.set_title("English → Darija, four hand-written LSTM cells (dots = seeds)", fontsize=9)
    save(fig, "hero")

def fig_attn_types(train, test, groups, retrieval=None):
    settings = [("attn", "dot"), ("attn_scaled", "scaled dot (1/√d)"), ("attn_general", "general (bilinear)"),
                ("attn_additive", "additive")]
    fig, ax = plt.subplots(figsize=(7.6, 2.8))
    grouped(ax, test, groups, settings, retrieval)
    ax.set_ylim(14, None); ax.legend(fontsize=7, ncol=4, loc="upper left")
    ax.set_title("Attention score function × cell (10k pairs, 3 seeds; scaled dot run for vanilla and LSTMP only)",
                 fontsize=8.5)
    save(fig, "attention_types")

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    train, test, groups = load()
    ret = test.get(("baseline_retrieval", "test", 0))
    fig_eda()
    fig_cells(train, test, groups, retrieval=ret["chrf"] if ret else None)
    fig_curves(train, groups)
    fig_hp(train, test, groups)
    fig_scaling(train, groups)
    fig_hero(train, test, groups, retrieval=ret["chrf"] if ret else None)
    fig_attn_types(train, test, groups, retrieval=ret["chrf"] if ret else None)
    import shutil
    if os.path.exists("outputs/attn_vanilla_0.png"):
        shutil.copy("outputs/attn_vanilla_0.png", f"{OUT}/attention_example.png")

if __name__ == "__main__":
    main()
