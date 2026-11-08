"""Back-translation (Sennrich et al., 2016): synthetic English for monolingual Darija.

    python -m darija_nmt.train --model vanilla --attn --direction dar2en --train_csv data/train_plus.csv --tag rev ...
    python scripts/backtranslate.py --run vanilla_rev_s0   # -> data/train_bt.csv = train_plus.csv + synthetic pairs

The Darija side is real (data/mono_dar.txt: 75k Darija-only sentences of the same corpus, val/test near-duplicates
already removed by fetch_mono_darija.py); the English side is produced by the reverse (Darija -> English) model.
Filters: Darija already in the training data, empty/degenerate outputs (a word repeated 3+ times in a row), length
ratio outside [0.4, 2.5], duplicate pairs, and (again) any pair whose English or Darija side matches or nearly matches
a val/test sentence. A `synthetic` column marks the generated rows (the dataset reads only en/dar)."""
import argparse, re, sys
import pandas as pd
from build_extra_data import key, near_duplicates
from darija_nmt.evaluate import load_run, greedy_batch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a --direction dar2en run")
    ap.add_argument("--runs_dir", default="checkpoints")
    ap.add_argument("--mono", default="data/mono_dar.txt")
    ap.add_argument("--base_csv", default="data/train_plus.csv")
    ap.add_argument("--out", default="data/train_bt.csv")
    ap.add_argument("--max_len", type=int, default=48)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    model, cfg, sp_src, sp_tgt = load_run(a.run, a.runs_dir)
    assert cfg.get("direction") == "dar2en", "need a Darija -> English model"
    base = pd.read_csv(a.base_csv)
    held = pd.concat([pd.read_csv("data/val.csv"), pd.read_csv("data/test.csv")])
    mono = [l.strip() for l in open(a.mono, encoding="utf-8") if l.strip()]
    steps = [("monolingual Darija lines", len(mono))]
    known = set(base["dar"].map(key))
    mono = list(dict.fromkeys(m for m in mono if key(m) and key(m) not in known))
    steps.append(("not already in the training data, unique", len(mono)))
    hyps = [sp_tgt.decode(ids) for ids in greedy_batch(model, sp_src, mono, a.max_len)]
    d = pd.DataFrame({"en": hyps, "dar": mono})
    rep = re.compile(r"\b(\w+)( \1\b){2,}")
    ratio = d["en"].str.split().str.len() / d["dar"].str.split().str.len()
    loops = d["en"].map(lambda s: bool(rep.search(s)))        # python re: pyarrow regexes have no back-references
    d = d[(d["en"].str.strip() != "") & ~loops & ratio.between(0.4, 2.5)]
    steps.append(("non-empty, no repetition loop, length ratio 0.4-2.5", len(d)))
    d = d[~d["en"].map(key).isin(set(held["en"].map(key))) & ~d["dar"].map(key).isin(set(held["dar"].map(key)))]
    d = d[~near_duplicates(d.assign(ek=d["en"].map(key), dk=d["dar"].map(key)), held)]
    steps.append(("no exact/near duplicate of a val/test sentence", len(d)))
    d = d.drop_duplicates(["en", "dar"])
    steps.append(("unique pairs", len(d)))
    out = pd.concat([base[["en", "dar"]].assign(synthetic=0), d.assign(synthetic=1)], ignore_index=True)
    out.to_csv(a.out, index=False)
    for name, n in steps:
        print(f"  {name:55s} {n:,}")
    print(f"{a.out}: {len(base):,} real + {len(d):,} synthetic = {len(out):,} pairs")
    print(d.sample(min(8, len(d)), random_state=0).to_string())

if __name__ == "__main__":
    main()
