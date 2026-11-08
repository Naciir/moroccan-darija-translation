"""Data-scaling experiment: data/train_plus.csv = data/train.csv + extra EN-Darija pairs from DODa.

Source: HF `atlasia/darija_english`, config `doda` (Darija Open Dataset, Latin script, 45k pairs). It is a
superset of our data (99.4% of our train/val/test pairs are in it), so leakage control is the whole point:
  - drop every extra row whose English OR Darija side matches ANY val/test sentence
    (lowercased, whitespace-collapsed, punctuation removed) -> val/test stay unseen, exactly as before;
  - drop exact-duplicate pairs and pairs already in train;
  - drop mojibake (Cyrillic/replacement characters) and empty or > 40-word sides.
val.csv / test.csv and the tokenizers are NOT changed, so test scores stay comparable with the main grid.

    python scripts/build_extra_data.py            # downloads the parquet (~2 MB) once into data/extra/
"""
import os, re, sys, urllib.request
import pandas as pd

URL = "https://huggingface.co/api/datasets/atlasia/darija_english/parquet/doda/train/0.parquet"
RAW = "data/extra/doda.parquet"
BAD = re.compile("[Ѐ-ӿ�؀-ۿ]")       # Cyrillic mojibake, replacement char, Arabic script

def key(s):
    return re.sub(r"[^\w\s]", "", re.sub(r"\s+", " ", str(s).lower().strip())).strip()

def near_duplicates(d, held, thr=0.8):
    """True where the EN word set (Jaccard) OR the Darija character-5-gram set (Jaccard) of an extra row is >= thr
    similar to ANY val/test sentence: catches typo / spelling-variant / one-word-different copies."""
    from collections import Counter, defaultdict
    def grams(s, n=5):
        s = f" {key(s)} "
        return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}
    flags = pd.Series(False, index=d.index)
    for col, feat in [("en", lambda s: set(key(s).split())), ("dar", grams)]:
        hs = [feat(x) for x in held[col]]
        index = defaultdict(list)
        for i, f in enumerate(hs):
            for g in f:
                index[g].append(i)
        for r, x in zip(d.index, d[col]):
            f = feat(x)
            inter = Counter(i for g in f for i in index.get(g, ()))
            if any(c / len(f | hs[i]) >= thr for i, c in inter.items()):
                flags[r] = True
    return flags

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if not os.path.exists(RAW):
        os.makedirs(os.path.dirname(RAW), exist_ok=True)
        urllib.request.urlretrieve(URL, RAW)
    tr, va, te = (pd.read_csv(f"data/{s}.csv") for s in ["train", "val", "test"])
    d = pd.read_parquet(RAW).rename(columns={"darija": "dar"})[["en", "dar"]].dropna()
    d["en"], d["dar"] = d["en"].astype(str).str.strip(), d["dar"].astype(str).str.strip()
    n0 = len(d)
    held_en = set(va["en"].map(key)) | set(te["en"].map(key))
    held_dar = set(va["dar"].map(key)) | set(te["dar"].map(key))
    d["ek"], d["dk"] = d["en"].map(key), d["dar"].map(key)
    steps = []
    d = d[(d.ek != "") & (d.dk != "")]; steps.append(("non-empty", len(d)))
    d = d[~d.ek.isin(held_en) & ~d.dk.isin(held_dar)]; steps.append(("no EN/DAR match with val/test", len(d)))
    d = d[~d.en.str.contains(BAD) & ~d.dar.str.contains(BAD)]; steps.append(("no mojibake / Arabic script", len(d)))
    d = d[(d.en.str.split().str.len() <= 40) & (d.dar.str.split().str.len() <= 40)]
    steps.append(("<= 40 words", len(d)))
    d = d.drop_duplicates(["ek", "dk"]); steps.append(("unique pairs", len(d)))
    near = near_duplicates(d, pd.concat([va, te]))
    d = d[~near]; steps.append(("no near-duplicate of val/test", len(d)))
    in_train = set(tr["en"].map(key) + "|" + tr["dar"].map(key))
    d = d[~(d.ek + "|" + d.dk).isin(in_train)]; steps.append(("not already in train", len(d)))
    # final leakage assertion
    assert not (set(d.ek) & held_en) and not (set(d.dk) & held_dar)
    out = pd.concat([tr[["en", "dar"]], d[["en", "dar"]]], ignore_index=True)
    out.to_csv("data/train_plus.csv", index=False)
    print(f"DODa rows {n0:,}")
    for name, n in steps:
        print(f"  {name:32s} {n:,}")
    print(f"data/train_plus.csv: {len(tr):,} train + {len(d):,} extra = {len(out):,} pairs")

if __name__ == "__main__":
    main()
