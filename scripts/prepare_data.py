"""Rebuild data/{train,val,test}.csv and data/train.{en,dar}.txt from Hugging Face (= notebooks/darija_data_prep.ipynb,
sections 2-10, without the EDA) and check them byte for byte against the files the experiments used.

Source: HF `imomayiz/darija-english`, config "sentences" (87,785 rows; 12,743 have an English side).
  1. keep (eng, darija) -> (en, dar); drop NaN; collapse whitespace; drop empty sides, duplicate pairs and pairs whose
     two sides are equal up to case                                                       -> 12,650 pairs
  2. "half" sample: groupby(script).sample(frac=1, random_state=42) = a seeded shuffle (FRACTION = 1, all Latin)
  3. GROUPED 80/10/10 split on the lowercased English side (a sentence and its case variants never straddle splits):
     unique keys permuted with RandomState(42); first 10% -> val, next 10% -> test, rest -> train
  4. train.{en,dar}.txt = one training sentence per line (the SentencePiece input, see retrain_tokenizers.py)
The files were written on Windows, so lines end in CRLF; this script writes CRLF on every OS to stay byte-identical.

    python scripts/prepare_data.py                     # -> data/, then sha256 check (exit 1 on any mismatch)
    python scripts/prepare_data.py --out_dir /tmp/d    # rebuild elsewhere, same check
"""
import argparse, hashlib, io, json, os, re, sys, urllib.request
import numpy as np
import pandas as pd

API = "https://huggingface.co/api/datasets/imomayiz/darija-english/parquet"
SEED, FRACTION = 42, 1
EN_COL, DAR_COL = "eng", "darija"
EOL = "\r\n"
SHA256 = {  # the splits every reported number was computed on
    "train.csv": "23a0864d73112819dc0ccd3d5fc7ecd3ee7155dad7726eb1d59fae9da3d97b18",
    "val.csv": "e0d58a3b52182fafae9c3933ae45e8249577335fd037a0f28bd35655a0735067",
    "test.csv": "148d1e87e1dfb291eee8f2fb44f0ce183a87260620d0272edff414286a65e903",
    "train.en.txt": "62bb90a3cda685193d65633e648633591107cc7408d3c4cef6af507e8e01a81b",
    "train.dar.txt": "dec115958d29c31809a8d2bb2c6cd6c413f14e4ab316719f76a76c1e5fb7fcf0",
}
AR, LAT = re.compile(r"[؀-ۿ]"), re.compile(r"[A-Za-z]")

def download():
    urls = json.load(urllib.request.urlopen(API, timeout=60))["sentences"]["sentences"]
    return pd.concat([pd.read_parquet(io.BytesIO(urllib.request.urlopen(u, timeout=300).read())) for u in urls],
                     ignore_index=True)

def script_of(s):
    a, l = len(AR.findall(s)), len(LAT.findall(s))
    if a + l == 0:
        return "other"
    return "arabic" if a >= l else "latin"

def clean(df):
    df = df[[EN_COL, DAR_COL]].rename(columns={EN_COL: "en", DAR_COL: "dar"})
    n0 = len(df)
    df = df.dropna()
    for c in ["en", "dar"]:
        df[c] = df[c].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[(df["en"] != "") & (df["dar"] != "")]
    n1 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n2 = len(df)
    same = df["en"].str.lower() == df["dar"].str.lower()
    df = df[~same].reset_index(drop=True)
    print(f"start: {n0} | after empty/NaN: {n1} | after duplicates: {n2} | after identical pairs: {len(df)}")
    df["script"] = df["dar"].map(script_of)
    df["en_len"] = df["en"].str.split().str.len()
    df["dar_len"] = df["dar"].str.split().str.len()
    return df

def split(df):
    np.random.seed(SEED)                                             # notebook cell 4 (no effect on the split itself)
    half = df.groupby("script", group_keys=False).sample(frac=FRACTION, random_state=SEED).reset_index(drop=True)
    key = half["en"].str.lower()
    g = np.random.RandomState(SEED).permutation(key.unique())
    k = len(g) // 10
    part = {s: "val" for s in g[:k]} | {s: "test" for s in g[k:2 * k]}
    sp = key.map(lambda s: part.get(s, "train"))
    return {s: half[sp == s].reset_index(drop=True) for s in ["train", "val", "test"]}

def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--parquet", default=None, help="local copy of the HF parquet (default: download)")
    ap.add_argument("--no_check", action="store_true", help="skip the sha256 comparison")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(a.out_dir, exist_ok=True)
    raw = pd.read_parquet(a.parquet) if a.parquet else download()
    parts = split(clean(raw))
    for name, d in parts.items():
        d.to_csv(os.path.join(a.out_dir, f"{name}.csv"), index=False, lineterminator=EOL)
        print(name, len(d))
    for col in ["en", "dar"]:
        with open(os.path.join(a.out_dir, f"train.{col}.txt"), "w", encoding="utf-8", newline="") as f:
            for s in parts["train"][col]:
                f.write(s.replace("\n", " ") + EOL)
    if a.no_check:
        return
    bad = [(n, sha(os.path.join(a.out_dir, n))) for n, h in SHA256.items() if sha(os.path.join(a.out_dir, n)) != h]
    for n, h in SHA256.items():
        print(f"  {n:14s} {'MISMATCH' if n in dict(bad) else 'ok'}  {h[:16]}")
    if bad:
        sys.exit(f"ERROR: {len(bad)} file(s) differ from the published splits: {[n for n, _ in bad]} "
                 "(upstream dataset or pandas/numpy sampling changed) -> results are NOT comparable")
    print("all files match the published sha256 hashes")

if __name__ == "__main__":
    main()
