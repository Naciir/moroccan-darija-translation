"""Optional ablation: extra MONOLINGUAL Darija (Latin) text for the DAR tokenizer only.

Takes the rows of imomayiz/darija-english ("sentences" config) that have NO English translation
(so they could not be used as parallel data), cleans them exactly like notebook cell 10, and drops
any sentence that appears in val.csv / test.csv, also up to case/punctuation (no leakage). Writes data/mono_dar.txt.
Needs pandas + pyarrow (no `datasets` lib). Then:
    python scripts/retrain_tokenizers.py --out_dir data/tok_mono --dar_extra data/mono_dar.txt
"""
import io, sys
import pandas as pd
import requests

API = "https://huggingface.co/api/datasets/imomayiz/darija-english/parquet"

def main(out="data/mono_dar.txt"):
    sys.stdout.reconfigure(encoding="utf-8")
    urls = requests.get(API, timeout=60).json()["sentences"]["sentences"]
    df = pd.concat([pd.read_parquet(io.BytesIO(requests.get(u, timeout=300).content)) for u in urls])
    print("downloaded rows:", len(df), "| columns:", df.columns.tolist())

    eng = df["eng"].fillna("").astype(str).str.strip()
    dar = df.loc[eng == "", "darija"].dropna().astype(str)
    n0 = len(dar)
    dar = dar.str.replace(r"\s+", " ", regex=True).str.strip()     # same cleaning as notebook cell 10
    dar = dar[dar != ""].drop_duplicates()
    n1 = len(dar)
    # strict held-out filter: drop exact AND near duplicates (same key after lowercasing + removing punctuation)
    key = lambda s: (s.str.lower().str.replace(r"[^\w\s]", "", regex=True)
                     .str.replace(r"\s+", " ", regex=True).str.strip())
    held = pd.concat([pd.read_csv(f"data/{s}.csv")["dar"] for s in ["val", "test"]]).astype(str).str.strip()
    dar = dar[~dar.isin(set(held)) & ~key(dar).isin(set(key(held)))]
    print(f"rows without English: {n0} | after clean+dedup: {n1} | "
          f"after dropping val/test (near-)duplicates: {len(dar)}")

    with open(out, "w", encoding="utf-8") as f:
        for s in dar:
            f.write(s + "\n")
    print("wrote", out)

if __name__ == "__main__":
    main(*sys.argv[1:])
