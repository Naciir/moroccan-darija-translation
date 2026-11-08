"""Non-neural reference points for the results table.
  retrieval : for each EN sentence, output the DAR side of the train pair whose lowercased EN word set
              has the highest Jaccard similarity (ties -> lowest train index)
  mostfreq  : the single most frequent train DAR sentence, for every input (ties -> first in train.csv order)
"""
import argparse, sys, os
from collections import Counter, defaultdict
import pandas as pd
from darija_nmt.evaluate import score
from darija_nmt.train import append_jsonl

def retrieval(train_en, train_dar, queries):
    sets = [set(str(s).lower().split()) for s in train_en]
    index = defaultdict(list)
    for i, s in enumerate(sets):
        for w in s:
            index[w].append(i)
    out = []
    for q in queries:
        q = set(str(q).lower().split())
        inter = Counter(i for w in q for i in index.get(w, ()))
        best, best_i = 0.0, 0                      # no shared word at all -> train index 0
        for i in sorted(inter):
            j = inter[i] / len(q | sets[i])
            if j > best:
                best, best_i = j, i
        out.append(str(train_dar[best_i]))
    return out

def most_frequent(sents):
    """Most frequent sentence; ties (train: yallah / wakha / hania, 4 each) -> the first one in file order."""
    sents = [str(s) for s in sents]
    c = Counter(sents)
    return next(s for s in sents if c[s] == max(c.values()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--results", default="results_test.jsonl")
    ap.add_argument("--train_csv", default="data/train.csv")
    ap.add_argument("--suffix", default="",
                    help="e.g. _plus: retrieval over data/train_plus.csv (data-scaling control)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")            # Windows: "▁" etc. must survive redirected stdout
    train, df = pd.read_csv(a.train_csv), pd.read_csv(f"data/{a.split}.csv")
    refs = [str(x) for x in df["dar"]]
    top = most_frequent(train["dar"])
    os.makedirs(a.out_dir, exist_ok=True)
    for name, hyps in [("retrieval", retrieval(train["en"], train["dar"], df["en"])), ("mostfreq", [top] * len(df))]:
        name += a.suffix
        m = score(hyps, refs)
        print(f"baseline_{name} | {a.split} | BLEU {m['bleu']:.2f} | chrF {m['chrf']:.2f} | chrF++ {m['chrfpp']:.2f}")
        with open(f"{a.out_dir}/baseline_{name}_{a.split}.txt", "w", encoding="utf-8") as f:
            f.write("".join(h.replace("\n", " ") + "\n" for h in hyps))
        append_jsonl(a.results, dict(run=f"baseline_{name}", model="baseline", tag=name, split=a.split,
                                     beam=0, n=len(hyps), **m))

if __name__ == "__main__":
    main()
