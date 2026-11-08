"""Blind human evaluation: a sheet for a Darija speaker, then the scores per system.

    python scripts/human_eval.py make --systems retrieval=outputs/baseline_retrieval_test.txt \
        lstm=outputs/vanilla_attn_s0_test_beam1.txt best=outputs/<best>_test_beam1.txt
        -> human_eval/sheet.csv (open in Excel / Google Sheets) + human_eval/key.json (do not open before scoring)
    python scripts/human_eval.py score                  # after filling the "adequacy" and "fluency" columns (1-5)

Each test sentence appears once per system, rows are shuffled so the rater cannot tell which system produced a
line, and duplicate outputs are shown once (their score is copied to every system that produced them).
Adequacy 1-5: how much of the English meaning is in the Darija (5 = all of it). Fluency 1-5: is it natural Darija,
whatever the spelling (5 = a native speaker could have written it)."""
import argparse, csv, json, os, random, sys
import numpy as np
import pandas as pd

DIR = "human_eval"

def make(a):
    df = pd.read_csv("data/test.csv")
    systems = dict(s.split("=", 1) for s in a.systems)
    hyps = {name: [l.rstrip("\n") for l in open(path, encoding="utf-8")] for name, path in systems.items()}
    assert all(len(h) == len(df) for h in hyps.values())
    rng = random.Random(a.seed)
    idx = sorted(rng.sample(range(len(df)), a.n))
    rows, key = [], {}
    for i in idx:
        outputs = {}
        for name, h in hyps.items():
            outputs.setdefault(h[i].strip(), []).append(name)
        for text, names in outputs.items():
            rid = f"r{len(rows):04d}"
            rows.append(dict(id=rid, english=df["en"].iloc[i], reference=df["dar"].iloc[i], translation=text,
                             adequacy="", fluency="", comment=""))
            key[rid] = dict(test_index=int(i), systems=names)
    rng.shuffle(rows)
    os.makedirs(DIR, exist_ok=True)
    with open(f"{DIR}/sheet.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    json.dump(dict(systems=systems, seed=a.seed, n=a.n, rows=key), open(f"{DIR}/key.json", "w"), indent=1)
    print(f"{DIR}/sheet.csv: {len(rows)} lines to rate ({a.n} sentences x {len(systems)} systems, duplicates merged)")

def score(a):
    key = json.load(open(f"{DIR}/key.json"))
    sheet = pd.read_csv(f"{DIR}/sheet.csv", encoding="utf-8-sig")
    per = {s: {"adequacy": [], "fluency": []} for s in key["systems"]}
    for _, r in sheet.iterrows():
        for s in key["rows"][r["id"]]["systems"]:
            for m in ["adequacy", "fluency"]:
                if pd.notna(r[m]) and str(r[m]).strip():
                    per[s][m].append(float(r[m]))
    rng = np.random.RandomState(0)
    print("| system | adequacy (95% CI) | fluency (95% CI) | rated |\n|---|---|---|---|")
    for s, d in per.items():
        cells = []
        for m in ["adequacy", "fluency"]:
            v = np.array(d[m])
            if not len(v):
                cells.append("-"); continue
            boots = [rng.choice(v, len(v)).mean() for _ in range(2000)]
            cells.append(f"{v.mean():.2f} ({np.percentile(boots, 2.5):.2f}-{np.percentile(boots, 97.5):.2f})")
        print(f"| {s} | {cells[0]} | {cells[1]} | {len(d['adequacy'])} |")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make"); m.add_argument("--systems", nargs="+", required=True)
    m.add_argument("--n", type=int, default=50); m.add_argument("--seed", type=int, default=2026)
    sub.add_parser("score")
    a = ap.parse_args()
    make(a) if a.cmd == "make" else score(a)
