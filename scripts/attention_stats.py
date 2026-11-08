"""How peaked is the attention? For every attention run: mean max weight and mean normalised entropy of the
attention distribution while greedy-decoding test sentences (1 = uniform over the source, 0 = all weight on one
position). Saturated attention (max weight ~1, entropy ~0) cannot learn alignments: this is what breaks LSTMP + dot.

    python scripts/attention_stats.py         # every *attn* run with a checkpoint -> attention_stats.json + table
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import torch
from darija_nmt.evaluate import load_run, left_pad, src_ids, BOS, EOS

@torch.no_grad()
def stats(model, sp_src, sentences, max_len=30):
    dec, mx, ent, norm = model.decoder, [], [], []
    for s in sentences:
        ids = src_ids(sp_src, s)
        if len(ids) < 2:
            continue
        st, enc, mask = model.encode(left_pad([ids]))
        norm.append(enc.norm(dim=-1).mean().item())
        cur = torch.tensor([BOS], device=enc.device)
        for _ in range(max_len):
            h, c = dec.cell(dec.embedding(cur), st); st = (h, c)
            a, out = dec.attend(h, enc, mask)
            cur = dec.fc_out(out).argmax(1)
            mx.append(a.max().item())
            ent.append((-(a * a.clamp_min(1e-12).log()).sum() / np.log(len(ids))).item())
            if cur.item() == EOS:
                break
    return dict(max_weight=float(np.mean(mx)), entropy=float(np.mean(ent)), enc_norm=float(np.mean(norm)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", help="default: every run whose config has attn=true")
    ap.add_argument("--runs_dir", default="checkpoints")
    ap.add_argument("--n", type=int, default=300, help="test sentences (first n)")
    ap.add_argument("--out", default="attention_stats.json")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    runs = a.runs or sorted(f[:-5] for f in os.listdir(a.runs_dir) if f.endswith(".json")
                            and json.load(open(os.path.join(a.runs_dir, f))).get("attn")
                            and json.load(open(os.path.join(a.runs_dir, f))).get("model") != "transformer"
                            and os.path.exists(os.path.join(a.runs_dir, f[:-5] + ".pt")))
    sents = [str(x) for x in pd.read_csv("data/test.csv")["en"].head(a.n)]
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for run in runs:
        model, cfg, sp_src, _ = load_run(run, a.runs_dir)
        if cfg.get("direction") == "dar2en":
            continue
        res[run] = stats(model, sp_src, sents)
        print(f"{run:32s} max weight {res[run]['max_weight']:.3f} | entropy {res[run]['entropy']:.3f} | "
              f"|enc out| {res[run]['enc_norm']:.2f}", flush=True)
        json.dump(res, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
