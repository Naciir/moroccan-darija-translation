"""Tune the beam length-normalisation exponent alpha on the VALIDATION set, then score the test set once.

    python scripts/tune_alpha.py --runs vanilla_attn_plus_s0 --beam 5      # -> results_val.jsonl, results_test.jsonl

Greedy (beam 1) is always a candidate: if no alpha beats it on validation, greedy is kept."""
import argparse, json, subprocess, sys

ALPHAS = [0.0, 0.3, 0.6, 0.8, 1.0, 1.2, 1.5]

def last(path, run, split, beam, alpha):
    best = None
    for l in open(path):
        r = json.loads(l)
        if (r["run"], r["split"], r["beam"]) == (run, split, beam) and (beam == 1 or r.get("alpha") == alpha):
            best = r
    return best

def ev(run, split, beam, alpha, results, runs_dir):
    cmd = [sys.executable, "-m", "darija_nmt.evaluate", "--runs", run, "--runs_dir", runs_dir, "--split", split,
           "--beam", str(beam), "--alpha", str(alpha), "--results", results, "--show", "0"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return last(results, run, split, beam, alpha if beam > 1 else None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--runs_dir", default="checkpoints")
    ap.add_argument("--beam", type=int, default=5)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    for run in a.runs:
        cands = [(ev(run, "val", 1, 0.6, "results_val.jsonl", a.runs_dir)["chrf"], 1, None)]
        for al in ALPHAS:
            cands.append((ev(run, "val", a.beam, al, "results_val.jsonl", a.runs_dir)["chrf"], a.beam, al))
        best = max(cands, key=lambda c: c[0])
        print(run, "val chrF:", ", ".join(f"{'greedy' if b == 1 else f'a={al:g}'} {c:.2f}" for c, b, al in cands))
        t = ev(run, "test", best[1], best[2] if best[2] is not None else 0.6, "results_test.jsonl", a.runs_dir)
        print(f"  -> selected {'greedy' if best[1] == 1 else f'beam {best[1]} alpha {best[2]:g}'}: "
              f"test chrF {t['chrf']:.2f} BLEU {t['bleu']:.2f} len ratio {t['len_ratio']:.3f}", flush=True)
        with open("results_alpha.jsonl", "a") as f:
            f.write(json.dumps(dict(run=run, val=[dict(beam=b, alpha=al, chrf=c) for c, b, al in cands],
                                    selected_beam=best[1], selected_alpha=best[2], test_chrf=t["chrf"],
                                    test_bleu=t["bleu"])) + "\n")

if __name__ == "__main__":
    main()
