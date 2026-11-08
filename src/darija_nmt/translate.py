"""Translate English to Darija (Latin script) with a trained run. Run from the repository root (the run's config
points at its tokenizers with a relative path, e.g. data/spm_en.model).

    python -m darija_nmt.translate "where are you?"
    python -m darija_nmt.translate "how are you?" "thank you" --run vanilla_attn_s0 --beam 5
    echo "good morning" | python -m darija_nmt.translate --run vanilla_attn_s0      # one sentence per stdin line
"""
import argparse, sys

from darija_nmt.evaluate import load_run, translate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*", help="English sentences (default: read one per line from stdin)")
    ap.add_argument("--run", default="vanilla_attn_s0", help="checkpoint name: <runs_dir>/<run>.pt (+ .json)")
    ap.add_argument("--runs_dir", default="checkpoints")
    ap.add_argument("--beam", type=int, default=1, help="1 = greedy")
    ap.add_argument("--max_len", type=int, default=48)
    ap.add_argument("--alpha", type=float, default=0.6, help="beam length-normalisation exponent (GNMT lp)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")            # Windows: Darija output must survive redirected stdout
    sents = a.text or [l.strip() for l in sys.stdin if l.strip()]
    if not sents:
        ap.error("no input: pass sentences as arguments or on stdin")
    model, cfg, sp_src, sp_tgt = load_run(a.run, a.runs_dir)
    for ids in translate(model, sp_src, sents, a.beam, a.max_len, a.alpha):
        print(sp_tgt.decode(ids))

if __name__ == "__main__":
    main()
