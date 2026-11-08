import argparse, sys, json, os
import pandas as pd
import sentencepiece as spm
import torch
from sacrebleu.metrics import BLEU, CHRF

from darija_nmt.cells import convert_legacy_state_dict
from darija_nmt.train import build_model, append_jsonl, EMBED_DIM, HIDDEN_DIM, DEVICE

BOS, EOS, PAD = 2, 3, 0

def load_run(run, runs_dir="checkpoints"):
    """-> (model, cfg, sp_src, sp_tgt). Rebuilds the exact model from <run>.json; runs without one
    are treated as LEGACY (original code: per-gate weights, data/tok_orig tokenizers)."""
    cfg_path = f"{runs_dir}/{run}.json"
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    else:
        cfg = dict(model=run.split("_")[0], tok_dir="data/tok_orig", mask=False, attn=False, no_src=False,
                   embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.0, legacy=True)
    sp_src = spm.SentencePieceProcessor(model_file=f"{cfg['tok_dir']}/spm_en.model")
    sp_tgt = spm.SentencePieceProcessor(model_file=f"{cfg['tok_dir']}/spm_dar.model")
    if cfg.get("direction") == "dar2en":                     # reverse model (back-translation)
        sp_src, sp_tgt = sp_tgt, sp_src
    cfg.setdefault("src_vocab", sp_src.get_piece_size()); cfg.setdefault("tgt_vocab", sp_tgt.get_piece_size())
    # Inference ALWAYS masks pads: a masked padded batch == every sentence encoded alone without pads,
    # which is exactly what the original per-sentence evaluate.py did (also for --no_mask models).
    model = build_model({**cfg, "mask": True}).to(DEVICE)
    sd = torch.load(f"{runs_dir}/{run}.pt", map_location=DEVICE)
    if any(".W_xf." in k for k in sd):
        sd = convert_legacy_state_dict(sd)
    model.load_state_dict(sd)
    model.eval()
    return model, cfg, sp_src, sp_tgt

def src_ids(sp_src, sentence):
    return (sp_src.encode(str(sentence)) + [EOS])[::-1]      # EOS, then reversed (matches training)

def left_pad(seqs):
    m = max(len(s) for s in seqs)
    return torch.tensor([[PAD] * (m - len(s)) + s for s in seqs], dtype=torch.long, device=DEVICE)

def _cut(ids):
    return ids[:ids.index(EOS)] if EOS in ids else ids

@torch.no_grad()
def greedy_one(model, sp_src, sentence, max_len):
    """Reference per-sentence greedy decoding (the original evaluate.py loop)."""
    state, enc_out, mask = model.encode(left_pad([src_ids(sp_src, sentence)]))
    cur, ids = torch.tensor([BOS], device=DEVICE), []
    for _ in range(max_len):
        pred, state = model.decoder.forward_step(cur, state, enc_out, mask)
        cur = pred.argmax(1)
        if cur.item() == EOS:
            break
        ids.append(cur.item())
    return ids

@torch.no_grad()
def greedy_batch(model, sp_src, sentences, max_len, bs=128):
    """Batched greedy: left-padded batches (masked encoder + masked attention), sorted by length."""
    srcs = [src_ids(sp_src, s) for s in sentences]
    order = sorted(range(len(srcs)), key=lambda i: len(srcs[i]))
    out = [None] * len(srcs)
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        state, enc_out, mask = model.encode(left_pad([srcs[i] for i in idx]))
        cur = torch.full((len(idx),), BOS, dtype=torch.long, device=DEVICE)
        done = torch.zeros(len(idx), dtype=torch.bool, device=DEVICE)
        steps = []
        for _ in range(max_len):
            pred, state = model.decoder.forward_step(cur, state, enc_out, mask)
            cur = pred.argmax(1)
            steps.append(cur)
            done |= cur == EOS
            if done.all():
                break
        for j, row in zip(idx, torch.stack(steps, 1).tolist()):
            out[j] = _cut(row)
    return out

@torch.no_grad()
def beam_one(model, sp_src, sentence, k, max_len, alpha=0.6):
    """Beam search for one sentence, the k live beams decoded as one batch. Hypotheses are ranked by
    logprob / lp(n) (GNMT, see lp()), n = scored tokens, the same convention for EOS-finished and
    max_len-cut hypotheses. Search runs until no live beam can still win; the greedy output is always a
    candidate too (so beam >= greedy on the model score). k=1 is exactly greedy."""
    if k <= 1:
        return greedy_one(model, sp_src, sentence, max_len)
    (h, c), enc_out, mask = model.encode(left_pad([src_ids(sp_src, sentence)]))
    seqs, scores = [[]], torch.zeros(1, dtype=torch.float64, device=DEVICE)   # float64: no rounding ties
    cur, finished = torch.tensor([BOS], device=DEVICE), []
    for t in range(1, max_len + 1):                       # t = scored tokens per hypothesis after this step
        n = len(seqs)
        logits, (h, c) = model.decoder.forward_step(cur, (h, c), enc_out.expand(n, -1, -1), mask.expand(n, -1))
        total = scores[:, None] + torch.log_softmax(logits.double(), dim=-1)
        # EVERY live beam may end here: finalise all EOS continuations, not only those ranked in the top-k
        finished += [(s / lp(t, alpha), q) for q, s in zip(seqs, total[:, EOS].tolist())]
        total[:, EOS] = float("-inf")
        top_s, top_i = total.view(-1).topk(k)          # the k best non-EOS continuations stay live
        V = total.size(1)
        idx = torch.div(top_i, V, rounding_mode="floor")
        seqs = [seqs[b] + [i % V] for b, i in zip(idx.tolist(), top_i.tolist())]
        scores, cur, h, c = top_s, top_i % V, h[idx], c[idx]
        if t == max_len:                                # cut at max_len: no EOS, t scored tokens
            finished += [(s / lp(t, alpha), q) for q, s in zip(seqs, scores.tolist())]
        # logprobs only fall and lp(n) <= lp(max_len): no live beam can beat s_max / lp(max_len)
        elif max(f[0] for f in finished) >= scores.max().item() / lp(max_len, alpha):
            break
    # beam pruning can drop the greedy path (a search error, e.g. k=5 loses it, k=8 finds it again):
    # the greedy hypothesis is one more candidate, so beam never scores below greedy on its objective
    g = greedy_one(model, sp_src, sentence, max_len)
    finished.append((hyp_score(model, sp_src, sentence, g, max_len, alpha), g))
    return max(finished, key=lambda x: x[0])[1]

def lp(n, alpha=0.6):
    """GNMT length penalty; n = number of scored tokens (EOS included when it was emitted)."""
    return ((5 + n) / 6) ** alpha

@torch.no_grad()
def hyp_score(model, sp_src, sentence, ids, max_len=48, alpha=0.6):
    """Length-normalised model score of one hypothesis exactly as beam_one ranks it (EOS scored unless
    the hypothesis was cut at max_len). Used by the tests: beam must never lose to greedy."""
    state, enc_out, mask = model.encode(left_pad([src_ids(sp_src, sentence)]))
    toks = ids + ([EOS] if len(ids) < max_len else [])
    cur, s = torch.tensor([BOS], device=DEVICE), 0.0
    for t in toks:
        logits, state = model.decoder.forward_step(cur, state, enc_out, mask)
        s += torch.log_softmax(logits.double(), -1)[0, t].item()
        cur = torch.tensor([t], device=DEVICE)
    return s / lp(len(toks), alpha)

def translate(model, sp_src, sentences, beam, max_len, alpha=0.6):
    if beam <= 1:
        return greedy_batch(model, sp_src, sentences, max_len)
    return [beam_one(model, sp_src, s, beam, max_len, alpha) for s in sentences]

def score(hyps, refs):
    """len_ratio = BLEU sys_len / ref_len: beam with GNMT alpha=0.6 shortens outputs on these models (~0.5 vs
    ~0.9 greedy, some empty), so a beam chrF drop is mostly a brevity effect, not a search bug."""
    b = BLEU().corpus_score(hyps, [refs])
    return dict(bleu=b.score, chrf=CHRF().corpus_score(hyps, [refs]).score,
                chrfpp=CHRF(word_order=2).corpus_score(hyps, [refs]).score,
                len_ratio=b.sys_len / max(b.ref_len, 1), empty=sum(not h.strip() for h in hyps))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="e.g. vanilla_main_s0 cifg_main_s0")
    ap.add_argument("--runs_dir", default="checkpoints")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--beam", type=int, default=1, help="1 = greedy")
    ap.add_argument("--max_len", type=int, default=48)
    ap.add_argument("--alpha", type=float, default=0.6, help="beam length-normalisation exponent (GNMT lp)")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N sentences")
    ap.add_argument("--show", type=int, default=5, help="how many sample translations to print")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--results", default="results_test.jsonl")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")            # Windows: "▁" etc. must survive redirected stdout

    df = pd.read_csv(f"data/{a.split}.csv")
    if a.limit:
        df = df.head(a.limit)
    os.makedirs(a.out_dir, exist_ok=True)

    for run in a.runs:
        if not os.path.exists(f"{a.runs_dir}/{run}.pt"):
            print(f"{a.runs_dir}/{run}.pt not found, skipping"); continue
        model, cfg, sp_src, sp_tgt = load_run(run, a.runs_dir)
        src_col, tgt_col = ("dar", "en") if cfg.get("direction") == "dar2en" else ("en", "dar")
        # the English tokenizer lowercases (nmt_nfkc_cf): reverse-model outputs are scored against lowercased refs
        refs = [str(x).lower() if tgt_col == "en" else str(x) for x in df[tgt_col]]
        hyps = [sp_tgt.decode(ids) for ids in translate(model, sp_src, list(df[src_col]), a.beam, a.max_len, a.alpha)]
        m = score(hyps, refs)
        for i in range(min(a.show, len(df))):
            print(f"  [SRC]  {df.iloc[i][src_col]}\n  [GOLD] {refs[i]}\n  [PRED] {hyps[i]}\n")
        print(f"{run} | {a.split} beam={a.beam} n={len(hyps)} | BLEU {m['bleu']:.2f} | chrF {m['chrf']:.2f} | "
              f"chrF++ {m['chrfpp']:.2f} | len ratio {m['len_ratio']:.3f} | empty {m['empty']}")
        alpha = a.alpha if a.beam > 1 else None            # greedy has no length normalisation
        suffix = ((f"_a{a.alpha:g}" if alpha is not None and alpha != 0.6 else "") +
                  (f"_lim{a.limit}" if a.limit else ""))
        with open(f"{a.out_dir}/{run}_{a.split}_beam{a.beam}{suffix}.txt", "w", encoding="utf-8") as f:
            f.write("".join(h.replace("\n", " ") + "\n" for h in hyps))
        if a.limit:                       # debug subset: never recorded (would shadow the full eval's key)
            print(f"--limit {a.limit}: not appended to {a.results}"); continue
        append_jsonl(a.results, dict(run=run, model=cfg["model"], tag=cfg.get("tag", "legacy"), split=a.split,
                                     beam=a.beam, alpha=alpha, max_len=a.max_len, n=len(hyps),
                                     ckpt_mtime=os.path.getmtime(f"{a.runs_dir}/{run}.pt"), **m))

if __name__ == "__main__":
    main()
