import argparse, json, os, random, time
import numpy as np
import sentencepiece as spm
import torch, torch.nn as nn
from torch.utils.data import DataLoader

from darija_nmt.data import TranslationDataset, pad_collate_fn, BucketBatchSampler
from darija_nmt.cells import VanillaLSTMCell, CIFGCell, LSTMPCell, WMCCell
from darija_nmt.models import Encoder, Decoder, Seq2Seq, init_weights

EMBED_DIM, HIDDEN_DIM, PROJ_DIM = 128, 256, 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CELLS = {"vanilla": (VanillaLSTMCell, None), "cifg": (CIFGCell, None), "lstmp": (LSTMPCell, PROJ_DIM),
         "wmc": (WMCCell, None)}
OPTS = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "rmsprop": torch.optim.RMSprop,
        "sgd": lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9)}

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def append_jsonl(path, rec, stale_s=30):
    """Append one JSON line; a lock file stops parallel runs (run_experiments --jobs N) clobbering each other.
    A lock older than stale_s seconds was left by a killed writer (an append takes ms): it is removed."""
    lock, fd = path + ".lock", None
    for _ in range(600):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); break
        except (FileExistsError, PermissionError):     # Windows: a delete-pending lock raises EACCES
            try:
                if time.time() - os.path.getmtime(lock) > stale_s:
                    os.remove(lock); continue
            except OSError:
                pass                                # released / removed by another writer meanwhile
            time.sleep(0.1)
    try:
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    finally:
        if fd is not None:
            os.close(fd); os.remove(lock)

def build_model(cfg):
    """Rebuild a model from a run config (also used by evaluate.py). Keys added later default to the old behaviour."""
    if cfg.get("model") == "transformer":
        from darija_nmt.transformer import TransformerSeq2Seq
        return TransformerSeq2Seq(cfg["src_vocab"], cfg["tgt_vocab"], dropout=cfg["dropout"])
    cell_cls, p_dim = CELLS[cfg["model"]]
    enc = Encoder(cfg["src_vocab"], cfg["embed_dim"], cfg["hidden_dim"], cell_cls=cell_cls, proj_dim=p_dim,
                  dropout=cfg["dropout"], use_mask=cfg["mask"], bidir=cfg.get("bidir", False))
    dec = Decoder(cfg["tgt_vocab"], cfg["embed_dim"], cfg["hidden_dim"], cell_cls=cell_cls, proj_dim=p_dim,
                  dropout=cfg["dropout"], attn=cfg["attn"],
                  attn_scale=cfg.get("attn_scale", False), attn_type=cfg.get("attn_type", "dot"))
    return Seq2Seq(enc, dec, no_src=cfg["no_src"])

def run_epoch(model, loader, criterion, tgt_v, optimizer=None, tf=1.0, clip=1.0, report=None):
    """criterion is optimised; `report` (default: criterion) is the loss that is averaged and returned, so a
    label-smoothed training objective can still report / early-stop on plain cross-entropy."""
    report = report or criterion
    train = optimizer is not None
    model.train(train)
    total, ntok = 0.0, 0     # token-weighted mean: independent of how the batches are formed (bucketed or not)
    with torch.set_grad_enabled(train):
        for src, tgt, _, _ in loader:
            # reverse the source; the right-padding becomes LEFT padding, which the masked encoder
            # skips (state stays at zeros until the first real token) -> same as inference, no pads
            src = torch.flip(src, dims=[1]).to(DEVICE); tgt = tgt.to(DEVICE)
            out = model(src, tgt, teacher_forcing_ratio=tf)
            logits, gold = out[:, 1:].reshape(-1, tgt_v), tgt[:, 1:].reshape(-1)
            loss = criterion(logits, gold)
            if train:
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip); optimizer.step()
            n = int((tgt[:, 1:] != 0).sum())
            total += (loss if report is criterion else report(logits.detach(), gold)).item() * n; ntok += n
    return total / ntok

def make_loaders(a, train_ds, val_ds, seed):
    g = torch.Generator().manual_seed(seed)
    if a.bucket:
        train_loader = DataLoader(train_ds,
                                  batch_sampler=BucketBatchSampler(train_ds.src_lens, a.bs, shuffle=True, generator=g),
                                  collate_fn=pad_collate_fn)
    else:
        train_loader = DataLoader(train_ds, batch_size=a.bs, shuffle=True, generator=g, collate_fn=pad_collate_fn)
    # val uses the SAME batching regime as train (fixed order): for --no_mask runs the pad exposure, and so
    # the val loss that drives early stopping / LR halving / checkpoint choice, matches training.
    if a.bucket:
        val_loader = DataLoader(val_ds, batch_sampler=BucketBatchSampler(val_ds.src_lens, a.bs, shuffle=False),
                                collate_fn=pad_collate_fn)
    else:
        val_loader = DataLoader(val_ds, batch_size=a.bs, shuffle=False, collate_fn=pad_collate_fn)
    return train_loader, val_loader

def train_model(model_name, a, train_ds, val_ds, src_v, tgt_v):
    set_seed(a.seed)
    run = f"{model_name}_{a.tag}_s{a.seed}"
    cfg = dict(model=model_name, run=run, tok_dir=a.tok_dir, mask=not a.no_mask, attn=a.attn, no_src=a.no_src,
               bucket=a.bucket, val_bucket=a.bucket, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
               proj_dim=CELLS.get(model_name, (None, None))[1], dropout=a.dropout, src_vocab=src_v, tgt_vocab=tgt_v,
               seed=a.seed, tag=a.tag, train_csv=a.train_csv, attn_scale=a.attn_scale,
               attn_type=a.attn_type, bidir=a.bidir, direction=a.direction, label_smoothing=a.ls, bpe_alpha=a.bpe_alpha)
    if a.limit_train:                     # recorded only when used: full runs keep exactly the original config / record
        cfg["limit_train"] = a.limit_train
    model = build_model(cfg).to(DEVICE)
    if model_name != "transformer":
        init_weights(model, a.init)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== {run} | {n_params:,} params | {DEVICE} | tok={a.tok_dir} mask={cfg['mask']} "
          f"attn={a.attn} no_src={a.no_src} bucket={a.bucket} ===", flush=True)
    os.makedirs(a.runs_dir, exist_ok=True)
    with open(f"{a.runs_dir}/{run}.json", "w") as f:
        json.dump(cfg, f, indent=1)

    train_loader, val_loader = make_loaders(a, train_ds, val_ds, a.seed)
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=a.ls)    # what is optimised
    plain = nn.CrossEntropyLoss(ignore_index=0)                               # what is reported / early-stopped on
    optimizer = OPTS[a.opt](model.parameters(), lr=a.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    best, best_ep, bad, t0 = float("inf"), 0, 0, time.time()
    tls, vls = [], []
    for ep in range(1, a.epochs + 1):
        te = time.time()
        tl = run_epoch(model, train_loader, criterion, tgt_v, optimizer, tf=a.tf, report=plain)
        vl = run_epoch(model, val_loader, plain, tgt_v, tf=1.0)      # teacher-forced val loss per target token
        scheduler.step(vl)
        tls.append(tl); vls.append(vl)
        print(f"Epoch {ep:02d}/{a.epochs} | Train {tl:.4f} | Val {vl:.4f} (ppl {np.exp(vl):.1f}) "
              f"| {time.time() - te:.1f}s", flush=True)
        if vl < best:
            best, best_ep, bad = vl, ep, 0
            torch.save(model.state_dict(), f"{a.runs_dir}/{run}.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop"); break

    peak = torch.cuda.max_memory_allocated() / 2**20 if DEVICE.type == "cuda" else 0.0
    args = {k: v for k, v in vars(a).items() if k != "limit_train"}     # (limit_train: in cfg when used)
    rec = {**args, **cfg, "params": n_params, "train_losses": tls, "val_losses": vls,
           "best_val_loss": best, "best_epoch": best_ep, "epochs_run": len(vls),
           "minutes": (time.time() - t0) / 60, "sec_per_epoch": (time.time() - t0) / len(vls), "peak_gpu_mb": peak}
    with open(f"{a.runs_dir}/{run}.json", "w") as f:      # config + summary (evaluate.py rebuilds from it)
        json.dump(rec, f, indent=1)
    append_jsonl(a.results, rec)
    print(f"done {run}: best val {best:.4f} @ epoch {best_ep} | {rec['minutes']:.1f} min | peak GPU {peak:.0f} MB")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="all", choices=["all", "vanilla", "cifg", "lstmp", "wmc", "transformer"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--opt", default="adam", choices=list(OPTS))
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--init", default="default", choices=["default", "xavier", "orthogonal"])
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--tf", type=float, default=1.0, help="train teacher forcing (1.0 = always)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="base", help="label for this run, used in checkpoint name")
    p.add_argument("--tok_dir", default="data", help="folder with spm_en.model / spm_dar.model")
    p.add_argument("--no_mask", action="store_true",
                   help="ablation: original unmasked encoder (pads run through the cell)")
    p.add_argument("--attn", action="store_true", help="Luong dot attention (secondary experiment only)")
    p.add_argument("--attn_scale", action="store_true", help="scaled dot-product attention (score / sqrt(d))")
    p.add_argument("--attn_type", default="dot", choices=["dot", "general", "additive"])
    p.add_argument("--bidir", action="store_true", help="bidirectional encoder (second cell of the same type)")
    p.add_argument("--ls", type=float, default=0.0,
                   help="label smoothing of the TRAINING loss (val loss stays plain CE)")
    p.add_argument("--bpe_alpha", type=float, default=0.0, help="BPE-dropout / subword sampling on training pairs")
    p.add_argument("--direction", default="en2dar", choices=["en2dar", "dar2en"])
    p.add_argument("--no_src", action="store_true", help="control: decoder never sees the source (pure LM)")
    p.add_argument("--bucket", dest="bucket", action="store_true", default=True,
                   help="length-bucketed batches (default)")
    p.add_argument("--no_bucket", dest="bucket", action="store_false")
    p.add_argument("--train_csv", default="data/train.csv",
                   help="data/train_plus.csv = train + extra DODa pairs "
                        "(data-scaling experiment, see scripts/build_extra_data.py)")
    p.add_argument("--limit_train", type=int, default=0,
                   help="smoke runs / CI: train on the first N pairs only (0 = all)")
    p.add_argument("--runs_dir", default="checkpoints")
    p.add_argument("--results", default="results.jsonl")
    a = p.parse_args()

    sp_src, sp_tgt = f"{a.tok_dir}/spm_en.model", f"{a.tok_dir}/spm_dar.model"
    cols = ("en", "dar")
    if a.direction == "dar2en":
        sp_src, sp_tgt, cols = sp_tgt, sp_src, ("dar", "en")
    src_v = spm.SentencePieceProcessor(model_file=sp_src).get_piece_size()
    tgt_v = spm.SentencePieceProcessor(model_file=sp_tgt).get_piece_size()
    if a.bpe_alpha > 0:
        spm.set_random_generator_seed(a.seed)
    train_ds = TranslationDataset(a.train_csv, sp_src, sp_tgt, *cols, sample_alpha=a.bpe_alpha)
    if a.limit_train:                     # the first N pairs (dataframe + pre-encoded ids + bucketing lengths)
        n = a.limit_train
        train_ds.df = train_ds.df.head(n)
        train_ds.src, train_ds.tgt, train_ds.src_lens = train_ds.src[:n], train_ds.tgt[:n], train_ds.src_lens[:n]
    val_ds = TranslationDataset("data/val.csv", sp_src, sp_tgt, *cols)
    for m in (["vanilla", "cifg", "lstmp"] if a.model == "all" else [a.model]):
        train_model(m, a, train_ds, val_ds, src_v, tgt_v)

if __name__ == "__main__":
    main()
