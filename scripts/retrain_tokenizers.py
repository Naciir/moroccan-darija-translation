"""Retrain the two SentencePiece BPE tokenizers (same settings as notebook cell 22, plus the fixes).

Fixes vs the shipped tokenizers:
  * split_by_number=False on both sides. With the default (True) every Arabizi word is cut at its
    digits (3la -> _3/la, ch7al -> _ch/7/al) because 3/7/9/2/5/8 are LETTERS in Darija Latin script.
  * EN side lowercased (normalization_rule_name="nmt_nfkc_cf"); case carries no meaning there.
  * DAR side keeps "identity": capitals S/D/T are emphatic consonants, so case IS meaningful.

    python scripts/retrain_tokenizers.py                      # -> data/spm_en.model, data/spm_dar.model
    python scripts/retrain_tokenizers.py --out_dir data/tok_mono --dar_extra data/mono_dar.txt
"""
import argparse, sys, os, re
import numpy as np
import pandas as pd
import sentencepiece as spm

def piece_core(p):
    return p.replace("▁", "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--en_norm", default="nmt_nfkc_cf", choices=["nmt_nfkc_cf", "identity"])
    ap.add_argument("--split_by_number", type=int, default=0, choices=[0, 1])
    ap.add_argument("--vocab", type=int, default=3000)
    ap.add_argument("--dar_extra", default=None, help="extra monolingual Darija text, DAR side only")
    ap.add_argument("--train_csv", default=None, help="train the tokenizers on this CSV (en, dar) instead of "
                                                       "data/train.{en,dar}.txt, e.g. data/train_plus.csv")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")            # Windows: "▁" etc. must survive redirected stdout
    os.makedirs(a.out_dir, exist_ok=True)

    def train_spm(txt, prefix, norm):
        spm.SentencePieceTrainer.train(
            input=txt, model_prefix=prefix, vocab_size=a.vocab,
            model_type="bpe", character_coverage=1.0,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            normalization_rule_name=norm, hard_vocab_limit=False,
            split_by_number=bool(a.split_by_number), minloglevel=2,
        )

    en_txt, dar_txt = "data/train.en.txt", "data/train.dar.txt"
    if a.train_csv:
        df = pd.read_csv(a.train_csv)
        en_txt, dar_txt = os.path.join(a.out_dir, "_en.txt"), os.path.join(a.out_dir, "_dar.txt")
        for col, path in [("en", en_txt), ("dar", dar_txt)]:
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(" ".join(str(x).split()) + "\n" for x in df[col])
    if a.dar_extra:
        base_dar, dar_txt = dar_txt, os.path.join(a.out_dir, "_dar_train_plus_extra.txt")
        with open(dar_txt, "w", encoding="utf-8") as f:
            for p in [base_dar, a.dar_extra]:
                with open(p, encoding="utf-8") as g:
                    f.write(g.read().rstrip("\n") + "\n")
    train_spm(en_txt, os.path.join(a.out_dir, "spm_en"), a.en_norm)
    train_spm(dar_txt, os.path.join(a.out_dir, "spm_dar"), "identity")
    for tmp in ["_en.txt", "_dar.txt", "_dar_train_plus_extra.txt"]:     # temporary text dumps only
        if os.path.exists(os.path.join(a.out_dir, tmp)):
            os.remove(os.path.join(a.out_dir, tmp))

    sp_en = spm.SentencePieceProcessor(model_file=os.path.join(a.out_dir, "spm_en.model"))
    sp_dar = spm.SentencePieceProcessor(model_file=os.path.join(a.out_dir, "spm_dar.model"))
    train, val, test = (pd.read_csv(f"data/{s}.csv") for s in ["train", "val", "test"])
    print(f"== tokenizers in {a.out_dir} | en_norm={a.en_norm} split_by_number={a.split_by_number} "
          f"vocab EN {sp_en.get_piece_size()} / DAR {sp_dar.get_piece_size()} | dar_extra={a.dar_extra}")

    for name, sp, col in [("EN", sp_en, "en"), ("DAR", sp_dar, "dar")]:
        words = sum(len(str(s).split()) for s in train[col])
        pieces = sum(len(sp.encode(str(s))) for s in train[col])
        print(f"{name}: tok/word (train) {pieces / words:.3f}", end="")
        for sn, d in [("val", val), ("test", test)]:
            ids = [i for s in d[col] for i in sp.encode(str(s))]
            print(f" | unk rate {sn} {np.mean([i == 1 for i in ids]):.4%}", end="")
        print()

    vocab = [sp_dar.id_to_piece(i) for i in range(sp_dar.get_piece_size())]
    mixed = sum(1 for p in vocab if re.search(r"\d", piece_core(p)) and re.search(r"[A-Za-z]", piece_core(p)))
    test_pieces = [piece_core(p) for s in test["dar"] for p in sp_dar.encode(str(s), out_type=str)]
    digit_only = np.mean([bool(p) and p.isdigit() for p in test_pieces])
    print(f"DAR: vocab pieces mixing digit+letter: {mixed} | digit-only pieces in test: {digit_only:.2%}")

    print("DAR sample:", {w: sp_dar.encode(w, out_type=str) for w in "3la ch7al t7kem nt9adou".split()})
    print("EN  sample:", sp_en.encode("You can't judge a Book by its COVER", out_type=str))

    en_lens = [len(sp_en.encode(str(s))) for s in train["en"]]
    dar_lens = [len(sp_dar.encode(str(s))) for s in train["dar"]]
    for name, l in [("EN", en_lens), ("DAR", dar_lens)]:
        print(name, "tokens  mean/95%/99%/max:",
              round(np.mean(l), 1), int(np.percentile(l, 95)), int(np.percentile(l, 99)), max(l))
    MAX_LEN = int(max(np.percentile(en_lens, 99), np.percentile(dar_lens, 99))) + 2  # +2 for <bos>/<eos>
    print("MAX_LEN =", MAX_LEN)

if __name__ == "__main__":
    main()
