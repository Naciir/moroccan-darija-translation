import torch
from torch.utils.data import Dataset, Sampler
import pandas as pd
import sentencepiece as spm

class TranslationDataset(Dataset):
    def __init__(self, csv_path, sp_src_path="data/spm_en.model", sp_tgt_path="data/spm_dar.model",
                 src_col="en", tgt_col="dar", sample_alpha=0.0):
        """src_col/tgt_col: ("dar", "en") with swapped tokenizers = the reverse direction (back-translation).
        sample_alpha > 0: BPE-dropout, every __getitem__ re-segments the pair with SentencePiece sampling
        (training only; lengths used for bucketing stay the deterministic ones)."""
        self.df = pd.read_csv(csv_path)
        self.src_col, self.tgt_col, self.sample_alpha = src_col, tgt_col, sample_alpha
        self.sp_src = spm.SentencePieceProcessor(model_file=sp_src_path)
        self.sp_tgt = spm.SentencePieceProcessor(model_file=sp_tgt_path)

        # SentencePiece special token IDs as configured in Day 1:
        # pad=0, unk=1, bos=2, eos=3
        self.pad_id = 0
        self.bos_id = 2
        self.eos_id = 3

        # Encode once up front (same ids as encoding in __getitem__, just faster).
        # Add BOS and EOS tokens to target sequence; EOS to source sequence
        self.src = [self.sp_src.encode(str(s)) + [self.eos_id] for s in self.df[src_col]]
        self.tgt = [[self.bos_id] + self.sp_tgt.encode(str(s)) + [self.eos_id] for s in self.df[tgt_col]]
        self.src_lens = [len(s) for s in self.src]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.sample_alpha > 0:
            enc = lambda sp, col: sp.encode(str(self.df[col].iat[idx]), enable_sampling=True,
                                            alpha=self.sample_alpha, nbest_size=-1)
            return (torch.tensor(enc(self.sp_src, self.src_col) + [self.eos_id], dtype=torch.long),
                    torch.tensor([self.bos_id] + enc(self.sp_tgt, self.tgt_col) + [self.eos_id], dtype=torch.long))
        return torch.tensor(self.src[idx], dtype=torch.long), torch.tensor(self.tgt[idx], dtype=torch.long)

def pad_collate_fn(batch, pad_id=0):
    src_batch, tgt_batch = zip(*batch)

    src_lens = [len(x) for x in src_batch]
    tgt_lens = [len(y) for y in tgt_batch]

    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    src_padded = torch.full((len(batch), max_src), pad_id, dtype=torch.long)
    tgt_padded = torch.full((len(batch), max_tgt), pad_id, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_batch, tgt_batch)):
        src_padded[i, :len(s)] = s
        tgt_padded[i, :len(t)] = t

    return src_padded, tgt_padded, torch.tensor(src_lens), torch.tensor(tgt_lens)

class BucketBatchSampler(Sampler):
    """Length-bucketed batches: shuffle, cut into chunks of ~batch_size*50 examples, sort each chunk
    by src length, slice into batches, shuffle the batch order. Little padding, still random.
    Deterministic given `seed` (each epoch draws from the same seeded generator)."""
    def __init__(self, lengths, batch_size, shuffle=True, seed=0, generator=None, chunk_mult=50):
        self.lengths, self.bs, self.shuffle, self.chunk = list(lengths), batch_size, shuffle, batch_size * chunk_mult
        self.g = generator if generator is not None else torch.Generator().manual_seed(seed)

    def _batches(self):
        n = len(self.lengths)
        idx = torch.randperm(n, generator=self.g).tolist() if self.shuffle else list(range(n))
        batches = []
        for s in range(0, n, self.chunk):
            part = sorted(idx[s:s + self.chunk], key=lambda i: self.lengths[i])   # stable sort
            batches += [part[j:j + self.bs] for j in range(0, len(part), self.bs)]
        if self.shuffle:
            batches = [batches[i] for i in torch.randperm(len(batches), generator=self.g).tolist()]
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        n = len(self.lengths)
        return sum(-(-min(self.chunk, n - s) // self.bs) for s in range(0, n, self.chunk))
