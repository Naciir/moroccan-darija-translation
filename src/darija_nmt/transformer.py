"""Small Transformer baseline (context row only: the study is about the LSTM cells).

Same interface as models.Seq2Seq, so train.py / evaluate.py (greedy, beam, scoring) work unchanged:
  encode(src)                       -> (state, memory, src_mask)
  decoder.forward_step(tok, state, memory, src_mask) -> (logits, state)
The decoder "state" is (prefix of generated tokens, dummy): the prefix is re-decoded at every step (sentences are
short), and both tensors are batch-first so beam search can index them like LSTM states.
Positions count real tokens only, so the left padding of the (reversed) source does not shift them."""
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, positions):
        return self.pe[positions]

class _Decoder(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.p = [parent]                     # no submodule registration (parameters live in the parent)
        self.vocab_size = parent.tgt_vocab

    def forward_step(self, token_input, state, enc_out=None, src_mask=None):
        prefix, dummy = state
        prefix = torch.cat([prefix, token_input.unsqueeze(1)], 1)
        logits = self.p[0].decode(prefix, enc_out, src_mask)[:, -1]
        return logits, (prefix, dummy)

class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d=256, heads=4, layers=3, ff=1024, dropout=0.3, pad_idx=0):
        super().__init__()
        self.tgt_vocab, self.pad_idx, self.d = tgt_vocab, pad_idx, d
        self.src_emb = nn.Embedding(src_vocab, d, padding_idx=pad_idx)
        self.tgt_emb = nn.Embedding(tgt_vocab, d, padding_idx=pad_idx)
        self.pos = PositionalEncoding(d)
        self.core = nn.Transformer(d, heads, layers, layers, ff, dropout, batch_first=True, norm_first=True)
        self.fc_out = nn.Linear(d, tgt_vocab)
        self.drop = nn.Dropout(dropout)
        self.decoder = _Decoder(self)
        self.no_src = False

    def _embed(self, emb, ids):
        real = ids != self.pad_idx
        positions = (real.long().cumsum(1) - 1).clamp(min=0)
        return self.drop(emb(ids) * math.sqrt(self.d) + self.pos(positions)), real

    def encode(self, src):
        x, real = self._embed(self.src_emb, src)
        memory = self.core.encoder(x, src_key_padding_mask=~real)
        b = src.size(0)
        state = (torch.zeros(b, 0, dtype=torch.long, device=src.device), torch.zeros(b, 1, device=src.device))
        return state, memory, real

    def decode(self, prefix, memory, src_mask):
        y, real = self._embed(self.tgt_emb, prefix)
        causal = torch.triu(torch.ones(prefix.size(1), prefix.size(1), dtype=torch.bool, device=prefix.device), 1)
        out = self.core.decoder(y, memory, tgt_mask=causal, tgt_key_padding_mask=~real,
                                memory_key_padding_mask=~src_mask)
        return self.fc_out(out)

    def forward(self, src, tgt, teacher_forcing_ratio=1.0):
        """Teacher forcing only. Same layout as Seq2Seq: outputs[:, t] predicts tgt[:, t] (outputs[:, 0] unused)."""
        _, memory, src_mask = self.encode(src)
        logits = self.decode(tgt[:, :-1], memory, src_mask)
        return torch.cat([torch.zeros_like(logits[:, :1]), logits], 1)
