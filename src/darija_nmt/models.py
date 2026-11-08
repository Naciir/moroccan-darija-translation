import random
import torch
import torch.nn as nn
from darija_nmt.cells import VanillaLSTMCell

NEG_INF = -1e9   # attention score at pad positions


class RecurrentLayer(nn.Module):
    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    def forward(self, x_seq, init_state, mask=None):
        # mask: (B, T), 1 = real token. Where mask == 0 the previous (h, c) is carried over unchanged,
        # so pad positions never touch the state (and the output there is the carried h).
        # mask=None is the original unmasked recurrence.
        h_t, c_t = init_state
        keep = None if mask is None else mask.bool().unsqueeze(-1)
        outputs = []
        for t in range(x_seq.size(1)):
            h_new, c_new = self.cell(x_seq[:, t, :], (h_t, c_t))
            if keep is None:
                h_t, c_t = h_new, c_new
            else:
                h_t, c_t = torch.where(keep[:, t], h_new, h_t), torch.where(keep[:, t], c_new, c_t)
            outputs.append(h_t)
        return torch.stack(outputs, dim=1), (h_t, c_t)


def _make_cell(cell_cls, embed_dim, hidden_dim, proj_dim):
    if proj_dim is not None and cell_cls.__name__ == "LSTMPCell":
        cell = cell_cls(embed_dim, hidden_dim, proj_dim=proj_dim)
        return cell, proj_dim
    cell = cell_cls(embed_dim, hidden_dim)
    return cell, getattr(cell, "proj_dim", hidden_dim)


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, cell_cls=VanillaLSTMCell,
                 pad_idx=0, proj_dim=None, dropout=0.0, use_mask=True, bidir=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.cell, self.out_dim = _make_cell(cell_cls, embed_dim, hidden_dim, proj_dim)
        self.rnn = RecurrentLayer(self.cell)
        self.hidden_dim = hidden_dim
        self.drop = nn.Dropout(dropout)
        self.pad_idx = pad_idx
        self.use_mask = use_mask   # False = original behaviour (pads run through the cell)
        self.bidir = bidir
        if bidir:   # a second cell of the same type reads the sentence the other way; bridges map back to one direction
            self.cell_b, _ = _make_cell(cell_cls, embed_dim, hidden_dim, proj_dim)
            self.rnn_b = RecurrentLayer(self.cell_b)
            self.bridge_out = nn.Linear(2 * self.out_dim, self.out_dim)    # memory for attention
            self.bridge_h = nn.Linear(2 * self.out_dim, self.out_dim)      # decoder initial h
            self.bridge_c = nn.Linear(2 * hidden_dim, hidden_dim)          # decoder initial c

    def forward(self, src_seq):
        """-> (outputs (B, T, out_dim), (h_n, c_n))"""
        b, dev = src_seq.size(0), src_seq.device
        embedded = self.drop(self.embedding(src_seq))
        h_0 = torch.zeros(b, self.out_dim, device=dev)
        c_0 = torch.zeros(b, self.hidden_dim, device=dev)
        mask = (src_seq != self.pad_idx) if self.use_mask else None
        out, (h, c) = self.rnn(embedded, (h_0, c_0), mask)
        if not self.bidir:
            return out, (h, c)
        # the source is left-padded; flipped it is right-padded, and the masked recurrence carries the state
        # unchanged through the trailing pads, so (h_b, c_b) is the state after the last real token
        out_b, (h_b, c_b) = self.rnn_b(torch.flip(embedded, [1]), (h_0, c_0),
                                       None if mask is None else torch.flip(mask, [1]))
        out = torch.tanh(self.bridge_out(torch.cat([out, torch.flip(out_b, [1])], -1)))
        return out, (torch.tanh(self.bridge_h(torch.cat([h, h_b], -1))), self.bridge_c(torch.cat([c, c_b], -1)))


class Decoder(nn.Module):
    """attn_type: "dot"      score = h_s . h_t                      (Luong dot; attn_scale: / sqrt(d))
                  "general"  score = h_s . (W_a h_t)                (Luong general / bilinear)
                  "additive" score = v . tanh(W_k h_s + W_q h_t)    (Bahdanau-style concat)
    and in every case h~_t = tanh(W_c [ctx_t ; h_t]) feeds the output layer."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, cell_cls=VanillaLSTMCell,
                 pad_idx=0, proj_dim=None, dropout=0.0, attn=False, attn_scale=False, attn_type="dot"):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.cell, self.out_dim = _make_cell(cell_cls, embed_dim, hidden_dim, proj_dim)
        self.fc_out = nn.Linear(self.out_dim, vocab_size)
        self.drop = nn.Dropout(dropout)
        self.attn = attn
        self.attn_scale = attn_scale   # scaled dot product: score / sqrt(d) (LSTMP's unbounded projection saturates it)
        self.attn_type = attn_type
        if attn:   # Luong global attention:  h~_t = tanh(W_c [ctx_t ; h_t])
            d = self.out_dim
            self.W_c = nn.Linear(2 * d, d, bias=False)
            if attn_type == "general":
                self.W_a = nn.Linear(d, d, bias=False)
            elif attn_type == "additive":
                self.W_k = nn.Linear(d, d, bias=False)
                self.W_q = nn.Linear(d, d)
                self.v = nn.Linear(d, 1, bias=False)
            elif attn_type != "dot":
                raise ValueError(attn_type)

    def scores(self, h_t, enc_out):
        """(B, d), (B, S, d) -> attention logits (B, S)"""
        if self.attn_type == "additive":
            return self.v(torch.tanh(self.W_k(enc_out) + self.W_q(h_t).unsqueeze(1))).squeeze(2)
        q = self.W_a(h_t) if self.attn_type == "general" else h_t
        score = torch.bmm(enc_out, q.unsqueeze(2)).squeeze(2)               # (B, S)  h_s . q_t
        return score / h_t.size(-1) ** 0.5 if self.attn_scale else score

    def attend(self, h_t, enc_out, src_mask=None):
        """-> (attention weights (B, S), attentional output h~_t (B, d))"""
        score = self.scores(h_t, enc_out)
        if src_mask is not None:
            score = score.masked_fill(~src_mask, NEG_INF)                      # never attend to pads
        a_t = torch.softmax(score, dim=1)
        ctx = torch.bmm(a_t.unsqueeze(1), enc_out).squeeze(1)                  # (B, out_dim)
        return a_t, torch.tanh(self.W_c(torch.cat([ctx, h_t], dim=1)))

    def forward_step(self, token_input, state, enc_out=None, src_mask=None):
        embedded = self.drop(self.embedding(token_input))
        h_t, c_t = self.cell(embedded, state)
        out = self.attend(h_t, enc_out, src_mask)[1] if self.attn else h_t
        return self.fc_out(self.drop(out)), (h_t, c_t)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, no_src=False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.no_src = no_src   # LM control: the decoder never sees the source

    def encode(self, src):
        """-> (decoder initial state, encoder outputs, src mask). With no_src the state and the
        encoder outputs (hence the attention context) are all zeros."""
        mask = src != self.encoder.pad_idx
        if self.no_src:
            b, s, dev, e = src.size(0), src.size(1), src.device, self.encoder
            state = (torch.zeros(b, e.out_dim, device=dev), torch.zeros(b, e.hidden_dim, device=dev))
            return state, torch.zeros(b, s, e.out_dim, device=dev), mask
        enc_out, state = self.encoder(src)
        return state, enc_out, mask

    def forward(self, src, tgt, teacher_forcing_ratio=1.0):
        b, tgt_len = tgt.size(0), tgt.size(1)
        outputs = torch.zeros(b, tgt_len, self.decoder.vocab_size, device=src.device)
        state, enc_out, src_mask = self.encode(src)
        curr = tgt[:, 0]
        for t in range(1, tgt_len):
            pred, state = self.decoder.forward_step(curr, state, enc_out, src_mask)
            outputs[:, t, :] = pred
            curr = tgt[:, t] if random.random() < teacher_forcing_ratio else pred.argmax(1)
        return outputs


def init_weights(model, how):
    if how == "default":
        return
    fn = nn.init.xavier_uniform_ if how == "xavier" else nn.init.orthogonal_
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # fused gate layers are initialised per gate block, exactly like the old per-gate Linears
            for w in m.weight.data.chunk(getattr(m, "n_gates", 1), dim=0):
                fn(w)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
