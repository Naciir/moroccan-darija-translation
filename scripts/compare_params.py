from darija_nmt.cells import VanillaLSTMCell, CIFGCell, LSTMPCell
from darija_nmt.models import Encoder, Decoder, Seq2Seq

VOCAB_SIZE = 3000
EMBED_DIM = 128
HIDDEN_DIM = 256
PROJ_DIM = 128

def get_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 1. Vanilla LSTM
v_enc = Encoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, cell_cls=VanillaLSTMCell)
v_dec = Decoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, cell_cls=VanillaLSTMCell)
v_model = Seq2Seq(v_enc, v_dec)

# 2. CIFG
c_enc = Encoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, cell_cls=CIFGCell)
c_dec = Decoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, cell_cls=CIFGCell)
c_model = Seq2Seq(c_enc, c_dec)

# 3. LSTMP
p_enc = Encoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, cell_cls=LSTMPCell, proj_dim=PROJ_DIM)
p_dec = Decoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, cell_cls=LSTMPCell, proj_dim=PROJ_DIM)
p_model = Seq2Seq(p_enc, p_dec)

print("Architecture Parameter Comparison (for Report Table 1):")
print(f"1. Vanilla LSTM: {get_params(v_model):,} parameters")
print(f"2. CIFG:         {get_params(c_model):,} parameters (Diff: {get_params(c_model) - get_params(v_model):,})")
print(f"3. LSTMP:        {get_params(p_model):,} parameters (Diff: {get_params(p_model) - get_params(v_model):,})")

# Secondary experiment: the same models with Luong attention (adds only W_c: 2*out_dim x out_dim)
print("With attention (secondary table):")
for name, cell, pd_ in [("Vanilla LSTM", VanillaLSTMCell, None), ("CIFG", CIFGCell, None),
                        ("LSTMP", LSTMPCell, PROJ_DIM)]:
    m = Seq2Seq(Encoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, cell_cls=cell, proj_dim=pd_),
                Decoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, cell_cls=cell, proj_dim=pd_, attn=True))
    print(f"   {name + ':':14s}{get_params(m):,} parameters")
