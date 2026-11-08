"""Gradio demo: English -> Darija (Latin script) + the attention heatmap of the greedy decode.

    pip install -e ".[app]"   &&   python app/app.py                     # run from the repository root
    DARIJA_NMT_RUN=vanilla_attn_s0 python app/app.py                     # which run to serve (default vanilla_attn_s0)
    DARIJA_NMT_HUB_REPO=<user>/<repo> python app/app.py                  # weights from a Hugging Face Hub repo instead

The run is loaded from <root>/checkpoints/<run>.pt + .json with its tokenizers under <root>/data/ (the paths stored in
the run config are relative): <root> = the current directory, or the downloaded snapshot of DARIJA_NMT_HUB_REPO
(same layout: checkpoints/<run>.{pt,json} + data/spm_{en,dar}.model). gradio is imported only in main().
"""
import contextlib, os, sys

try:
    import darija_nmt  # noqa: F401
except ImportError:                       # HF Space / plain checkout: the package sits next to app.py or in ../src
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:0] = [here, os.path.join(here, "..", "src")]

import numpy as np
import torch

from darija_nmt.evaluate import load_run, left_pad, src_ids, BOS, EOS

RUN = os.environ.get("DARIJA_NMT_RUN", "vanilla_attn_s0")
HUB = os.environ.get("DARIJA_NMT_HUB_REPO")
MAX_LEN = 48

def load(run=RUN, hub=HUB):
    root = "."
    if hub:
        from huggingface_hub import snapshot_download
        root = snapshot_download(hub, allow_patterns=[f"checkpoints/{run}.*", "data/*.model"])
    with contextlib.chdir(root):           # load_run resolves checkpoints/ and the config's tok_dir relative to cwd
        return load_run(run, "checkpoints")

@torch.no_grad()
def greedy_with_attention(model, cfg, sp_src, sentence, max_len=MAX_LEN):
    """Greedy decode (= evaluate.greedy_one) that also returns the attention weights of every step (attention runs).
    -> (target ids, (steps, source positions) array or None, source pieces in encoder order)"""
    ids = src_ids(sp_src, sentence)                                   # EOS + reversed source pieces
    state, enc_out, mask = model.encode(left_pad([ids]))
    dec, cur, out, att = model.decoder, torch.tensor([BOS], device=enc_out.device), [], []
    use_attn = cfg.get("attn") and cfg.get("model") != "transformer"
    for _ in range(max_len):
        if use_attn:                                                  # Decoder.forward_step, keeping the weights
            h, c = dec.cell(dec.drop(dec.embedding(cur)), state); state = (h, c)
            a, o = dec.attend(h, enc_out, mask)
            logits = dec.fc_out(dec.drop(o))
        else:
            logits, state = dec.forward_step(cur, state, enc_out, mask)
        cur = logits.argmax(1)
        if cur.item() == EOS:
            break
        out.append(cur.item())
        if use_attn:
            att.append(a[0].cpu().numpy())
    return out, (np.array(att) if att else None), [sp_src.id_to_piece(i) for i in ids]

def heatmap(A, src_pieces, tgt_pieces, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    A, src_lab = A[:, ::-1], src_pieces[::-1]                         # back to reading order
    fig, ax = plt.subplots(figsize=(max(4, 0.45 * len(src_lab) + 1.5), max(3, 0.35 * len(tgt_pieces) + 1.2)))
    ax.imshow(A, cmap="Greys", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(src_lab))); ax.set_xticklabels(src_lab, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(tgt_pieces))); ax.set_yticklabels(tgt_pieces, fontsize=8)
    ax.set_title(title, fontsize=8); fig.tight_layout()
    return fig

def make_translate(model, cfg, sp_src, sp_tgt):
    def translate(text):
        text = (text or "").strip()
        if not text:
            return "", None
        ids, A, src_pieces = greedy_with_attention(model, cfg, sp_src, text)
        fig = heatmap(A, src_pieces, [sp_tgt.id_to_piece(i) for i in ids], f"{RUN}: {text[:60]}") \
            if A is not None and len(ids) else None
        return sp_tgt.decode(ids), fig
    return translate

def build_demo():
    import gradio as gr
    model, cfg, sp_src, sp_tgt = load()
    fn = make_translate(model, cfg, sp_src, sp_tgt)
    return gr.Interface(
        fn=fn, inputs=gr.Textbox(label="English", placeholder="where are you?"),
        outputs=[gr.Textbox(label="Darija (Latin script)"), gr.Plot(label="Attention (rows: output, cols: source)")],
        title="English -> Moroccan Darija", description=f"run `{RUN}`, greedy decoding",
        examples=[["how are you?"], ["where are you?"], ["thank you very much"]], flagging_mode="never")

def main():
    build_demo().launch()

if __name__ == "__main__":
    main()
