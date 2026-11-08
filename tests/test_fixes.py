"""Checks for the fixes (one test per check, numbered as in the original script). The working directory is set by
conftest.py: data/, legacy/ etc. are relative to it."""
import functools, importlib.util, os, sys
from pathlib import Path

import pandas as pd
import pytest
import sentencepiece as spm
import torch

from darija_nmt.cells import VanillaLSTMCell, CIFGCell, LSTMPCell, convert_legacy_state_dict
from darija_nmt.models import Encoder, Decoder, Seq2Seq
from darija_nmt.data import TranslationDataset, BucketBatchSampler
from darija_nmt.train import build_model, DEVICE
import darija_nmt.evaluate as ev

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
# the original (unfused, one nn.Linear per gate) cells, kept verbatim in legacy/cells_original.py
ORIG_CELLS = os.environ.get("ORIG_CELLS", str(ROOT / "legacy" / "cells_original.py"))
spec = importlib.util.spec_from_file_location("orig_cells", ORIG_CELLS)
orig_cells = importlib.util.module_from_spec(spec); spec.loader.exec_module(orig_cells)

E, H, P = 128, 256, 128
NEW = {"vanilla": (VanillaLSTMCell, orig_cells.VanillaLSTMCell, "ifog"),
       "cifg": (CIFGCell, orig_cells.CIFGCell, "fog"),
       "lstmp": (LSTMPCell, orig_cells.LSTMPCell, "ifog")}
nparams = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
TEST = pd.read_csv("data/test.csv")

def pair(name):
    new_cls, old_cls, gates = NEW[name]
    kw = {"proj_dim": P} if name == "lstmp" else {}
    torch.manual_seed(0)
    old, new = old_cls(E, H, **kw), new_cls(E, H, **kw)
    with torch.no_grad():
        new.W_x.weight.copy_(torch.cat([getattr(old, f"W_x{g}").weight for g in gates]))
        new.W_x.bias.copy_(torch.cat([getattr(old, f"W_x{g}").bias for g in gates]))
        new.W_h.weight.copy_(torch.cat([getattr(old, f"W_h{g}").weight for g in gates]))
        if name == "lstmp":
            new.W_p.load_state_dict(old.W_p.state_dict())
    return old, new

def test_01_fused_cells_equal_original_per_gate_cells():
    """fused cells == original per-gate cells (10 steps, atol 1e-6)"""
    for name in NEW:
        old, new = pair(name)
        rec = P if name == "lstmp" else H
        torch.manual_seed(1)
        h1 = h2 = torch.randn(7, rec); c1 = c2 = torch.randn(7, H)
        for _ in range(10):
            x = torch.randn(7, E)
            h1, c1 = old(x, (h1, c1)); h2, c2 = new(x, (h2, c2))
            assert torch.allclose(h1, h2, atol=1e-6) and torch.allclose(c1, c2, atol=1e-6), name
        # the generic legacy converter gives the same fused weights
        conv = convert_legacy_state_dict({f"cell.{k}": v for k, v in old.state_dict().items()})
        for k, v in new.state_dict().items():
            assert torch.equal(conv[f"cell.{k}"], v), (name, k)
        assert new.input_dim == E and new.hidden_dim == H
        assert getattr(new, "proj_dim", None) == getattr(old, "proj_dim", None)

def test_02_parameter_counts_identical():
    """parameter counts identical (cells + full models 2,327,480 / 2,130,360 / 1,747,128)"""
    for name in NEW:
        old, new = pair(name)
        assert nparams(old) == nparams(new), name
    want = {"vanilla": 2327480, "cifg": 2130360, "lstmp": 1747128}
    for name, n in want.items():
        cell, _, _ = NEW[name]
        pd_ = P if name == "lstmp" else None
        m = Seq2Seq(Encoder(3000, E, H, cell_cls=cell, proj_dim=pd_), Decoder(3000, E, H, cell_cls=cell, proj_dim=pd_))
        assert nparams(m) == n, (name, nparams(m))

def test_03_masked_encoder_left_padded_batch_equals_alone():
    """masked encoder: left-padded batch == each sentence alone; use_mask=False drifts"""
    torch.manual_seed(0)
    seqs = [[5, 9, 3], [7, 8, 9, 10, 11, 12, 3], [4, 3], [20, 21, 22, 23, 3]]
    m = max(map(len, seqs))
    batch = torch.tensor([[0] * (m - len(s)) + s for s in seqs])
    for name in NEW:
        cell, _, _ = NEW[name]
        enc = Encoder(100, E, H, cell_cls=cell, proj_dim=P if name == "lstmp" else None).eval()
        _, (hb, cb) = enc(batch)
        for i, s in enumerate(seqs):
            _, (h1, c1) = enc(torch.tensor([s]))
            assert torch.allclose(hb[i], h1[0], atol=1e-5) and torch.allclose(cb[i], c1[0], atol=1e-5), (name, i)
        enc.use_mask = False                             # original behaviour: pads go through the cell
        _, (hu, _) = enc(batch)
        _, (h1, _) = enc(torch.tensor([seqs[0]]))
        assert (hu[0] - h1[0]).abs().max() > 1e-3, name  # short sentence with leading pads drifts
        assert torch.allclose(hu[1], hb[1], atol=1e-6)   # the longest (no pads) is unaffected

# ---- decoding models: the trained legacy vanilla model (needs legacy/checkpoints) and a random LSTMP + attention ----
@functools.cache
def decoding_model(name):
    if name == "legacy vanilla":
        legacy, _, sp_src, _ = ev.load_run("vanilla_base_s0", "legacy/checkpoints")   # trained, no attention
        return legacy, sp_src
    torch.manual_seed(0)
    sp_new = spm.SentencePieceProcessor(model_file="data/spm_en.model")
    cfg = dict(model="lstmp", mask=True, attn=True, no_src=False, embed_dim=E, hidden_dim=H, dropout=0.0,
               src_vocab=3000, tgt_vocab=3000)
    attn = build_model(cfg).to(DEVICE).eval()                                      # random init, attention
    return attn, sp_new

DEC = [pytest.param("legacy vanilla", marks=pytest.mark.needs_checkpoints), "random lstmp+attn"]
SENTS = [str(s) for s in TEST["en"][:20]]

@pytest.mark.parametrize("name", DEC)
def test_04_batched_greedy_equals_per_sentence_greedy(name):
    """batched greedy == per-sentence greedy (20 test sentences)"""
    model, sp = decoding_model(name)
    assert ev.greedy_batch(model, sp, SENTS, 48, bs=7) == [ev.greedy_one(model, sp, s, 48) for s in SENTS], name

@pytest.mark.parametrize("name", DEC)
def test_05_beam1_equals_greedy_beam4_runs(name):
    """beam=1 == greedy; beam=4 runs"""
    model, sp = decoding_model(name)
    g = [ev.greedy_one(model, sp, s, 48) for s in SENTS]
    assert [ev.beam_one(model, sp, s, 1, 48) for s in SENTS] == g, name
    b = [ev.beam_one(model, sp, s, 4, 48) for s in SENTS[:5]]
    assert all(isinstance(x, list) and 3 not in x for x in b)

@pytest.mark.needs_checkpoints
@pytest.mark.needs_data
def test_06_legacy_checkpoint_reproduces_shipped_outputs():
    """legacy checkpoint + converted weights reproduce shipped outputs exactly"""
    if not os.path.exists("legacy/outputs/vanilla_base_s0_pen0.0.csv"):
        pytest.skip("legacy/outputs/vanilla_base_s0_pen0.0.csv not present")
    model, cfg, sp_src, sp_tgt = ev.load_run("vanilla_base_s0", "legacy/checkpoints")
    old = pd.read_csv("legacy/outputs/vanilla_base_s0_pen0.0.csv")["pred"].fillna("").astype(str).tolist()[:50]
    new = [sp_tgt.decode(x) for x in ev.greedy_batch(model, sp_src, [str(s) for s in TEST["en"][:50]], 48)]
    assert new == old

def test_07_attention_ignores_pad_positions():
    """attention ignores pad positions"""
    torch.manual_seed(0)
    for name in NEW:
        cell, _, _ = NEW[name]
        dec = Decoder(50, E, H, cell_cls=cell, proj_dim=P if name == "lstmp" else None, attn=True).eval()
        D = dec.out_dim
        enc_out = torch.randn(3, 6, D)
        mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1] * 6, [0, 0, 0, 0, 0, 1]]).bool()
        state = (torch.randn(3, D), torch.randn(3, H))
        tok = torch.tensor([2, 5, 7])
        l1, _ = dec.forward_step(tok, state, enc_out, mask)
        enc2 = enc_out.clone(); enc2[~mask] = torch.randn(int((~mask).sum()), D) * 100
        l2, _ = dec.forward_step(tok, state, enc2, mask)
        assert torch.equal(l1, l2), name
        enc2[1, 0] += 1.0                                  # a real position DOES matter
        l3, _ = dec.forward_step(tok, state, enc2, mask)
        assert not torch.allclose(l1[1], l3[1])
        dec_plain = Decoder(50, E, H, cell_cls=cell, proj_dim=P if name == "lstmp" else None)
        assert not hasattr(dec_plain, "W_c")               # no attention = original decoder (fc_out on h)

def test_08_tokenizer_arabizi_words_whole_en_lowercased():
    """new tokenizer: Arabizi words kept whole, EN lowercased, DAR case kept"""
    dar = spm.SentencePieceProcessor(model_file="data/spm_dar.model")
    en = spm.SentencePieceProcessor(model_file="data/spm_en.model")
    assert dar.encode("3la", out_type=str) == ["▁3la"], dar.encode("3la", out_type=str)
    assert dar.encode("ch7al", out_type=str) == ["▁ch7al"], dar.encode("ch7al", out_type=str)
    assert en.encode("Hello WORLD Book") == en.encode("hello world book")
    assert all(p == p.lower() for p in en.encode("Hello WORLD Book", out_type=str))
    assert "".join(dar.encode("Sba7 lkhir", out_type=str)).replace("▁", " ").strip() == "Sba7 lkhir"
    old = spm.SentencePieceProcessor(model_file="data/tok_orig/spm_dar.model")
    assert old.encode("3la", out_type=str) == ["▁3", "la"]          # the bug, kept for the ablation

def test_09_no_src_zeroes_encoder_state():
    """Seq2Seq(no_src=True) zeroes the encoder state / context"""
    torch.manual_seed(0)
    for attn in (False, True):
        m = Seq2Seq(Encoder(100, E, H), Decoder(100, E, H, attn=attn), no_src=True).eval()
        src1, src2 = torch.tensor([[0, 5, 6, 3]]), torch.tensor([[9, 8, 7, 3]])
        (h, c), enc_out, _ = m.encode(src1)
        assert h.abs().sum() == 0 and c.abs().sum() == 0 and enc_out.abs().sum() == 0
        tgt = torch.tensor([[2, 10, 11, 3]])
        assert torch.equal(m(src1, tgt), m(src2, tgt))
        m.no_src = False
        assert not torch.equal(m(src1, tgt), m(src2, tgt))

@pytest.mark.needs_data
def test_10_bucket_sampler_deterministic_and_complete():
    """bucket sampler: deterministic per seed, covers every example once"""
    ds = TranslationDataset("data/val.csv")
    s1 = list(BucketBatchSampler(ds.src_lens, 64, seed=3)); s2 = list(BucketBatchSampler(ds.src_lens, 64, seed=3))
    s3 = list(BucketBatchSampler(ds.src_lens, 64, seed=4))
    assert s1 == s2 and s1 != s3
    assert sorted(i for b in s1 for i in b) == list(range(len(ds)))
    assert len(s1) == len(BucketBatchSampler(ds.src_lens, 64))
    pad = lambda bs: sum(max(ds.src_lens[i] for i in b) * len(b) - sum(ds.src_lens[i] for i in b) for b in bs)
    rnd = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0)).tolist()
    assert pad(s1) < 0.3 * pad([rnd[j:j + 64] for j in range(0, len(rnd), 64)])

def test_11_notebook_cells_parse():
    """notebook: every code cell parses (cell 22 keeps its backslash-n literals)"""
    import ast, json
    nb = json.load(open(ROOT / "notebooks" / "darija_data_prep.ipynb", encoding="utf-8"))
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            src = "".join(c["source"]).split("\n")
            ast.parse("\n".join(("pass #" if l.lstrip()[:1] in "!%" and l.strip() else "") + l for l in src))
    s22 = "".join(nb["cells"][22]["source"])
    assert 'f.write(s.replace("\\n", " ") + "\\n")' in s22 and "split_by_number=False" in s22

@pytest.mark.needs_data
def test_12_ablation_arms_keep_original_pad_regime(tmp_path):
    """ablation arms keep the original pad regime (no bucketing, train AND val); dry_run has no side effects"""
    import argparse, subprocess, shutil
    import run_experiments as rx
    from darija_nmt.train import make_loaders
    for tag in ["old", "tok", "mask"]:
        assert "--no_bucket" in rx.GRID[tag][1], tag
    assert "--no_mask" in rx.GRID["old"][1] and "--no_bucket" not in rx.GRID["main"][1]
    tok = ("data/tok_orig/spm_en.model", "data/tok_orig/spm_dar.model")
    tr, va = TranslationDataset("data/train.csv", *tok), TranslationDataset("data/val.csv", *tok)
    def pad_frac(loader):
        pads = tot = 0
        for src, _, _, _ in loader:
            pads += int((src == 0).sum()); tot += src.numel()
        return pads / tot
    fr = {b: [pad_frac(l) for l in make_loaders(argparse.Namespace(bucket=b, bs=64), tr, va, 0)] for b in (False, True)}
    assert fr[False][0] > 0.5 and fr[False][1] > 0.5, fr        # original regime in train and val
    assert fr[True][0] < 0.15 and fr[True][1] < 0.15, fr        # bucketed train and val alike
    shutil.copy(SCRIPTS / "run_experiments.py", tmp_path)
    subprocess.run([sys.executable, "run_experiments.py", "--dry_run", "--seeds", "0"], cwd=tmp_path, check=True,
                   capture_output=True)
    assert not os.path.exists(tmp_path / "logs")

@pytest.mark.parametrize("name", DEC)
def test_13_beam_never_loses_to_greedy(name):
    """beam search never loses to greedy on its own (length-normalised) objective"""
    model, sp = decoding_model(name)
    for s in SENTS:
        g, b = ev.greedy_one(model, sp, s, 48), ev.beam_one(model, sp, s, 5, 48)
        sg, sb = ev.hyp_score(model, sp, s, g), ev.hyp_score(model, sp, s, b)
        assert sb >= sg - 1e-9, (name, s, sg, sb)

@pytest.mark.needs_checkpoints
@pytest.mark.needs_data
def test_14_evaluate_limit_never_writes_results(tmp_path):
    """evaluate --limit never writes to the results file; scripts survive redirected stdout"""
    import subprocess
    env = {k: v for k, v in os.environ.items() if k != "PYTHONIOENCODING"}
    res = tmp_path / "r.jsonl"
    p = subprocess.run([sys.executable, "-m", "darija_nmt.evaluate", "--runs", "vanilla_base_s0",
                        "--runs_dir", "legacy/checkpoints", "--limit", "5", "--results", str(res),
                        "--out_dir", str(tmp_path)], capture_output=True, env=env)
    assert p.returncode == 0, p.stderr.decode(errors="replace")[-500:]
    assert not os.path.exists(res)
    p = subprocess.run([sys.executable, str(SCRIPTS / "retrain_tokenizers.py"), "--out_dir", str(tmp_path)],
                       capture_output=True, env=env)
    assert p.returncode == 0 and b"MAX_LEN" in p.stdout, p.stderr.decode(errors="replace")[-500:]

@pytest.mark.needs_checkpoints
@pytest.mark.needs_data
def test_15_val_loss_token_weighted():
    """val loss is token-weighted: identical for bucketed and random val batches (same masked model)"""
    import argparse, torch.nn as nn
    from darija_nmt.train import make_loaders, run_epoch
    model, cfg, _, _ = ev.load_run("vanilla_base_s0", "legacy/checkpoints")      # mask on at eval, tok_orig
    tok = ("data/tok_orig/spm_en.model", "data/tok_orig/spm_dar.model")
    tr, va = TranslationDataset("data/train.csv", *tok), TranslationDataset("data/val.csv", *tok)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    vl = {b: run_epoch(model, make_loaders(argparse.Namespace(bucket=b, bs=64), tr, va, 0)[1], crit, cfg["tgt_vocab"])
          for b in (False, True)}
    assert abs(vl[True] - vl[False]) < 1e-4, vl                  # per-batch mean gave 4.1189 vs 4.2547
    assert abs(vl[True] - 4.2666) < 2e-3, vl

def test_16_shortened_record_does_not_count_as_real_run(tmp_path):
    """run_experiments: a shortened (1-epoch) record does not count as the real run; a matching one does"""
    import subprocess, shutil, json
    tmp = str(tmp_path)
    shutil.copy(SCRIPTS / "run_experiments.py", tmp)
    os.makedirs(os.path.join(tmp, "ck")); open(os.path.join(tmp, "ck", "vanilla_old_s0.pt"), "wb").close()
    def dry(epochs, extra=()):
        with open(os.path.join(tmp, "r.jsonl"), "w") as f:
            f.write(json.dumps({"run": "vanilla_old_s0", "epochs": epochs, "patience": 5}) + "\n")
        with open(os.path.join(tmp, "t.jsonl"), "w") as f:
            f.write(json.dumps({"run": "vanilla_old_s0", "split": "test", "beam": 1, "ckpt_mtime": mt[0]}) + "\n")
        out = subprocess.run([sys.executable, "run_experiments.py", "--dry_run", "--only", "old", "--seeds", "0",
                              "--runs_dir", "ck", "--results", "r.jsonl", "--test_results", "t.jsonl", *extra],
                             cwd=tmp, check=True, capture_output=True, text=True).stdout
        return [l for l in out.splitlines() if l.startswith("[vanilla_old_s0]")]
    mt = [os.path.getmtime(os.path.join(tmp, "ck", "vanilla_old_s0.pt"))]   # eval record of THIS checkpoint
    cmds = dry(1)                                                    # stale 1-epoch record -> retrain + re-eval
    assert any("] -m darija_nmt.train" in c for c in cmds) and any("darija_nmt.evaluate" in c for c in cmds), cmds
    assert dry(40) == [], dry(40)                                    # real record + eval -> nothing to do
    assert dry(1, ["--train_args", "--epochs 1"]) == []              # explicit short grid matches its own record
    mt[0] -= 100                                                     # eval of an OLDER checkpoint -> re-evaluate only
    cmds = dry(40)
    assert any("darija_nmt.evaluate" in c for c in cmds) and not any("] -m darija_nmt.train" in c for c in cmds), cmds

def test_17_significance_opposite_signs_and_pooled_bootstrap():
    """significance: opposite-sign significant seeds are NOT '2/2'; pooled bootstrap = bootstrap of summed stats"""
    import numpy as np, analyze as an
    cells = an.sig_cells([(0, (-1.16, 0.001)), (1, (0.40, 0.009))], (-0.38, 0.2))
    assert cells[1] == "1 / 1 / 0", cells                         # was '2/2' regardless of direction
    assert cells[0].startswith("-0.38 ± "), cells                  # seed-level mean ± std of the deltas
    assert an.sig_cells([(0, (0.5, 0.01)), (1, (0.7, 0.02)), (2, (0.1, 0.4))], None)[1] == "2 / 0 / 1"
    rng = np.random.RandomState(0)
    S = {f"r{i}": rng.randint(0, 20, size=(50, 18)).astype(float) for i in range(4)}
    R = an.Results.__new__(an.Results); R._stats = S
    assert R.bootstrap(["r0", "r1"], ["r2", "r3"]) == an.boot_stats(S["r0"] + S["r1"], S["r2"] + S["r3"])
    d, p = R.bootstrap("r0", "r0")
    assert d == 0 and p == 1.0

def test_18_append_jsonl_removes_stale_lock(tmp_path):
    """append_jsonl removes a stale lock left by a killed writer (no 60 s wait, lock gone afterwards)"""
    import time, json
    from darija_nmt.train import append_jsonl
    path = str(tmp_path / "r.jsonl"); open(path + ".lock", "w").close()
    os.utime(path + ".lock", (time.time() - 3600, time.time() - 3600))
    t0 = time.time(); append_jsonl(path, {"a": 1})
    assert time.time() - t0 < 5 and not os.path.exists(path + ".lock")
    assert [json.loads(l) for l in open(path)] == [{"a": 1}]

def test_19_failed_job_retried_once_exit_1_stale_locks_removed(tmp_path):
    """run_experiments: failed job is retried once, exit code 1, stale result locks removed at startup"""
    import subprocess, shutil
    tmp = str(tmp_path)
    shutil.copy(SCRIPTS / "run_experiments.py", tmp)
    # stub package in the cwd: `python -m darija_nmt.train` resolves it before the installed one (cwd is sys.path[0])
    os.makedirs(os.path.join(tmp, "darija_nmt")); open(os.path.join(tmp, "darija_nmt", "__init__.py"), "w").close()
    with open(os.path.join(tmp, "darija_nmt", "train.py"), "w") as f:      # stub: always fails, counts its attempts
        f.write("open('attempts', 'a').write('x')\nraise SystemExit(3)\n")
    with open(os.path.join(tmp, "darija_nmt", "baselines.py"), "w") as f:
        f.write("pass\n")
    open(os.path.join(tmp, "r.jsonl.lock"), "w").close()
    p = subprocess.run([sys.executable, "run_experiments.py", "--only", "old", "--seeds", "0", "--jobs", "1",
                        "--runs_dir", "ck", "--results", "r.jsonl", "--test_results", "t.jsonl",
                        "--retry_wait", "0"], cwd=tmp, capture_output=True, text=True)
    assert p.returncode == 1, (p.returncode, p.stdout, p.stderr)
    assert open(os.path.join(tmp, "attempts")).read() == "xx"  # first try + 1 retry
    assert not os.path.exists(os.path.join(tmp, "r.jsonl.lock"))
    assert "1 failed" in p.stdout, p.stdout

@pytest.mark.needs_data
def test_20_mostfreq_baseline_ties_first_occurrence():
    """mostfreq baseline: ties broken by first occurrence in train.csv"""
    from darija_nmt.baselines import most_frequent
    assert most_frequent(["b", "a", "a", "b", "c"]) == "b"
    dar = [str(x) for x in pd.read_csv("data/train.csv")["dar"]]
    top = most_frequent(dar)
    from collections import Counter
    c = Counter(dar); tied = [s for s in c if c[s] == c[top]]
    assert len(tied) >= 2, tied                                   # the tie the reviewer found (yallah/wakha/hania)
    assert top == min(tied, key=dar.index), (top, tied)

def test_21_score_len_ratio_empty_and_analyze_ignores_alpha_records(tmp_path):
    """score() records len_ratio and empty hypotheses; analyze ignores non-default --alpha records"""
    import json, argparse, analyze as an
    m = ev.score(["", "a b c d"], ["x y z w", "a b c d"])
    assert m["empty"] == 1 and abs(m["len_ratio"] - 0.5) < 1e-9, m
    tmp = str(tmp_path)
    t = os.path.join(tmp, "t.jsonl")
    with open(t, "w") as f:
        for alpha, chrf in [(0.6, 1.0), (1.0, 2.0)]:
            f.write(json.dumps(dict(run="r", split="test", beam=5, alpha=alpha, chrf=chrf)) + "\n")
    R = an.Results(argparse.Namespace(train_results=os.path.join(tmp, "none"), test_results=t, out_dir=tmp))
    assert R.test[("r", "test", 5)]["chrf"] == 1.0

def test_22_attn_scale_divides_by_sqrt_d():
    """--attn_scale divides the attention scores by sqrt(d) and adds no parameters; default is unscaled"""
    torch.manual_seed(0)
    cfg = dict(model="lstmp", src_vocab=50, tgt_vocab=50, embed_dim=E, hidden_dim=H, dropout=0.0, mask=True,
               attn=True, no_src=False)
    a, b = build_model(cfg).to(DEVICE).eval(), build_model({**cfg, "attn_scale": True}).to(DEVICE).eval()
    b.load_state_dict(a.state_dict())
    assert nparams(a) == nparams(b) and not a.decoder.attn_scale and b.decoder.attn_scale
    src = torch.randint(4, 50, (3, 7), device=DEVICE); src[0, :2] = 0
    with torch.no_grad():
        st, enc, mask = a.encode(src)
        h, _ = a.decoder.cell(a.decoder.embedding(torch.full((3,), 2, device=DEVICE)), st)
        for m, div in [(a, 1.0), (b, h.size(-1) ** 0.5)]:
            sc = torch.bmm(enc, h.unsqueeze(2)).squeeze(2) / div
            att = torch.softmax(sc.masked_fill(~mask, float("-inf")), 1)
            ctx = torch.bmm(att.unsqueeze(1), enc).squeeze(1)
            want = m.decoder.fc_out(torch.tanh(m.decoder.W_c(torch.cat([ctx, h], 1))))
            got, _ = m.decoder.forward_step(torch.full((3,), 2, device=DEVICE), st, enc, mask)
            assert torch.allclose(got, want, atol=1e-5)

@pytest.mark.needs_data
def test_23_train_plus_has_no_val_test_leakage():
    """the extra rows of data/train_plus.csv share no EN or Darija sentence with val/test; train is kept first"""
    if not os.path.exists("data/train_plus.csv"):
        pytest.skip("data/train_plus.csv not built: python scripts/build_extra_data.py")
    from build_extra_data import key
    plus, tr = pd.read_csv("data/train_plus.csv"), pd.read_csv("data/train.csv")
    held = pd.concat([pd.read_csv("data/val.csv"), TEST])
    extra = plus.iloc[len(tr):]       # (train itself: 3.9% of test EN match train up to punctuation, original split)
    assert not set(extra["en"].map(key)) & set(held["en"].map(key))
    assert not set(extra["dar"].map(key)) & set(held["dar"].map(key))
    assert (plus["en"].head(len(tr)).values == tr["en"].values).all() and len(plus) > 4 * len(tr)

def test_24_wmc_cell_equals_appendix_equations():
    """WMC cell == appendix eqs (7)-(12) written gate by gate; parameter count = vanilla + 3H^2 per cell"""
    from darija_nmt.cells import WMCCell
    torch.manual_seed(0)
    cell = WMCCell(E, H).double()
    x, h, c = torch.randn(4, E).double(), torch.randn(4, H).double(), torch.randn(4, H).double()
    Wx, Wh = cell.W_x.weight.chunk(4, 0), cell.W_h.weight.chunk(4, 0)
    bx = cell.W_x.bias.chunk(4, 0)
    Wci, Wcf = cell.W_c.weight.chunk(2, 0)
    lin = lambda k: x @ Wx[k].T + h @ Wh[k].T + bx[k]                  # k: 0=i 1=f 2=o 3=g
    i = torch.sigmoid(lin(0) + torch.tanh(c @ Wci.T))
    f = torch.sigmoid(lin(1) + torch.tanh(c @ Wcf.T))
    g = torch.tanh(lin(3))
    c_new = f * c + i * g
    o = torch.sigmoid(lin(2) + torch.tanh(c_new @ cell.W_co.weight.T))
    h_new = o * torch.tanh(c_new)
    h2, c2 = cell(x, (h, c))
    assert torch.allclose(h2, h_new) and torch.allclose(c2, c_new)
    assert nparams(cell) == nparams(VanillaLSTMCell(E, H)) + 3 * H * H

def test_25_attention_types_and_bidir_padded_batch_equals_alone():
    """attention types (dot/general/additive) and the bidirectional encoder: padded batch == each sentence alone"""
    torch.manual_seed(0)
    base = dict(src_vocab=60, tgt_vocab=60, embed_dim=E, hidden_dim=H, dropout=0.0, mask=True, attn=True, no_src=False)
    seqs = [[5, 6, 7, 8, 9, 10], [11, 12, 3], [13, 14, 15, 3]]
    for m in ["vanilla", "lstmp", "wmc"]:
        for extra in [dict(attn_type="general"), dict(attn_type="additive"), dict(attn_type="dot", bidir=True)]:
            model = build_model({**base, "model": m, **extra}).to(DEVICE).eval()
            with torch.no_grad():
                st, enc, mask = model.encode(ev.left_pad(seqs))
                cur = torch.full((3,), 2, device=DEVICE)
                logits_b, _ = model.decoder.forward_step(cur, st, enc, mask)
                for j, sq in enumerate(seqs):
                    st1, enc1, m1 = model.encode(ev.left_pad([sq]))
                    l1, _ = model.decoder.forward_step(cur[:1], st1, enc1, m1)
                    assert torch.allclose(l1[0], logits_b[j], atol=1e-4), (m, extra, j)

def test_26_transformer_left_padding_invariant():
    """Transformer baseline: left padding does not change the output (positions count real tokens only)"""
    torch.manual_seed(0)
    model = build_model(dict(model="transformer", src_vocab=60, tgt_vocab=60, dropout=0.0)).to(DEVICE).eval()
    seqs = [[5, 6, 7, 8, 9, 10], [11, 12, 3]]
    with torch.no_grad():
        st, mem, mask = model.encode(ev.left_pad(seqs))
        cur = torch.full((2,), 2, device=DEVICE)
        lb, st = model.decoder.forward_step(cur, st, mem, mask)
        lb2, _ = model.decoder.forward_step(lb.argmax(1), st, mem, mask)
        for j, sq in enumerate(seqs):
            s1, m1, k1 = model.encode(ev.left_pad([sq]))
            l1, s1 = model.decoder.forward_step(cur[:1], s1, m1, k1)
            l2, _ = model.decoder.forward_step(lb[j:j + 1].argmax(1), s1, m1, k1)
            assert torch.allclose(l1[0], lb[j], atol=1e-4) and torch.allclose(l2[0], lb2[j], atol=1e-4)

def test_27_reverse_direction_and_bpe_dropout():
    """reverse direction swaps columns and tokenizers; BPE-dropout keeps the text and changes segmentations"""
    import sentencepiece as spm_
    fwd = TranslationDataset("data/val.csv", "data/spm_en.model", "data/spm_dar.model")
    rev = TranslationDataset("data/val.csv", "data/spm_dar.model", "data/spm_en.model", "dar", "en")
    assert rev.src[0][:-1] == fwd.tgt[0][1:-1] and rev.tgt[0][1:-1] == fwd.src[0][:-1]
    spm_.set_random_generator_seed(1)
    smp = TranslationDataset("data/val.csv", "data/spm_en.model", "data/spm_dar.model", sample_alpha=0.3)
    sp = spm_.SentencePieceProcessor(model_file="data/spm_dar.model")
    seen = set()
    for _ in range(20):
        t = smp[0][1].tolist()
        assert sp.decode(t[1:-1]) == str(smp.df["dar"].iat[0])
        seen.add(tuple(t))
    assert len(seen) > 1

def test_28_label_smoothing_reported_loss_stays_plain_ce():
    """label smoothing changes the optimised loss but the reported loss stays plain cross-entropy"""
    import torch.nn as nn
    from darija_nmt.train import run_epoch
    torch.manual_seed(0)
    model = build_model(dict(model="vanilla", src_vocab=60, tgt_vocab=60, embed_dim=E, hidden_dim=H, dropout=0.0,
                             mask=True, attn=False, no_src=False)).to(DEVICE)
    model.decoder.fc_out.weight.data *= 30                       # peaked distributions (else smoothing barely matters)
    src = torch.randint(4, 60, (4, 5)); tgt = torch.randint(4, 60, (4, 6)); tgt[:, 0] = 2
    loader = [(src, tgt, None, None)]
    plain = nn.CrossEntropyLoss(ignore_index=0)
    a = run_epoch(model, loader, plain, 60)
    b = run_epoch(model, loader, nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1), 60, report=plain)
    c = run_epoch(model, loader, nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1), 60)
    assert abs(a - b) < 1e-6 and abs(a - c) > 1e-3
