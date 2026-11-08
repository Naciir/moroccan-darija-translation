"""Test working directory. The code reads data/, checkpoints/, legacy/ relative to the CURRENT directory (the repository
root in normal use), so the whole session runs from one of:
  1. $DARIJA_NMT_ROOT, if set: a working copy with the full data/ splits and legacy/checkpoints/ (runs every test);
  2. the repository root, if data/train.csv exists there (scripts/prepare_data.py was run);
  3. otherwise a temporary copy of data/ (tokenizers) + legacy/ with tests/fixtures/mini_*.csv as the splits
     (40 rows each): tests marked needs_data / needs_checkpoints are skipped.
"""
import os, shutil, sys, tempfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]          # also works without `pip install -e .`
os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src")] + ([os.environ["PYTHONPATH"]]
                                                                   if os.environ.get("PYTHONPATH") else []))
_TMP = None

def _fixture_workdir():
    tmp = Path(tempfile.mkdtemp(prefix="darija_nmt_tests_"))
    shutil.copytree(ROOT / "data", tmp / "data")
    if (ROOT / "legacy").exists():
        shutil.copytree(ROOT / "legacy", tmp / "legacy")
    for s in ["train", "val", "test"]:
        shutil.copy(FIX / f"mini_{s}.csv", tmp / "data" / f"{s}.csv")
    train = pd.read_csv(tmp / "data" / "train.csv")
    for col in ["en", "dar"]:                                      # SentencePiece input, as prepare_data.py writes it
        with open(tmp / "data" / f"train.{col}.txt", "w", encoding="utf-8", newline="") as f:
            f.writelines(str(s).replace("\n", " ") + "\r\n" for s in train[col])
    return tmp

def pytest_configure(config):
    global _TMP
    if os.environ.get("DARIJA_NMT_ROOT"):
        work = Path(os.environ["DARIJA_NMT_ROOT"]).resolve()
    elif (ROOT / "data" / "train.csv").exists():
        work = ROOT
    else:
        work = _TMP = _fixture_workdir()
    os.chdir(work)

def pytest_unconfigure(config):
    if _TMP is not None:
        os.chdir(ROOT)
        shutil.rmtree(_TMP, ignore_errors=True)

def full_data():
    return os.path.exists("data/train.csv") and len(pd.read_csv("data/train.csv")) > 1000

def has_checkpoints():
    return os.path.exists("legacy/checkpoints/vanilla_base_s0.pt")

def pytest_report_header(config):
    return f"darija_nmt workdir: {os.getcwd()} | full data: {full_data()} | legacy checkpoints: {has_checkpoints()}"

def pytest_collection_modifyitems(config, items):
    full, ck = full_data(), has_checkpoints()
    for it in items:
        if "needs_data" in it.keywords and not full:
            it.add_marker(pytest.mark.skip(reason="full data/ splits missing (python scripts/prepare_data.py)"))
        if "needs_checkpoints" in it.keywords and not ck:
            it.add_marker(pytest.mark.skip(reason="legacy/checkpoints/*.pt not present"))
