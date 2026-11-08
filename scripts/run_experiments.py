"""The full experiment grid: darija_nmt.train then darija_nmt.evaluate per run, N runs in parallel on the GPU.
Resumable: a run is skipped when its checkpoint + a results record with the same epochs/patience exist
(same for each evaluation; a retrained run is always re-evaluated). Run from the repository root.

    python scripts/run_experiments.py --jobs 3              # everything
    python scripts/run_experiments.py --only main,attn --seeds 0
    python scripts/run_experiments.py --dry_run
"""
import argparse, json, os, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

# Ablation arms run WITHOUT bucketing: random batches like the original train.py, so the unmasked arms
# keep the original ~65% leading-pad exposure (bucketing alone would hide the pad bug: ~7% pads) and
# train/val see the same pad regime. old->mask = mask effect, old->tok = tokenizer effect (single factor);
# main adds bucketing on top (with the mask on it only changes batch composition).
GRID = {  # tag: (cells, extra train.py args, beam-5 eval too?)
    "old":   (["vanilla"], ["--tok_dir", "data/tok_orig", "--no_mask", "--no_bucket"], False),
    "tok":   (["vanilla"], ["--tok_dir", "data", "--no_mask", "--no_bucket"], False),
    "mask":  (["vanilla"], ["--tok_dir", "data/tok_orig", "--no_bucket"], False),
    "main":  (["vanilla", "cifg", "lstmp", "wmc"], ["--tok_dir", "data"], True),
    "nosrc": (["vanilla"], ["--tok_dir", "data", "--no_src"], False),
    "attn":  (["vanilla", "cifg", "lstmp", "wmc"], ["--tok_dir", "data", "--attn"], True),
    "mono":  (["vanilla"], ["--tok_dir", "data/tok_mono"], False),
}
# Hyperparameter study (brief: lr / optimizer / init / batch size; + dropout): one factor at a time around the
# default (Adam 1e-3, PyTorch default init, batch 64, dropout 0.4), on the vanilla cell WITH attention (the best
# architecture), 20 epochs / patience 4 to fit the budget. hp_base is run with 3 seeds = the noise bar.
HP_BASE = ["--tok_dir", "data", "--attn", "--epochs", "20", "--patience", "4"]
HP = {"base": [], "lr3e-4": ["--lr", "3e-4"], "lr3e-3": ["--lr", "3e-3"], "adamw": ["--opt", "adamw"],
      "rmsprop": ["--opt", "rmsprop"], "sgd": ["--opt", "sgd", "--lr", "0.05"], "xavier": ["--init", "xavier"],
      "orthogonal": ["--init", "orthogonal"], "bs32": ["--bs", "32"], "bs128": ["--bs", "128"],
      "do0.2": ["--dropout", "0.2"], "do0.5": ["--dropout", "0.5"]}
GRID.update({f"hp_{k}": (["vanilla"], HP_BASE + v, False) for k, v in HP.items()})
# Data scaling: + 31,661 leakage-filtered DODa pairs (build_extra_data.py); same val/test and tokenizers.
PLUS = ["--tok_dir", "data", "--train_csv", "data/train_plus.csv", "--epochs", "25", "--patience", "4"]
# LSTMP + dot attention saturates (normalised attention entropy 0.07 vs 0.40 for vanilla): scaled dot product fix
GRID.update({"attn_scaled": (["lstmp", "vanilla"], ["--tok_dir", "data", "--attn", "--attn_scale"], True)})
# attention score functions with a learned map (they set their own scale): every cell, 3 seeds
GRID.update({f"attn_{t}": (["vanilla", "cifg", "lstmp", "wmc"],
                           ["--tok_dir", "data", "--attn", "--attn_type", t], False)
             for t in ["general", "additive"]})
# tokenizer study on the 42k corpus (vanilla + attention, 1 seed): BPE 3k/6k/10k retrained on train_plus.csv;
# compared on VALIDATION chrF (val loss is per BPE token, so not comparable across vocabularies)
GRID.update({f"tokp{k}": (["vanilla"], PLUS + ["--attn", "--tok_dir", f"data/tok_plus{k}"], False)
             for k in ["3k", "6k", "10k"]})
# context row: a small Transformer (nn.Transformer, 3+3 layers, d 256), same data / tokenizer / early stopping
GRID.update({"trf": (["transformer"], ["--tok_dir", "data", "--lr", "5e-4", "--dropout", "0.3"], False)})
# reverse model (Darija -> English) for back-translation, on the 42k corpus
GRID.update({"rev": (["vanilla"], PLUS + ["--attn", "--direction", "dar2en"], False)})
GRID.update({"plus": (["vanilla"], PLUS, False), "attn_plus": (["vanilla", "cifg", "lstmp"], PLUS + ["--attn"], True)})
# tags decided at run time (e.g. by a scheduler once the tokenizer / recipe is chosen): {tag: [cells, args, beam5]}
if os.path.exists("grid_extra.json"):
    with open("grid_extra.json") as _f:
        GRID.update({k: (v[0], v[1], v[2]) for k, v in json.load(_f).items()})
LOCK = threading.Lock()

def records(path, key):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {key(json.loads(l)) for l in f if l.strip()}

def last_records(path):
    """run -> its LAST training record (a later retrain supersedes an earlier one, as in analyze.py)."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {r["run"]: r for r in map(json.loads, filter(str.strip, f))}

def wanted(train_args):
    """epochs/patience the train.py calls of THIS invocation will use (train.py defaults unless overridden)."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--epochs", type=int, default=40); p.add_argument("--patience", type=int, default=5)
    w, _ = p.parse_known_args(train_args.split())
    return w.epochs, w.patience

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3,
                    help="limit by HOST commit memory (~3 GB per torch process), not GPU memory (~250 MB per run)")
    ap.add_argument("--retries", type=int, default=1, help="re-run a failed command this many times")
    ap.add_argument("--retry_wait", type=float, default=30, help="seconds before a retry (lets host memory free up)")
    ap.add_argument("--only", default=",".join(GRID), help="comma-separated tags")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--cells", default=None, help="comma-separated subset of each tag's cells (default: all)")
    ap.add_argument("--keep_locks", action="store_true",
                    help="do not remove result locks at startup (other run_experiments processes are writing)")
    ap.add_argument("--runs_dir", default="checkpoints")
    ap.add_argument("--results", default="results.jsonl")
    ap.add_argument("--test_results", default="results_test.jsonl")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--train_args", default="", help="extra args passed to every train.py call (e.g. '--epochs 1')")
    a = ap.parse_args()
    py = sys.executable

    # a run counts as trained only if its record was made with the SAME epochs/patience as now: a shortened run
    # (e.g. --train_args '--epochs 1' into the default files) must not be mistaken for the real one
    prev = last_records(a.results)
    # a tag may set its own --epochs/--patience (HP study): compare against what THIS run's train.py call will use
    done = lambda run, want: run in prev and (prev[run].get("epochs"), prev[run].get("patience")) == want
    # an eval counts only if it scored the CURRENT checkpoint (a stale record from an older .pt must not count);
    # an evaluate.py --alpha ablation is not the grid eval
    mtime = lambda run: os.path.getmtime(f"{a.runs_dir}/{run}.pt") if os.path.exists(f"{a.runs_dir}/{run}.pt") else None
    # (2.5 s tolerance: zip archives store file times at 2-second resolution, so an unzipped checkpoint keeps
    # its evaluations; a retrained checkpoint is minutes newer)
    same = lambda r: mtime(r["run"]) is not None and abs(r["ckpt_mtime"] - mtime(r["run"])) <= 2.5
    done_eval = records(a.test_results, lambda r: (r["run"], r["split"], r["beam"])
                        if r.get("alpha") in (None, 0.6) and r.get("ckpt_mtime") is not None and same(r) else None)
    jobs = []
    for tag in a.only.split(","):
        cells, extra, beam5 = GRID[tag]
        if tag == "mono" and not os.path.exists("data/tok_mono/spm_dar.model"):
            print("data/tok_mono missing -> skipping mono"); continue
        want = wanted(" ".join(extra + [a.train_args]))      # later flags win, as in argparse
        for seed in map(int, a.seeds.split(",")):
            for m in cells:
                if a.cells and m not in a.cells.split(","):
                    continue
                run = f"{m}_{tag}_s{seed}"
                cmds = []
                retrain = not (os.path.exists(f"{a.runs_dir}/{run}.pt") and done(run, want))
                if retrain:
                    if run in prev:
                        print(f"[{run}] existing record has epochs/patience "
                              f"{prev[run].get('epochs')}/{prev[run].get('patience')} != "
                              f"{want[0]}/{want[1]} -> retrain")
                    cmds.append([py, "-m", "darija_nmt.train", "--model", m, "--tag", tag, "--seed", str(seed), *extra,
                                 "--runs_dir", a.runs_dir, "--results", a.results, *a.train_args.split()])
                for beam in ([1, 5] if beam5 else [1]):
                    if retrain or (run, "test", beam) not in done_eval:    # new checkpoint -> re-evaluate
                        cmds.append([py, "-m", "darija_nmt.evaluate", "--runs", run, "--runs_dir", a.runs_dir,
                                     "--beam", str(beam),
                                     "--results", a.test_results, "--show", "0"])
                if cmds:
                    jobs.append((run, cmds))
    if not records(a.test_results, lambda r: r["run"]) & {"baseline_retrieval"}:
        jobs.append(("baselines", [[py, "-m", "darija_nmt.baselines", "--results", a.test_results]]))

    print(f"{len(jobs)} jobs, {a.jobs} in parallel")
    if a.dry_run:
        for run, cmds in jobs:
            for c in cmds:
                print(f"[{run}]", " ".join(c[1:]))
        return
    os.makedirs("logs", exist_ok=True)
    for p in (a.results, a.test_results):        # no writer is running yet: a lock now is left by a killed process
        if not a.keep_locks and os.path.exists(p + ".lock"):
            os.remove(p + ".lock"); print("removed stale", p + ".lock")

    def work(job):
        run, cmds = job
        with open(f"logs/{run}.log", "a", encoding="utf-8") as log:
            for c in cmds:
                for attempt in range(1 + a.retries):     # transient CUDA / host-commit-memory errors: retry
                    if attempt:
                        # rc = exit code of the previous attempt (set below before any retry)
                        log.write(f"# exit {rc}, retry {attempt}/{a.retries} in {a.retry_wait} s\n")  # noqa: F821
                        log.flush(); time.sleep(a.retry_wait)
                    log.write("$ " + " ".join(c[1:]) + "\n"); log.flush()
                    rc = subprocess.run(c, stdout=log, stderr=subprocess.STDOUT,
                                        env={**os.environ, "PYTHONIOENCODING": "utf-8"}).returncode
                    if not rc:
                        break
                if rc:
                    with LOCK:
                        print(f"FAILED {run}: {' '.join(c[1:])} (see logs/{run}.log)", flush=True)
                    return run, False
        with LOCK:
            print(f"finished {run}", flush=True)
        return run, True

    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(work, jobs))
    failed = [r for r, ok in res if not ok]
    if failed:                                   # non-zero exit: the grid has holes; re-run this script to fill them
        print(f"{len(failed)} failed: {failed}\nre-run scripts/run_experiments.py (resumable) to finish them")
        sys.exit(1)
    print("all done\nnext: python scripts/analyze.py")

if __name__ == "__main__":
    main()
