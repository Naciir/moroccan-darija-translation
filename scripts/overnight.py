"""Experiment scheduler for the extended study: a small DAG of jobs, at most N training processes at once, with the
two decisions (tokenizer, final recipe) taken automatically from VALIDATION chrF.

    python scripts/overnight.py     # from the repository root; log night.log, decisions night_decisions.json
    NIGHT_JOBS=2 NIGHT_HOURS=7.5 python scripts/overnight.py     # 2 parallel runs, no new run after 7.5 h
Every unit of work is `run_experiments.py --only TAG --seeds S --cells M` (resumable, retries, evaluates on test),
so a killed night can simply be restarted: finished runs are skipped."""
import json, os, subprocess, sys, time, datetime, traceback

ROOT = os.environ.get("NIGHT_ROOT") or os.getcwd()
DRY = os.environ.get("NIGHT_DRY") == "1"          # test the scheduling logic with 1-second stub jobs
os.chdir(ROOT)
PY = sys.executable
N = int(os.environ.get("NIGHT_JOBS", "2"))
LOG, DEC = "night.log", "night_decisions.json"
CELLS4 = ["vanilla", "cifg", "lstmp", "wmc"]
DEADLINE = time.time() + float(os.environ.get("NIGHT_HOURS", "10000")) * 3600   # no NEW training after this
SWITCH = 0.3          # a candidate must beat the default by this much VALIDATION chrF to be adopted (1 seed)

def log(msg):
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)

def load(path, default):
    return json.load(open(path)) if os.path.exists(path) else default

decisions = load(DEC, {})
def decide(key, value):
    decisions[key] = value
    json.dump(decisions, open(DEC, "w"), indent=1)
    log(f"DECISION {key} = {value}")

def add_grid(entries):
    g = load("grid_extra.json", {})
    g.update(entries)
    json.dump(g, open("grid_extra.json", "w"), indent=1)

def val_chrf(run):
    """validation chrF (greedy) of a run; evaluates it if needed. None if the checkpoint is missing."""
    if not os.path.exists(f"checkpoints/{run}.pt"):
        return None
    def lookup():
        best = None
        if os.path.exists("results_val.jsonl"):
            for l in open("results_val.jsonl"):
                r = json.loads(l)
                if r["run"] == run and r["split"] == "val" and r["beam"] == 1 and \
                        abs(r.get("ckpt_mtime", 0) - os.path.getmtime(f"checkpoints/{run}.pt")) <= 2.5:
                    best = r["chrf"]
        return best
    v = lookup()
    for attempt in range(40):          # host commit memory can be exhausted while 3 runs train (WinError 1455): retry
        if v is not None or DRY:
            break
        r = subprocess.run([PY, "-m", "darija_nmt.evaluate", "--runs", run, "--split", "val",
                            "--results", "results_val.jsonl", "--show", "0"], capture_output=True, text=True)
        v = lookup()
        if v is None:
            log(f"val eval of {run} failed (attempt {attempt + 1}): {(r.stderr or '').strip().splitlines()[-1:]}")
            time.sleep(60)
    log(f"val chrF {run}: {v}")
    return v

class Job:
    def __init__(self, name, cmd=None, fn=None, deps=(), prio=50):
        self.name, self.cmd, self.fn, self.deps, self.prio = name, cmd, fn, list(deps), prio

def rx(tag, seed, cell, prio, deps=()):
    return Job(f"{tag}:{cell}:s{seed}", [PY, "-u", "scripts/run_experiments.py", "--only", tag, "--seeds", str(seed),
                                        "--cells", cell, "--jobs", "1", "--retries", "2", "--keep_locks"],
               deps=deps, prio=prio)

jobs, running, finished = {}, {}, {}
def add(job):
    if job.name not in jobs and job.name not in finished:
        jobs[job.name] = job

# ---------------- decisions -------------------------------------------------------------------------------------
def base_args(tok):
    return ["--tok_dir", tok, "--train_csv", "data/train_plus.csv", "--epochs", "25", "--patience", "4"]

ABL = {"abl_bidir": ["--bidir"], "abl_ls": ["--ls", "0.1"], "abl_bpe": ["--bpe_alpha", "0.1"],
       "abl_lr": ["--lr", "3e-3"], "abl_bs32": ["--bs", "32"]}

def decide_tok():
    cands = {"data": "vanilla_attn_plus_s0", "data/tok_plus3k": "vanilla_tokp3k_s0",
             "data/tok_plus6k": "vanilla_tokp6k_s0", "data/tok_plus10k": "vanilla_tokp10k_s0"}
    scores = {k: val_chrf(r) for k, r in cands.items()}
    decide("tok_val_chrf", scores)
    ref = scores.get("data") or 0.0
    ok = {k: v for k, v in scores.items() if v is not None}
    best = max(ok, key=ok.get) if ok else "data"
    tok = best if best == "data" or ok[best] >= ref + SWITCH else "data"
    decide("tok", tok)
    b = base_args(tok)
    add_grid({
        "c42": [CELLS4, b + ["--attn", "--attn_type", "additive"], False],          # the central 42k table
        "n42": [CELLS4, b, False],                                                   # no-attention anchor
        "trf42": [["transformer"], b + ["--lr", "5e-4", "--dropout", "0.3"], False],
        "bt": [["vanilla"], b + ["--attn", "--attn_type", "additive", "--train_csv", "data/train_bt.csv"], False],
        **{k: [["vanilla"], b + ["--attn", "--attn_type", "additive"] + v, False] for k, v in ABL.items()},
    })
    for m in CELLS4:
        add(rx("c42", 0, m, 10))
    for k in ABL:
        add(rx(k, 0, "vanilla", 20))
    add(rx("bt", 0, "vanilla", 25, deps=["bt_data"]))
    for s in (1, 2):
        for m in CELLS4:
            add(rx("c42", s, m, 30 + s))
    for m in CELLS4:
        add(rx("n42", 0, m, 45))
    add(rx("trf42", 0, "transformer", 60))
    add(Job("decide_final", fn=decide_final, prio=5,
            deps=["c42:vanilla:s0", "bt:vanilla:s0"] + [f"{k}:vanilla:s0" for k in ABL]))
    log("STAGE tok_decided")

def decide_final():
    tok = decisions.get("tok", "data")
    base = val_chrf("vanilla_c42_s0") or 0.0
    gains, flags = {}, []
    for k, v in ABL.items():
        s = val_chrf(f"vanilla_{k}_s0")
        gains[k] = None if s is None else round(s - base, 2)
        if s is not None and s >= base + SWITCH:
            flags += v
    bt = val_chrf("vanilla_bt_s0")
    gains["bt"] = None if bt is None else round(bt - base, 2)
    use_bt = bt is not None and bt >= base + SWITCH
    decide("final_gains_vs_c42_vanilla_val", gains)
    if any(v is None for v in gains.values()):
        # an ablation / back-translation run is missing (failed or deferred): no final decision yet, so a restart
        # takes it once they exist; meanwhile the base recipe (c42) is the provisional best system
        log("final recipe NOT decided: missing results " + str([k for k, v in gains.items() if v is None]))
        add(Job("alpha", [PY, "scripts/tune_alpha.py", "--runs", "vanilla_c42_s0", "cifg_c42_s0"],
                deps=["c42:cifg:s0", "c42:vanilla:s0"], prio=80))
        return
    args = base_args(tok) + ["--attn", "--attn_type", "additive"] + flags + \
        (["--train_csv", "data/train_bt.csv"] if use_bt else [])
    decide("final_args", args)
    if not flags and not use_bt:
        log("no ablation beat the base recipe on validation: the final recipe IS c42 (no extra runs)")
        decide("final_tag", "c42")
    else:
        add_grid({"fin": [["vanilla", "cifg", "wmc", "lstmp"], args, False]})
        decide("final_tag", "fin")
        for s in (0, 1, 2):
            for i, m in enumerate(["vanilla", "cifg"]):
                add(rx("fin", s, m, 35 + s))
        for s in (0, 1, 2):
            for m in ["wmc", "lstmp"]:
                add(rx("fin", s, m, 70 + s))
    add(Job("alpha", [PY, "scripts/tune_alpha.py", "--runs", f"vanilla_{decisions['final_tag']}_s0",
                      f"cifg_{decisions['final_tag']}_s0"], deps=[f"{decisions['final_tag']}:cifg:s0",
                                                                  f"{decisions['final_tag']}:vanilla:s0"], prio=80))
    log("STAGE final_decided")

def analysis():
    for c in [["analyze.py"], ["extras.py"], ["figures.py"], ["attention_stats.py"]]:
        r = subprocess.run([PY, "scripts/" + c[0]], stdout=open(f"logs/night_{c[0]}.log", "w"),
                           stderr=subprocess.STDOUT)
        log(f"{c[0]} exit {r.returncode}")

# ---------------- initial jobs ----------------------------------------------------------------------------------
for k in ["3k", "6k", "10k"]:
    add(rx(f"tokp{k}", 0, "vanilla", 0))
add(rx("rev", 0, "vanilla", 1))
add(Job("bt_data", [PY, "scripts/backtranslate.py", "--run", "vanilla_rev_s0"], deps=["rev:vanilla:s0"], prio=2))
add(Job("decide_tok", fn=decide_tok, deps=["tokp3k:vanilla:s0", "tokp6k:vanilla:s0", "tokp10k:vanilla:s0"], prio=3))
for s in (0, 1, 2):
    add(rx("trf", s, "transformer", 50))
if "tok" in decisions:                       # restarted after the decision: rebuild the dependent jobs
    finished["decide_tok"] = 0; jobs.pop("decide_tok", None); decide_tok()
if "final_tag" in decisions:
    finished["decide_final"] = 0; jobs.pop("decide_final", None); decide_final()

# ---------------- scheduler -------------------------------------------------------------------------------------
log(f"night start: {len(jobs)} jobs, {N} parallel")
os.makedirs("logs", exist_ok=True)
while jobs or running:
    for name, (p, fh) in list(running.items()):
        if p.poll() is not None:
            fh.close(); finished[name] = p.returncode; del running[name]
            log(f"{'done' if p.returncode == 0 else 'FAILED'} {name} (exit {p.returncode}); running {len(running)}")
    ready = sorted((j for j in jobs.values() if all(d in finished for d in j.deps)), key=lambda j: j.prio)
    for j in ready:
        if j.fn is not None:                  # decisions run inline (they may add jobs)
            del jobs[j.name]
            try:
                j.fn(); finished[j.name] = 0
                break                         # it may have added higher-priority jobs: re-rank
            except Exception:
                finished[j.name] = 1; log(f"FAILED {j.name}:\n{traceback.format_exc()}")
        elif time.time() > DEADLINE:
            del jobs[j.name]; finished[j.name] = 2
            log(f"DEFERRED (deadline) {j.name}: run it later, everything is resumable")
        elif len(running) < N:
            del jobs[j.name]
            fh = open(f"logs/night_{j.name.replace(':', '_')}.log", "a", encoding="utf-8")
            cmd = [PY, "-c", "import time; time.sleep(1)"] if DRY else j.cmd
            running[j.name] = (subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT), fh)
            log(f"start {j.name} (prio {j.prio}); running {len(running)}, queued {len(jobs)}")
    if not running and jobs and not any(all(d in finished for d in j.deps) for j in jobs.values()):
        log(f"STUCK: unmet dependencies for {sorted(jobs)}"); break
    time.sleep(1 if DRY else 15)
log("STAGE training_done")
if not DRY:
    analysis()
log("ALL DONE")
