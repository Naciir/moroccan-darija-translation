"""Run ONCE right after unzipping a packaged working state.

Every test/validation record stores the modification time of the checkpoint it scored (`ckpt_mtime`), so that a
retrained checkpoint is re-evaluated. Some unzip tools (Python's zipfile, some archive managers) give every file the
extraction time instead, which would make every evaluation look stale. This script restores the original times
from checkpoints/MTIMES.json and then deletes the manifest, so it can never be applied twice.

    python scripts/restore_mtimes.py
    python scripts/restore_mtimes.py --make        # (packaging side) write the manifest
"""
import json, os, sys

MANIFEST = "checkpoints/MTIMES.json"

def main():
    if "--make" in sys.argv:
        times = {f: os.path.getmtime(os.path.join("checkpoints", f))
                 for f in os.listdir("checkpoints") if f.endswith(".pt")}
        json.dump(times, open(MANIFEST, "w"), indent=1)
        print(f"wrote {MANIFEST}: {len(times)} checkpoints"); return
    if not os.path.exists(MANIFEST):
        print("no manifest (already applied, or not a packaged state): nothing to do"); return
    times = json.load(open(MANIFEST))
    n = 0
    for f, t in times.items():
        p = os.path.join("checkpoints", f)
        if os.path.exists(p):
            os.utime(p, (t, t)); n += 1
    os.remove(MANIFEST)
    print(f"restored the modification time of {n} checkpoints; manifest removed")

if __name__ == "__main__":
    main()
