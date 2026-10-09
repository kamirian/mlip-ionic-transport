"""Run parity_predictions.py for every entry of manifests/parity_runs.json of one framework.

Usage: python scripts/run_parity.py manifests/parity_runs.json <mace|chgnet> <data_root> <out_dir> [key]
Runs whose output already exists are skipped; give a key to run a single entry.
"""
import glob
import json
import os
import subprocess
import sys

manifest, framework, root, out = sys.argv[1:5]
only = sys.argv[5] if len(sys.argv) > 5 else None
cfg = json.load(open(manifest))
os.makedirs(out, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))
for run in cfg[framework]:
    d = os.path.join(root, run["dir"])
    target = os.path.join(out, run["key"] + ".npz")
    if (only and run["key"] != only) or os.path.exists(target):
        continue
    if framework == "mace":
        files = ([os.path.join(root, run["xyz"].format(split=s)) for s in ("train", "val", "test")] if "xyz" in run
                 else [os.path.join(d, run[s]) for s in ("train", "valid", "test")])
        cmd = [sys.executable, f"{here}/parity_predictions.py", "mace", os.path.join(d, run["model"]), *files, target]
    else:
        log = sorted(glob.glob(os.path.join(d, "slurm-*.out")))[-1]
        cmd = [sys.executable, f"{here}/parity_predictions.py", "chgnet", os.path.join(d, run["checkpoint"]),
               os.path.join(root, cfg["chgnet_frames_csv"]), str(run["n_rows"]), log, target]
    print("###", run["key"], flush=True)
    subprocess.run(cmd, check=True)
