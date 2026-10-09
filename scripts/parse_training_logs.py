"""Collect per-epoch training/validation curves and final test errors from existing logs.

CHGNet (chgnet.trainer.Trainer stdout, one slurm-*.out per run):
  "Epoch: [k][i/N] ... | Loss a(avg) | MAE e a(avg)  f a(avg)"  -> last line of epoch k = epoch average
  "*   e_MAE (x)  f_MAE (y)"                                      -> validation MAE after each epoch
  "**  e_MAE (x)  f_MAE (y)"                                      -> test-set MAE (end of run)
  If a folder has several logs, the newest one that contains a "**" line is used.
MACE (run-42_train.txt JSON lines + slurm-*.out error tables):
  mode "opt"  -> training loss per batch (averaged per epoch)
  mode "eval" -> validation metrics per epoch (epoch null = before training)

Usage: python scripts/parse_training_logs.py results/training_runs.json
"""

import glob
import json
import os
import re
import sys
from collections import defaultdict

SCRATCH = "/path/to/data"
DG = f"{SCRATCH}/RCHnet/JOBS/LYC_fiinetune2/data_grabber"
NEW = f"{DG}/NEW_materials_NTGO_NTPO_NGSO/Fine_tune_models_different_selection/correct_each_structure_10000"
MT = f"{SCRATCH}/MACE_env/jobs/Training"
NA2O = f"{SCRATCH}/MACE_env/jobs/NEB/Na2O/train_scripts_without_stresses"

# (study, label, folder, extra info). Labels describe the actual training data (fix tracker B6).
CHGNET_RUNS = [
    ("chgnet_lyc_data_size", "first 7,980 frames (900 K block), 10 epochs", f"{DG}/Fine_tune/10_epochs", {"n_frames": 7980, "epochs": 10, "batch": 8}),
    ("chgnet_lyc_data_size", "first 15,960 frames (900 K block), 10 epochs", f"{DG}/Fine_tune/25_itteration_10_epochs", {"n_frames": 15960, "epochs": 10, "batch": 8}),
    ("chgnet_lyc_data_size", "first 30,693 frames (900 K block), 10 epochs", f"{DG}/Fine_tune/13_itteration_10_epochs", {"n_frames": 30693, "epochs": 10, "batch": 8}),
    ("chgnet_lyc_epochs", "first 7,980 frames (900 K block), 200 epochs", f"{DG}/Fine_tune/200_epochs", {"n_frames": 7980, "epochs": 200, "batch": 8}),
    ("chgnet_lyc_epochs", "first 15,960 frames (900 K block), 20 epochs", f"{DG}/Fine_tune/25_itteration_20_epochs", {"n_frames": 15960, "epochs": 20, "batch": 8}),
    ("chgnet_lyc_histogram", "energy-histogram 8,121 frames", f"{DG}/Fine_tune/histogram_8000-first-try", {"n_frames": 8121, "epochs": 10, "batch": 8}),
    ("chgnet_lyc_histogram", "energy-histogram 16,279 frames", f"{DG}/Fine_tune/histogram_16000-first-try", {"n_frames": 16279, "epochs": 10, "batch": 8}),
    ("chgnet_lyc_histogram", "energy-histogram 30,000 frames", f"{DG}/Fine_tune/histogram_Total", {"n_frames": 30000, "epochs": 10, "batch": 8}),
    ("chgnet_lyc_histogram", "energy-histogram 30,222 frames", f"{DG}/Fine_tune/histogram_Total_new", {"n_frames": 30222, "epochs": 10, "batch": 8}),
    ("chgnet_lyc_batch_size", "energy-histogram 16,279 frames, batch 16", f"{DG}/Fine_tune/different_batch_sizes/histogram_16000-first-try_batch_16", {"n_frames": 16279, "epochs": 10, "batch": 16}),
    ("chgnet_lyc_batch_size", "energy-histogram 16,279 frames, batch 32", f"{DG}/Fine_tune/different_batch_sizes/histogram_16000-first-try_batch_32", {"n_frames": 16279, "epochs": 10, "batch": 32}),
    ("chgnet_lyc_batch_size", "energy-histogram 30,222 frames, batch 16", f"{DG}/Fine_tune/different_batch_sizes/histogram_Total_new_16", {"n_frames": 30222, "epochs": 10, "batch": 16}),
    ("chgnet_lyc_batch_size", "energy-histogram 30,222 frames, batch 32", f"{DG}/Fine_tune/different_batch_sizes/histogram_Total_new_32", {"n_frames": 30222, "epochs": 10, "batch": 32}),
    ("chgnet_na_oxides", "NTPO (Na8Ti8P8O40), 1150 K block held out", f"{NEW}/NTPO_batch32", {"epochs": 10, "batch": 32}),
    ("chgnet_na_oxides", "NSGO (Na12Ge12Sb16P4O80), 1150 K block held out", f"{NEW}/NSGO_batch32", {"epochs": 10, "batch": 32}),
]
MACE_RUNS = [
    ("mace_lyc", "LYC from scratch", f"{MT}/lyc_nvt/from_scratch_training", "first_train.sh"),
    ("mace_lyc", "LYC fine-tuned from MACE-MPA-0", f"{MT}/lyc_nvt/fine_tuning_improved_model", "first_train.sh"),
] + [
    (f"mace_{m}", f"{m} {kind}", f"{MT}/{m}_nvt_training/{folder}", "first_train.sh")
    for m in ("20258", "45802", "75421")
    for kind, folder in (("from scratch", "from_scratch_training"), ("fine-tuned from MACE-MPA-0", "fine_tuning_improved_model"))
] + [
    ("mace_na2o_neb_strategies", os.path.basename(d), d, os.path.basename(d) + "_train.sh")
    for d in sorted(glob.glob(f"{NA2O}/strategy*"))
]

RE_EPOCH = re.compile(r"^Epoch: \[(\d+)\]\[(\d+)/(\d+)\].*?Loss [\d.]+\(([\d.]+)\).*?MAE e [\d.]+\(([\d.]+)\)\s+f [\d.]+\(([\d.]+)\)")
RE_STAR = re.compile(r"^(\*{1,2})\s+e_MAE \(([\d.]+)\)\s+f_MAE \(([\d.]+)\)")


def parse_chgnet_log(path):
    train = {}
    val, test = [], None
    with open(path, errors="replace") as fh:
        for line in fh:
            m = RE_EPOCH.match(line)
            if m:
                k = int(m.group(1))
                train[k] = {"loss": float(m.group(4)), "e_mae": float(m.group(5)), "f_mae": float(m.group(6)),
                            "batch": int(m.group(2)), "n_batches": int(m.group(3))}
                continue
            m = RE_STAR.match(line)
            if m:
                rec = {"e_mae": float(m.group(2)), "f_mae": float(m.group(3))}
                if m.group(1) == "**":
                    test = rec
                else:
                    val.append(rec)
    epochs = sorted(train)
    return {
        "units": {"e_mae": "eV/atom", "f_mae": "eV/A", "loss": "MSE (CHGNet default weights)"},
        "train_epoch_avg": [dict(epoch=k, **train[k]) for k in epochs],
        "val_per_epoch": [dict(epoch=i, **v) for i, v in enumerate(val)],
        "test": test,
    }


def chgnet_run(study, label, folder, info):
    logs = sorted(glob.glob(f"{folder}/slurm-*.out"), key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0]))
    parsed = [(p, parse_chgnet_log(p)) for p in logs]
    with_test = [(p, d) for p, d in parsed if d["test"] is not None]
    chosen = with_test[-1] if with_test else (parsed[-1] if parsed else (None, None))
    script = next(iter(sorted(glob.glob(f"{folder}/*.py"))), None)
    return {"framework": "CHGNet", "study": study, "label": label, "folder": folder, **info,
            "base_model": "CHGNet v0.3.0 pretrained (412,525 parameters)",
            "log_used": chosen[0], "other_logs": [p for p, _ in parsed if p != chosen[0]],
            "training_script": open(script).read() if script else None,
            **(chosen[1] or {})}


TABLE_ROW = re.compile(r"^\|\s*(\S+)\s*\|(.+)\|\s*$")


def parse_mace_tables(path):
    """Error tables printed at the end of MACE training (train/valid and test)."""
    tables, current, header = {}, None, None
    with open(path, errors="replace") as fh:
        for line in fh:
            if "Error-table on" in line:
                current, header = ("test" if "TEST" in line else "train_valid"), None
                tables[current] = []
            elif current and line.startswith("+"):
                continue
            elif current and line.startswith("|"):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if header is None:
                    header = cells
                else:
                    tables[current].append(dict(zip(header, cells)))
            else:
                current = None
    return {k: v for k, v in tables.items() if v}


def mace_run(study, label, folder, cmd_file):
    train_txt = sorted(glob.glob(f"{folder}/MACE_models/*_train.txt"))
    opt = defaultdict(list)
    evals = []
    if train_txt:
        with open(train_txt[0]) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("mode") == "opt":
                    opt[r["epoch"]].append(r["loss"])
                elif r.get("mode") == "eval":
                    evals.append({k: r.get(k) for k in ("epoch", "loss", "mae_e_per_atom", "rmse_e_per_atom",
                                                       "mae_f", "rmse_f", "rel_mae_f", "rel_rmse_f",
                                                       "mae_stress", "rmse_stress")})
    logs = sorted(glob.glob(f"{folder}/slurm-*.out"), key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0]))
    tables, log_used = {}, None
    for p in logs:
        t = parse_mace_tables(p)
        if t.get("test"):
            tables, log_used = t, p
    cmd = os.path.join(folder, cmd_file)
    return {"framework": "MACE", "study": study, "label": label, "folder": folder,
            "train_txt": train_txt[0] if train_txt else None, "log_used": log_used,
            "units": {"loss": "MACE weighted loss", "mae_e_per_atom": "eV/atom", "mae_f": "eV/A",
                      "rel_mae_f": "%", "final_tables": "as printed (meV/atom, meV/A)"},
            "train_loss_epoch_mean": [{"epoch": k, "loss": sum(v) / len(v), "n_batches": len(v)}
                                      for k, v in sorted(opt.items())],
            "val_per_epoch": evals,
            "final_error_tables": tables,
            "training_command": open(cmd).read() if os.path.exists(cmd) else None}


def main(out):
    runs = [chgnet_run(*r) for r in CHGNET_RUNS] + [mace_run(*r) for r in MACE_RUNS]
    with open(out, "w") as fh:
        json.dump({"schema_version": "1.0", "description": "Per-epoch training/validation curves and final "
                   "test errors parsed from the original training logs (no retraining).", "runs": runs}, fh, indent=1)
    for r in runs:
        if r["framework"] == "CHGNet":
            n, t = len(r.get("train_epoch_avg", [])), r.get("test")
            print(f"CHGNet {r['label']:<55} epochs {n:>3}  test {t}")
        else:
            test = (r["final_error_tables"].get("test") or [{}])[0]
            print(f"MACE   {r['label']:<55} epochs {len(r['train_loss_epoch_mean']):>3}  test {test}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/training_runs.json")
