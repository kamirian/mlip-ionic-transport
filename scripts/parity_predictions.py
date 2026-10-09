"""Predictions of trained models on their own train/validation/test frames, for parity plots.

Inference only: each saved model is evaluated on exactly the frames it was trained, validated and
tested on, and reference vs predicted energies (per atom) and force components are written to an
.npz file. The mean absolute errors recomputed from these predictions should reproduce the test
errors printed in the original training logs (a check that the right frames and model were used).

Usage:
  python parity_predictions.py mace   <model> <train.xyz> <valid.xyz> <test.xyz> <out.npz>
  python parity_predictions.py chgnet <checkpoint> <frames.csv> <n_rows> <training_log> <out.npz>

CHGNet mode rebuilds the data exactly as the original training script did: the first ``n_rows`` rows
of the frame table, structures rebuilt from lattice lengths/angles, and the train/validation/test
row indices printed in the training log.
"""

import ast
import json
import re
import sys

import numpy as np

MAX_FORCE_COMPONENTS = 300_000  # per split; force components are subsampled above this (fixed seed)


def _pack(out, split, e_ref, e_pred, f_ref, f_pred, rng):
    f_ref, f_pred = np.concatenate(f_ref).ravel(), np.concatenate(f_pred).ravel()
    keep = (np.sort(rng.choice(len(f_ref), MAX_FORCE_COMPONENTS, replace=False))
            if len(f_ref) > MAX_FORCE_COMPONENTS else slice(None))
    out[f"{split}_energy_ref"] = np.array(e_ref)
    out[f"{split}_energy_pred"] = np.array(e_pred)
    out[f"{split}_force_ref"] = f_ref[keep]
    out[f"{split}_force_pred"] = f_pred[keep]
    print(f"{split}: {len(e_ref)} frames, energy MAE {1000 * np.mean(np.abs(np.array(e_pred) - e_ref)):.3f} meV/atom, "
          f"force MAE {1000 * np.mean(np.abs(f_pred - f_ref)):.2f} meV/A (all components)", flush=True)


def run_mace(model, train, valid, test, out_path):
    from ase.io import read
    from mace.calculators import MACECalculator

    calc = MACECalculator(model_paths=[model], device="cpu", default_dtype="float64")
    rng, out = np.random.default_rng(0), {}
    for split, path in (("train", train), ("valid", valid), ("test", test)):
        e_ref, e_pred, f_ref, f_pred = [], [], [], []
        for atoms in read(path, index=":"):
            n = len(atoms)
            ref_e = atoms.info["energy"] if "energy" in atoms.info else atoms.get_potential_energy()
            ref_f = atoms.arrays["forces"] if "forces" in atoms.arrays else atoms.get_forces()
            a = atoms.copy()
            a.calc = calc
            e_ref.append(ref_e / n)
            e_pred.append(a.get_potential_energy() / n)
            f_ref.append(np.array(ref_f))
            f_pred.append(a.get_forces())
        _pack(out, split, e_ref, e_pred, f_ref, f_pred, rng)
    np.savez_compressed(out_path, **out)


def _indices_from_log(log, name):
    text = open(log).read()
    m = re.search(rf"^{name}_data[ _]?is[ _]?\s*(\[[^\]]*\])", text, re.M)
    return ast.literal_eval(m.group(1)) if m else None


def run_chgnet(checkpoint, frames_csv, n_rows, log, out_path):
    import pandas as pd
    from chgnet.model import CHGNet
    from pymatgen.core import Lattice, Structure

    df = pd.read_csv(frames_csv, nrows=int(n_rows))

    def rebuild(text):  # same construction as the original training script
        lines = text.strip().split("\n")
        abc = list(map(float, lines[2].split(":")[1].split()))
        ang = list(map(float, lines[3].split(":")[1].split()))
        species, coords = [], []
        for line in lines[8:]:
            p = line.split()
            if len(p) >= 5 and not line.startswith("#"):
                species.append(p[1])
                coords.append([float(p[2]), float(p[3]), float(p[4])])
        return Structure(Lattice.from_parameters(*abc, *ang), species, coords)

    model = CHGNet.from_file(checkpoint)
    rng, out = np.random.default_rng(0), {}
    for split in ("train", "test"):
        idx = _indices_from_log(log, split)
        if idx is None:
            raise ValueError(f"no {split} indices in {log}")
        e_ref, e_pred, f_ref, f_pred = [], [], [], []
        rows = df.iloc[idx]
        structures = [rebuild(t) for t in rows["Structure_Formatted"]]
        preds = model.predict_structure(structures, task="ef", batch_size=16)
        for (_, row), pr in zip(rows.iterrows(), preds):
            e_ref.append(float(row["Energy_per_Atom"]))
            e_pred.append(float(pr["e"]))
            f_ref.append(np.array(json.loads(row["Force"])))
            f_pred.append(np.array(pr["f"]))
        _pack(out, split, e_ref, e_pred, f_ref, f_pred, rng)
    np.savez_compressed(out_path, **out)


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    {"mace": run_mace, "chgnet": run_chgnet}[mode](*args)
