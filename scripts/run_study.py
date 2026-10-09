"""Analyze one study manifest and write <out>.json plus <out>_msd_curves.npz.

Usage: python scripts/run_study.py manifests/lyc_7fp_nvt.json results/lyc_7fp_nvt.json --workers 16
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from iondiff.study import run_study  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("manifest")
p.add_argument("out")
p.add_argument("--workers", type=int, default=1)
a = p.parse_args()
doc = run_study(a.manifest, a.out, workers=a.workers)
for s in doc["series"]:
    fit = s["arrhenius"].get("all (no framework melt)", {})
    n_ok = sum(1 for r in s["temperatures"] if r.get("fit_ok"))
    print(f"{s['label']:<60} T analyzed {len(s['temperatures']):>3}, fit ok {n_ok:>3}, "
          f"Ea(all, weighted, no melt) = {fit.get('Ea_eV', float('nan')):.4f} +/- {fit.get('Ea_std_eV', float('nan')):.4f} eV")
