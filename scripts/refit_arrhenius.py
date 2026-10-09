"""Recompute the Arrhenius fits in a results file from its per-temperature records.

Usage: python scripts/refit_arrhenius.py results/<study>.json manifests/<study>.json
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from iondiff.study import refit  # noqa: E402

doc = refit(sys.argv[1], sys.argv[2])
for s in doc["series"]:
    f = s["arrhenius"]["all (no framework melt)"]
    print(f"{s['label']:<60} Ea = {f.get('Ea_eV', float('nan')):.4f} +/- {f.get('Ea_std_eV', float('nan')):.4f} eV "
          f"({f['n_points']} T)")
