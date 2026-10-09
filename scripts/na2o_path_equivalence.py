"""Check which Na2O DFT NEB paths are symmetry-equivalent (image-by-image StructureMatcher).

Input: CSV with one pymatgen structure string and total energy per NEB image (7 consecutive images per
path). Output: JSON with DFT barriers of each path and the pairwise equivalence matrix.

Usage: python scripts/na2o_path_equivalence.py <images.csv> <out.json> [n_paths]
"""

import csv
import json
import re
import sys

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure


def parse_structure(text):
    abc = [float(x) for x in re.search(r"abc\s*:\s*([\d.\s]+)\n", text).group(1).split()]
    ang = [float(x) for x in re.search(r"angles:\s*([\d.\s]+)\n", text).group(1).split()]
    species, frac = [], []
    for line in text.splitlines():
        m = re.match(r"\s*\d+\s+([A-Z][a-z]?)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$", line)
        if m:
            species.append(m.group(1))
            frac.append([float(m.group(i)) for i in (2, 3, 4)])
    return Structure(Lattice.from_parameters(*abc, *ang), species, frac)


def main(src, out, n_paths=5):
    csv.field_size_limit(10 ** 9)
    rows = list(csv.DictReader(open(src)))
    structures = [parse_structure(r["Structures"]) for r in rows[: 7 * n_paths]]
    energies = [float(r["Energies"]) for r in rows[: 7 * n_paths]]
    paths = {f"{7 * k}-{7 * k + 6}": list(range(7 * k, 7 * k + 7)) for k in range(n_paths)}
    sm = StructureMatcher(ltol=0.05, stol=0.05, angle_tol=1, primitive_cell=False, scale=False)
    doc = {"matcher": {"ltol": 0.05, "stol": 0.05, "angle_tol": 1, "primitive_cell": False, "scale": False},
           "paths": {}, "equivalent": {}}
    for name, idx in paths.items():
        e = [energies[i] - energies[idx[0]] for i in idx]
        doc["paths"][name] = {"images": idx, "relative_energies_eV": e, "barrier_eV": max(e)}
    names = list(paths)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            fwd = all(sm.fit(structures[p], structures[q]) for p, q in zip(paths[a], paths[b]))
            rev = all(sm.fit(structures[p], structures[q]) for p, q in zip(paths[a], paths[b][::-1]))
            doc["equivalent"][f"{a}|{b}"] = bool(fwd or rev)
    json.dump(doc, open(out, "w"), indent=1)
    for k, v in doc["equivalent"].items():
        print(k, v)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 5)
