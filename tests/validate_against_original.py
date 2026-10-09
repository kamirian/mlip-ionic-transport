"""Validate iondiff against the original MoGroup diffusion module.

Run in an environment where the original module is importable. Checks:
  1. the vectorized MSD equals the original per-step loop (random walk);
  2. DiffusivityAnalyzer reproduces the original D/MSD on a real trajectory when fed exactly the
     frames the original analysis used ("legacy stitching": first 100 frames of every run dropped);
  3. reports D with the corrected stitching, for comparison.

Usage: python validate_against_original.py <temperature_dir> <frame_interval_fs> <specie> <old_results_csv>
"""

import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from iondiff.diffusion import DiffusivityAnalyzer, one_ion_msd_fft  # noqa: E402
from iondiff.trajectory import build_chains, read_ase_trajectory  # noqa: E402


def original_one_ion_msd(r, dt_indices):
    """Verbatim copy of the S1/S2 loop in the original module."""
    def autocorrelation_fft(x):
        N = x.shape[0]
        F = np.fft.fft(x, n=2 * N)
        res = np.fft.ifft(F * F.conjugate())[:N].real
        return res / (N * np.ones(N) - np.arange(N))
    n_step, dim = r.shape
    r_square = np.append(np.square(r), np.zeros((1, dim)), axis=0)
    S1_component = np.zeros((dim, n_step))
    r_square_sum = 2 * np.sum(r_square, axis=0)
    for i in range(n_step):
        r_square_sum = r_square_sum - r_square[i - 1, :] - r_square[n_step - i, :]
        S1_component[:, i] = r_square_sum / (n_step - i)
    S2_component = np.array([autocorrelation_fft(r[:, i]) for i in range(dim)])
    return (S1_component.sum(0) - 2 * S2_component.sum(0))[dt_indices], (S1_component - 2 * S2_component)[:, dt_indices]


def check_msd():
    rng = np.random.default_rng(0)
    r = np.cumsum(rng.normal(size=(5001, 3)), axis=0)
    idx = np.arange(1, 5001, 5)
    a, ac = one_ion_msd_fft(r, idx)
    b, bc = original_one_ion_msd(r, idx)
    err = max(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)), np.max(np.abs(ac - bc) / np.maximum(np.abs(bc), 1e-12)))
    print(f"[1] vectorized vs original MSD: max relative difference = {err:.2e}")
    assert err < 1e-9


def main():
    tdir, interval, specie, old_csv = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
    check_msd()
    temperature = float(os.path.basename(tdir.rstrip("/")).split("_")[-1].rstrip("K"))
    paths = [os.path.join(tdir, "md_nvt_nhc.traj")] + sorted(glob.glob(os.path.join(tdir, "run*_md_nvt_nhc.traj")))
    segs = [read_ase_trajectory(p) for p in paths]
    spec = {"lower_bound": 0.5 * 3.38 ** 2, "upper_bound": 0.8, "minimum_msd_diff": 0.5 * 3.38 ** 2}

    # Legacy stitching: [100:] of every run, all concatenated, first-frame lattice.
    frac = np.concatenate([s.frac[100:] for s in segs])
    legacy = DiffusivityAnalyzer.from_frames(segs[0].symbols, frac, segs[0].lattices[0], specie,
                                             temperature, interval, spec_dict=spec)

    # Original module on the same frames.
    from pymatgen.core import Lattice, Structure
    from toyota.md.diffusion import DiffusivityAnalyzer as Orig
    lat = Lattice(segs[0].lattices[0])
    structs = [Structure(lat, segs[0].symbols, f) for f in frac]
    orig = Orig.from_structures(structs, specie, temperature, interval, 1, 1000, spec)
    rel = abs(legacy.diffusivity - orig.diffusivity) / abs(orig.diffusivity)
    print(f"[2] legacy-stitched D: iondiff {legacy.diffusivity:.10e}  original {orig.diffusivity:.10e}  rel.diff {rel:.2e}")
    assert rel < 1e-9

    old = None
    with open(old_csv) as fh:
        for row in csv.DictReader(fh):
            if abs(float(row["T"]) - temperature) < 1e-6:
                old = float(row["D"])
    print(f"    D in the old results CSV at {temperature:g} K: {old}")

    chains = build_chains(segs, 100)
    ch = max(chains, key=lambda c: c.n_frames)
    fixed = DiffusivityAnalyzer.from_frames(ch.symbols, ch.frac, ch.lattices, specie, temperature,
                                            interval, spec_dict=spec)
    print(f"[3] chains={len(chains)} frames legacy={frac.shape[0]} fixed={ch.n_frames}")
    print(f"    fixed-stitching D = {fixed.diffusivity:.6e}  (legacy {legacy.diffusivity:.6e}, ratio {fixed.diffusivity / legacy.diffusivity:.4f})")


if __name__ == "__main__":
    main()
