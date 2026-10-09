"""Load MD trajectory segments and stitch continuation runs into one trajectory.

A temperature is often simulated as several consecutive jobs (run1, run2, ...). A later job is a
*continuation* when its first stored frame is the previous job's last frame (same positions and
cell, typically with re-drawn velocities). Continuations are concatenated after dropping the
duplicated first frame; only the very first segment of a chain has equilibration frames removed.
A segment that does not start where the previous one ended is *independent* and starts a new
chain. Chains are analyzed separately; they are never concatenated, because that would insert an
artificial jump into the displacement history.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Segment:
    path: str
    symbols: list
    frac: np.ndarray        # (n_frames, n_atoms, 3), wrapped fractional coordinates
    lattices: np.ndarray    # (n_frames, 3, 3)

    @property
    def n_frames(self) -> int:
        return self.frac.shape[0]


@dataclass
class Chain:
    segments: list = field(default_factory=list)
    frames_used: list = field(default_factory=list)  # frames taken from each segment
    frac: np.ndarray | None = None
    lattices: np.ndarray | None = None
    symbols: list | None = None

    @property
    def n_frames(self) -> int:
        return 0 if self.frac is None else self.frac.shape[0]


def read_ase_trajectory(path: str, index=":") -> Segment:
    """Read an ASE .traj (or any ase.io-readable file) into arrays."""
    from ase.io import read

    images = read(path, index=index)
    if not isinstance(images, list):
        images = [images]
    if not images:
        raise ValueError(f"{path} contains no frames")
    symbols = images[0].get_chemical_symbols()
    frac = np.array([a.get_scaled_positions(wrap=True) for a in images])
    lattices = np.array([a.cell.array for a in images])
    return Segment(path=path, symbols=symbols, frac=frac, lattices=lattices)


def _read_poscar_like(lines, start=0):
    """Parse a POSCAR/XDATCAR header at ``start``: returns (lattice, symbols, index of next line)."""
    scale = float(lines[start + 1].split()[0])
    lattice = np.array([[float(x) for x in lines[start + i].split()[:3]] for i in (2, 3, 4)]) * scale
    elements = lines[start + 5].split()
    counts = [int(x) for x in lines[start + 6].split()]
    symbols = [e for e, c in zip(elements, counts) for _ in range(c)]
    return lattice, symbols, start + 7


def read_vasp_run(run_dir: str, stride: int = 1) -> Segment:
    """One VASP MD run (fixed cell): POSCAR as frame 0 followed by every XDATCAR configuration.

    Frame 0 is the run's starting structure, so a run started from the previous run's CONTCAR is
    recognised as a continuation by :func:`is_continuation`. ``stride`` keeps every stride-th frame
    counted from frame 0 (so the last frame is kept when the number of steps is a multiple of stride).
    """
    import os

    with open(os.path.join(run_dir, "POSCAR")) as fh:
        p = fh.read().splitlines()
    lattice, symbols, i = _read_poscar_like(p)
    if p[i].strip()[0] in "sS":  # selective dynamics
        i += 1
    n = len(symbols)
    first = np.array([[float(x) for x in l.split()[:3]] for l in p[i + 1:i + 1 + n]])
    if p[i].strip()[0] in "cCkK":  # Cartesian POSCAR
        first = first @ np.linalg.inv(lattice)
    with open(os.path.join(run_dir, "XDATCAR")) as fh:
        x = fh.read().splitlines()
    xlat, xsym, j = _read_poscar_like(x)
    if xsym != symbols:
        raise ValueError(f"{run_dir}: POSCAR and XDATCAR species differ")
    frames = [first]
    while j < len(x):
        if x[j].startswith("Direct configuration"):
            block = x[j + 1:j + 1 + n]
            if len(block) < n:  # truncated last frame of a killed job
                break
            frames.append(np.array([[float(v) for v in l.split()[:3]] for l in block]))
            j += n + 1
        else:  # variable-cell XDATCAR repeats the header; not used for NVT runs
            raise ValueError(f"{run_dir}: variable-cell XDATCAR is not supported")
    frac = np.array(frames[::stride]) % 1.0
    return Segment(path=run_dir, symbols=symbols, frac=frac,
                   lattices=np.broadcast_to(xlat, (frac.shape[0], 3, 3)).copy())


def is_continuation(prev: Segment, nxt: Segment, pos_tol: float = 1e-4, cell_tol: float = 1e-4) -> bool:
    """True if ``nxt`` starts at the last frame of ``prev`` (minimum-image fractional match)."""
    if prev.symbols != nxt.symbols:
        return False
    dfrac = nxt.frac[0] - prev.frac[-1]
    dfrac -= np.round(dfrac)
    dcart = dfrac @ nxt.lattices[0]
    return (np.max(np.abs(dcart)) < pos_tol
            and np.max(np.abs(nxt.lattices[0] - prev.lattices[-1])) < cell_tol)


def build_chains(segments: list, equilibration_frames: int) -> list:
    """Group ordered segments into chains of continuations.

    ``equilibration_frames`` stored frames are removed from the first segment of every chain.
    """
    chains, parts = [], []
    for seg in segments:
        if chains and is_continuation(chains[-1].segments[-1], seg):
            chain = chains[-1]
            start = 1  # drop the duplicated first frame
        else:
            chain = Chain(symbols=seg.symbols)
            chains.append(chain)
            parts.append(([], []))
            start = equilibration_frames
        chain.segments.append(seg)
        chain.frames_used.append(max(seg.n_frames - start, 0))
        if seg.n_frames > start:
            parts[-1][0].append(seg.frac[start:])
            parts[-1][1].append(seg.lattices[start:])
    for chain, (fr, la) in zip(chains, parts):  # concatenate once per chain
        if fr:
            chain.frac, chain.lattices = np.concatenate(fr), np.concatenate(la)
    return chains
