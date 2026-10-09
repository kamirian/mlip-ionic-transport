"""Run the diffusivity / Arrhenius analysis for a study described by a manifest.

A manifest (JSON) describes one study, e.g. "LYC, 7 foundation potentials, NVT":

{
  "study": "lyc_7fp_nvt",
  "description": "...",
  "specie": "Li", "charge": 1, "site_distance": 3.38,
  "fit": {"lower_bound_in_a_square": 0.5, "upper_bound": 0.8,
          "minimum_msd_diff_in_a_square": 0.5, "time_intervals_number": 1000},
  "arrhenius_windows": {"all": null, "500-1000K": [500, 1000]},
  "series": [
    {"label": "MACE-MPA-0 (medium)", "model_file": "...", "root": "/path/to/runs",
     "temperature_dir_regex": "^ase_test_(\\d+)K$",
     "segments": ["md_nvt_nhc.traj", "run{n}_md_nvt_nhc.traj"],
     "frame_interval_fs": 200, "equilibration_frames": 100,
     "log_for_segment": {"md_nvt_nhc.traj": "md_nvt_nhc.log"}}
  ]
}

``segments`` are tried in order inside every temperature directory; a pattern containing
``{n}`` expands to n = 2, 3, ... until a file is missing.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import __version__
from .diffusion import ArrheniusAnalyzer, DiffusivityAnalyzer, ErrorAnalysis, get_conversion_factor
from .trajectory import build_chains, read_ase_trajectory, read_vasp_run

SCHEMA_VERSION = "1.0"
# Single-atom check of the original module (kept as a diagnostic only): any framework atom moved > 4 A.
FRAMEWORK_SINGLE_ATOM_THRESHOLD_A = 4.0


MELT_CRITERION = ("framework melted if the time-averaged MSD of framework (non-mobile) atoms at the end of "
                  "the fit window exceeds site_distance^2; the original single-atom check (any framework "
                  "atom displaced > 4 A) is recorded as framework_single_atom_moved_4A")


def is_framework_melted(rec: dict, site_distance: float) -> bool:
    """Collective framework diffusion: the time-averaged framework MSD at the end of the fit window
    exceeds a^2 (framework atoms have, on average, moved farther than one hop distance)."""
    v = rec.get("framework_msd_at_fit_end_A2")
    return v is not None and v > site_distance ** 2


def _vasp_run_dirs(tdir: str, s: dict) -> list:
    """RUN_<n> directories in numeric order, from ``min_run_index`` on (RUN_0 is the heating run)."""
    rx = re.compile(s.get("run_dir_regex", r"^RUN_(\d+)$"))
    runs = []
    for name in os.listdir(tdir):
        m = rx.match(name)
        if m and int(m.group(1)) >= s.get("min_run_index", 1) and os.path.isdir(os.path.join(tdir, name)):
            runs.append((int(m.group(1)), os.path.join(tdir, name)))
    return [p for _, p in sorted(runs)]


def _expand_segments(tdir: str, patterns: list) -> list:
    found = []
    for pat in patterns:
        if "{n}" in pat:
            n = 2
            while os.path.exists(os.path.join(tdir, pat.format(n=n))):
                found.append(os.path.join(tdir, pat.format(n=n)))
                n += 1
        elif os.path.exists(os.path.join(tdir, pat)):
            found.append(os.path.join(tdir, pat))
    return found


def _log_interval_fs(traj_path: str):
    """Frame interval from the ASE MD log written next to the trajectory (Time[ps] column)."""
    log = re.sub(r"\.traj$", ".log", traj_path)
    if not os.path.exists(log):
        return None
    times = []
    with open(log) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            try:
                times.append(float(parts[0]))
            except ValueError:
                continue
            if len(times) == 3:
                break
    if len(times) < 2:
        return None
    return round((times[1] - times[0]) * 1000.0, 6)


def analyze_temperature(job: dict) -> dict:
    """Analyze one temperature directory. ``job`` holds everything needed (picklable)."""
    s, cfg, temperature, tdir = job["series"], job["cfg"], job["temperature"], job["tdir"]
    # Per-temperature override, e.g. a few runs made with a different MD timestep.
    interval_fs = s.get("frame_interval_overrides_fs", {}).get(f"{temperature:g}", s["frame_interval_fs"])
    vasp = s.get("reader") == "vasp_runs"
    paths = _vasp_run_dirs(tdir, s) if vasp else _expand_segments(tdir, s["segments"])
    rec = {"temperature_K": temperature, "directory": tdir, "segments": [], "status": "ok",
           "frame_interval_fs": interval_fs}
    if not paths:
        rec["status"] = "no_trajectory"
        return rec
    segments = []
    for p in paths:
        try:
            seg = read_vasp_run(p, stride=s.get("stride", 1)) if vasp else read_ase_trajectory(p)
        except Exception as exc:  # unreadable/truncated file
            rec["segments"].append({"path": p, "error": repr(exc)})
            continue
        interval = None if vasp else _log_interval_fs(p)
        if interval is not None and abs(interval - interval_fs) > 1e-6:
            raise ValueError(f"{p}: log interval {interval} fs != manifest {interval_fs} fs")
        segments.append(seg)
    chains = build_chains(segments, s["equilibration_frames"])
    for chain_id, ch in enumerate(chains):
        for k, seg in enumerate(ch.segments):
            rec["segments"].append({
                "path": seg.path, "chain": chain_id, "n_frames": seg.n_frames,
                "frames_used": ch.frames_used[k],
                "simulated_time_ps": (seg.n_frames - 1) * interval_fs / 1000.0,
                "continuation_of_previous": k > 0})
    chains = [c for c in chains if c.n_frames > 1]
    if not chains:
        rec["status"] = "too_short"
        return rec
    rec["n_independent_chains"] = len(chains)
    # Longest chain is used for D; any additional independent chains are listed but not merged.
    ch = max(chains, key=lambda c: c.n_frames)
    a = cfg["site_distance"]
    spec = {"lower_bound": cfg["fit"]["lower_bound_in_a_square"] * a * a,
            "upper_bound": cfg["fit"]["upper_bound"],
            "minimum_msd_diff": cfg["fit"]["minimum_msd_diff_in_a_square"] * a * a}
    da = DiffusivityAnalyzer.from_frames(
        ch.symbols, ch.frac, ch.lattices, cfg["specie"], temperature, interval_fs,
        time_intervals_number=cfg["fit"]["time_intervals_number"], spec_dict=spec)
    volumes = np.abs(np.linalg.det(ch.lattices))
    n_specie = sum(1 for x in ch.symbols if x == cfg["specie"])
    rec.update({
        "analyzed_time_ps": (ch.n_frames - 1) * interval_fs / 1000.0,
        "n_frames_analyzed": int(ch.n_frames),
        "n_atoms": len(ch.symbols),
        "n_diffusing_ions": n_specie,
        "mean_volume_A3": float(volumes.mean()),
        "max_framework_displacement_A": da.max_framework_displacement,
        "framework_single_atom_moved_4A": da.max_framework_displacement > FRAMEWORK_SINGLE_ATOM_THRESHOLD_A,
        "framework_fraction_moved_4A": da.framework_fraction_moved_4A,
        "framework_msd_at_fit_end_A2": da.framework_msd_at_fit_end,
        "mobile_msd_at_fit_end_A2": float(da.msd[da.upper_bound_index]) if da.upper_bound_index >= 0 else None,
        "max_msd_A2": float(np.max(da.msd)),
        "fit_ok": bool(da.fit_ok),
    })
    rec["framework_melt_flag"] = is_framework_melted(rec, a)
    curve = {"dt_fs": da.dt.tolist(), "msd_A2": da.msd.tolist(),
             "msd_component_A2": da.msd_component.tolist(),
             "fit_window_index": [int(da.lower_bound_index), int(da.upper_bound_index)]}
    if da.fit_ok:
        ea = ErrorAnalysis(da, site_distance=a)
        factor = get_conversion_factor(n_specie, float(volumes.mean()), cfg["charge"], temperature)
        rec.update({
            "fit_window_ps": [float(da.dt[da.lower_bound_index]) / 1000.0,
                              float(da.dt[da.upper_bound_index]) / 1000.0],
            "D_cm2_s": float(da.diffusivity),
            "D_std_cm2_s": ea.diffusivity_std,
            "D_rsd": float(ea.rsd),
            "n_jump": float(ea.n_jump),
            "D_components_cm2_s": da.diffusivity_components.tolist(),
            "D_components_std_cm2_s": ea.diffusivity_component_std.tolist(),
            "D_components_rsd": ea.rsd_component.tolist(),
            "conductivity_mS_cm": float(factor * da.diffusivity),
            "conductivity_std_mS_cm": float(factor * ea.diffusivity_std),
            "nernst_einstein_factor": float(factor),
        })
    else:
        rec["status"] = "msd_fit_requirements_not_met"
    rec["_curve"] = curve
    return rec


def _arrhenius(records, window, cfg, weighted=True, exclude_melt=False):
    pts = [r for r in records if r.get("fit_ok") and r.get("D_cm2_s", 0) > 0]
    if exclude_melt:
        pts = [r for r in pts if not r.get("framework_melt_flag")]
    if window is not None:
        pts = [r for r in pts if window[0] <= r["temperature_K"] <= window[1]]
    out = {"temperature_window_K": window, "weighted": weighted, "exclude_framework_melt": exclude_melt,
           "temperatures_used_K": [r["temperature_K"] for r in pts], "n_points": len(pts)}
    if len(pts) < 3:
        out["status"] = "fewer_than_3_points"
        return out
    T = [r["temperature_K"] for r in pts]
    D = [r["D_cm2_s"] for r in pts]
    err = [r["D_std_cm2_s"] for r in pts] if weighted else None
    aa = ArrheniusAnalyzer(T, D, err)
    lowest = min(pts, key=lambda r: r["temperature_K"])
    sig300, (lo, hi) = aa.predict_conductivity(
        300.0, lowest["n_diffusing_ions"], lowest["mean_volume_A3"], cfg["charge"])
    out.update({"status": "ok", "Ea_eV": aa.Ea, "Ea_std_eV": aa.Ea_error, "r_squared": aa.r_squared,
                "slope": aa.slope, "intercept": aa.intercept,
                "D_300K_cm2_s": aa.predict_diffusivity(300.0)[0],
                "conductivity_300K_mS_cm": sig300, "conductivity_300K_range_mS_cm": [lo, hi],
                "volume_for_extrapolation_A3": lowest["mean_volume_A3"]})
    return out


def fit_arrhenius(records, cfg) -> dict:
    """All Arrhenius fits of one series. For every temperature window: weighted (He et al. errors),
    unweighted, and weighted after removing temperatures where the framework melted
    (see is_framework_melted); the melt-excluded weighted fit is the primary result."""
    fits = {}
    for wname, window in cfg.get("arrhenius_windows", {"all": None}).items():
        fits[wname + " (no framework melt)"] = _arrhenius(records, window, cfg, weighted=True, exclude_melt=True)
        fits[wname] = _arrhenius(records, window, cfg, weighted=True)
        fits[wname + " (unweighted)"] = _arrhenius(records, window, cfg, weighted=False)
    return fits


def refit(results_path: str, manifest_path: str) -> dict:
    """Recompute the Arrhenius block of an existing results file (no trajectories needed)."""
    with open(manifest_path) as fh:
        cfg = json.load(fh)
    with open(results_path) as fh:
        doc = json.load(fh)
    for s in doc["series"]:
        for r in s["temperatures"]:
            if "framework_msd_at_fit_end_A2" in r:
                r["framework_single_atom_moved_4A"] = r.get("max_framework_displacement_A", 0) > FRAMEWORK_SINGLE_ATOM_THRESHOLD_A
                r["framework_melt_flag"] = is_framework_melted(r, cfg["site_distance"])
        s["arrhenius"] = fit_arrhenius(s["temperatures"], cfg)
    doc["method"]["framework_melt_criterion"] = MELT_CRITERION
    doc["method"].pop("framework_melt_threshold_A", None)
    doc["method"]["arrhenius_windows"] = cfg.get("arrhenius_windows", {"all": None})
    doc["method"]["primary_fit"] = "'<window> (no framework melt)': weighted, melted temperatures excluded"
    with open(results_path, "w") as fh:
        json.dump(doc, fh, indent=1)
    return doc


def run_study(manifest_path: str, out_path: str, workers: int = 1) -> dict:
    with open(manifest_path) as fh:
        cfg = json.load(fh)
    jobs = []
    for s in cfg["series"]:
        rx = re.compile(s["temperature_dir_regex"])
        for name in sorted(os.listdir(s["root"])):
            m = rx.match(name)
            if m and os.path.isdir(os.path.join(s["root"], name)):
                jobs.append({"series": s, "cfg": cfg, "temperature": float(m.group(1)),
                             "tdir": os.path.join(s["root"], name)})
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(analyze_temperature, jobs, chunksize=1))
    else:
        results = [analyze_temperature(j) for j in jobs]

    curves = {}
    series_out = []
    for s in cfg["series"]:
        recs = sorted([r for j, r in zip(jobs, results) if j["series"] is s],
                      key=lambda r: r["temperature_K"])
        for r in recs:
            curves[f"{s['label']}|{r['temperature_K']:g}"] = r.pop("_curve", None)
        series_out.append({k: v for k, v in s.items()} | {"temperatures": recs,
                                                           "arrhenius": fit_arrhenius(recs, cfg)})

    import ase
    import pymatgen.core
    import scipy

    doc = {
        "schema_version": SCHEMA_VERSION,
        "study": cfg["study"],
        "description": cfg.get("description", ""),
        "units": {"D": "cm^2/s", "conductivity": "mS/cm", "time": "ps", "msd": "A^2",
                  "Ea": "eV", "volume": "A^3", "temperature": "K"},
        "method": {
            "reference": "X. He, Y. Zhu, A. Epstein, Y. Mo, npj Comput. Mater. 4, 18 (2018)",
            "msd": "FFT time-averaged MSD, framework centre-of-mass drift removed",
            "D_error": "RSD = 3.43/sqrt(N_jump) + 0.04, D_std = RSD * D",
            "arrhenius": "least squares on log10(D) vs 1000/T, weighted by sigma_log10D = log10(e)*D_std/D",
            "stitching": "continuation runs concatenated after dropping their duplicated first frame; "
                         "equilibration frames removed only from the start of each chain",
            "specie": cfg["specie"], "charge": cfg["charge"], "site_distance_A": cfg["site_distance"],
            "fit": cfg["fit"],
            "framework_melt_criterion": MELT_CRITERION,
            "arrhenius_windows": cfg.get("arrhenius_windows", {"all": None}),
            "primary_fit": "'<window> (no framework melt)': weighted, melted temperatures excluded",
        },
        "generation_metadata": {
            "iondiff_version": __version__,
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "ase": ase.__version__, "pymatgen": pymatgen.core.__version__,
            "manifest": os.path.basename(manifest_path),
        },
        "series": series_out,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(doc, fh, indent=1)
    np.savez_compressed(re.sub(r"\.json$", "", out_path) + "_msd_curves.npz",
                        **{k: np.array(json.dumps(v)) for k, v in curves.items()})
    return doc
