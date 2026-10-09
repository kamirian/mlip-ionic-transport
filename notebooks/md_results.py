"""Helpers for the MD notebooks: load a study's results JSON, tabulate, and plot.

Every number shown comes from results/<study>.json (written by scripts/run_study.py) or the matching
results/<study>_msd_curves.npz.
"""

import json
import os

import numpy as np
import pandas as pd

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
PRIMARY = "all (no framework melt)"


def load(study):
    with open(os.path.join(RESULTS, f"{study}.json")) as fh:
        return json.load(fh)


def load_curves(study):
    npz = np.load(os.path.join(RESULTS, f"{study}_msd_curves.npz"))
    return {k: json.loads(str(npz[k])) for k in npz.files}


def series(doc, label):
    return next(s for s in doc["series"] if s["label"] == label)


def summary_table(doc, fit_key=PRIMARY):
    """One row per series: sampling, completeness and the chosen Arrhenius fit."""
    rows = []
    for s in doc["series"]:
        recs = s["temperatures"]
        f = s["arrhenius"][fit_key]
        used = f.get("temperatures_used_K", [])
        rows.append({
            "model": s["label"],
            "temperatures": len(recs),
            "total analyzed time (ns)": round(sum(r.get("analyzed_time_ps", 0) for r in recs) / 1000, 1),
            "MSD fit ok": sum(1 for r in recs if r.get("fit_ok")),
            "framework melted": sum(1 for r in recs if r.get("framework_melt_flag")),
            "T used in fit (K)": f"{min(used):g}-{max(used):g} ({len(used)})" if used else "-",
            "Ea (eV)": round(f["Ea_eV"], 3) if f.get("status") == "ok" else None,
            "Ea std (eV)": round(f["Ea_std_eV"], 3) if f.get("status") == "ok" else None,
            "R2": round(f["r_squared"], 3) if f.get("status") == "ok" else None,
            "sigma(300 K) extrapolated (mS/cm)": round(f["conductivity_300K_mS_cm"], 2) if f.get("status") == "ok" else None,
            "sigma(300 K) range (mS/cm)": (f"{f['conductivity_300K_range_mS_cm'][0]:.2f}-{f['conductivity_300K_range_mS_cm'][1]:.2f}"
                                           if f.get("status") == "ok" else None),
        })
    return pd.DataFrame(rows).set_index("model")


def fit_comparison(doc, keys):
    """Ea of every series under several fit choices (columns)."""
    return pd.DataFrame({k: {s["label"]: (round(s["arrhenius"][k]["Ea_eV"], 3)
                                          if s["arrhenius"][k].get("status") == "ok" else None)
                             for s in doc["series"]} for k in keys})


def per_temperature(s):
    rows = []
    for r in s["temperatures"]:
        rows.append({"T (K)": r["temperature_K"], "status": r["status"],
                     "analyzed time (ps)": r.get("analyzed_time_ps"),
                     "jobs (segments)": len(r.get("segments", [])),
                     "D (cm2/s)": r.get("D_cm2_s"), "D std (cm2/s)": r.get("D_std_cm2_s"),
                     "N_jump": r.get("n_jump"), "max framework disp. (A)": r.get("max_framework_displacement_A"),
                     "melted": r.get("framework_melt_flag")})
    return pd.DataFrame(rows).set_index("T (K)")


def arrhenius_plot(ax, s, color, label=None, fit_key=PRIMARY, show_melted=True):
    """log10 D vs 1000/T. Filled: used in the fit; open: melted framework; line: fit."""
    f = s["arrhenius"][fit_key]
    used = set(f.get("temperatures_used_K", []))
    ok = [r for r in s["temperatures"] if r.get("fit_ok") and r.get("D_cm2_s", 0) > 0]
    for subset, filled in (([r for r in ok if r["temperature_K"] in used], True),
                           ([r for r in ok if r["temperature_K"] not in used and r.get("framework_melt_flag")], False)):
        if not subset or (not filled and not show_melted):
            continue
        x = np.array([1000 / r["temperature_K"] for r in subset])
        d = np.array([r["D_cm2_s"] for r in subset])
        e = np.array([r["D_std_cm2_s"] for r in subset])
        ax.errorbar(x, d, yerr=e, fmt="o", ms=4, color=color, mfc=color if filled else "white",
                    elinewidth=0.8, capsize=0, alpha=0.9 if filled else 0.6,
                    label=(f"{label}: Ea = {f['Ea_eV']:.3f} ± {f['Ea_std_eV']:.3f} eV" if filled and f.get("status") == "ok"
                           else None))
    if f.get("status") == "ok":
        xs = np.linspace(1000 / max(used), 1000 / min(used), 50)
        ax.plot(xs, 10 ** (f["slope"] * xs + f["intercept"]), "-", color=color, lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("1000 / T (1/K)")
    ax.set_ylabel("D (cm$^2$/s)")


def msd_plot(ax, curves, label, temperatures, cmap="viridis"):
    import matplotlib.pyplot as plt
    cm = plt.get_cmap(cmap)
    for i, T in enumerate(temperatures):
        c = curves.get(f"{label}|{T:g}")
        if not c:
            continue
        dt = np.array(c["dt_fs"]) / 1000
        msd = np.array(c["msd_A2"])
        lo, hi = c["fit_window_index"]
        col = cm(i / max(len(temperatures) - 1, 1))
        ax.plot(dt, msd, color=col, lw=1, label=f"{T:g} K")
        if hi > lo:
            ax.plot(dt[lo:hi + 1], msd[lo:hi + 1], color=col, lw=3, alpha=0.35)
    ax.set_xlabel("time lag (ps)")
    ax.set_ylabel("MSD (Å$^2$)")


# ---------------------------------------------------------------- comparison with AIMD (DFT)
def _iondiff():
    import sys
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    if root not in sys.path:
        sys.path.insert(0, root)
    from iondiff import ArrheniusAnalyzer
    return ArrheniusAnalyzer


def usable(s):
    """Temperatures that enter the primary fit: MSD fit ok, framework not melted."""
    return [r for r in s["temperatures"] if r.get("fit_ok") and r.get("D_cm2_s", 0) > 0
            and not r.get("framework_melt_flag")]


def fit_window(s, tmin, tmax):
    """Weighted Arrhenius fit of one series restricted to [tmin, tmax] (same method as the primary fit)."""
    pts = [r for r in usable(s) if tmin <= r["temperature_K"] <= tmax]
    if len(pts) < 3:
        return None
    aa = _iondiff()([r["temperature_K"] for r in pts], [r["D_cm2_s"] for r in pts],
                    [r["D_std_cm2_s"] for r in pts])
    return {"Ea_eV": aa.Ea, "Ea_std_eV": aa.Ea_error, "n": len(pts),
            "T_range": (min(r["temperature_K"] for r in pts), max(r["temperature_K"] for r in pts))}


def predicted_D(s, T, fit_key=PRIMARY):
    f = s["arrhenius"][fit_key]
    return 10 ** (f["slope"] * 1000 / T + f["intercept"])


def compare_to_aimd(series_list, aimd_series):
    """Ea over the AIMD temperature window and FP/AIMD diffusivity ratios at the AIMD temperatures
    (only temperatures inside each FP's own fitted range; no extrapolation)."""
    ref = usable(aimd_series)
    tmin, tmax = min(r["temperature_K"] for r in ref), max(r["temperature_K"] for r in ref)
    fa = aimd_series["arrhenius"][PRIMARY]
    rows = [{"model": aimd_series["label"], "Ea in AIMD window (eV)": round(fa["Ea_eV"], 3),
             "Ea std (eV)": round(fa["Ea_std_eV"], 3), "T used (K)": f"{tmin:g}-{tmax:g} ({len(ref)})"}]
    ratios = {}
    for s in series_list:
        fw = fit_window(s, tmin, tmax)
        rows.append({"model": s["label"],
                     "Ea in AIMD window (eV)": round(fw["Ea_eV"], 3) if fw else None,
                     "Ea std (eV)": round(fw["Ea_std_eV"], 3) if fw else None,
                     "T used (K)": f"{fw['T_range'][0]:g}-{fw['T_range'][1]:g} ({fw['n']})" if fw else "<3 points"})
        used = s["arrhenius"][PRIMARY].get("temperatures_used_K", [])
        lo, hi = (min(used), max(used)) if used else (np.inf, -np.inf)
        ratios[s["label"]] = {r["temperature_K"]: (predicted_D(s, r["temperature_K"]) / r["D_cm2_s"]
                                                   if lo <= r["temperature_K"] <= hi else np.nan) for r in ref}
    ratio = pd.DataFrame(ratios).T
    ratio.columns = [f"D_FP/D_AIMD at {c:g} K" for c in ratio.columns]
    return pd.DataFrame(rows).set_index("model"), ratio


def aimd_overlay(ax, aimd_series, color="k", label="AIMD (PBE)"):
    ref = usable(aimd_series)
    x = np.array([1000 / r["temperature_K"] for r in ref])
    d = np.array([r["D_cm2_s"] for r in ref]); e = np.array([r["D_std_cm2_s"] for r in ref])
    f = aimd_series["arrhenius"][PRIMARY]
    ax.errorbar(x, d, yerr=e, fmt="s", ms=6, color=color, mfc="white", mew=1.5, capsize=2, zorder=5,
                label=f"{label}: Ea = {f['Ea_eV']:.3f} ± {f['Ea_std_eV']:.3f} eV")
    xs = np.linspace(x.min(), x.max(), 20)
    ax.plot(xs, 10 ** (f["slope"] * xs + f["intercept"]), "--", color=color, lw=1.5, zorder=5)
    ax.set_yscale("log")
