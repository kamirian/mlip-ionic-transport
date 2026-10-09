"""Shared figure style for the notebooks."""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2",
           "#17becf", "#7f7f7f", "#bcbd22", "#393b79", "#ad494a", "#637939", "#7b4173"]

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.prop_cycle": mpl.cycler(color=PALETTE),
})

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


def save(fig, name):
    """Save a figure as PNG under figures/ (subfolders allowed in ``name``)."""
    path = os.path.join(FIG_DIR, name + ".png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    return os.path.relpath(path, os.path.join(FIG_DIR, ".."))


__all__ = ["PALETTE", "plt", "save"]
