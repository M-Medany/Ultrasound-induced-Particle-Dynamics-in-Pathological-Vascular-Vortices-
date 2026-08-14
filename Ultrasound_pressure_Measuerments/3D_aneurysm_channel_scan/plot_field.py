"""
Plot the 3D-aneurysm-channel hydrophone scan (2 MHz, 30 Vpp transducer drive,
hydrophone SN 4746, 50 Ohm termination). Single-plane XY scan at Z = 13 mm.

Reproducibility:
- Reads data_table.csv (X_mm, Y_mm, Z_mm, Vpp_mV, Pressure_pp_kPa, RawFile).
- Vpp was computed as full-trace peak-to-peak voltage directly from the raw
  waveforms in raw_csv/ (Tektronix TBS2072 exports).
- Pressure uses S = 654.0046 mV/MPa, the hydrophone's calibrated sensitivity
  AT EXACTLY 2 MHz (certificate 20251211-03) -- no interpolation needed,
  unlike the 1.7 MHz baseline scan.


Outputs (written to ./figures/):
- field_map_2D.png          : interpolated pressure map of the XY scan at Z=13 mm
- field_profiles.png        : X-scan (Y=9, Y=11) and Y-scan (X=9,10,11) cuts
"""

import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.tri import Triangulation, LinearTriInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(HERE, "data_table.csv")
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_data(path=DATA_CSV):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({
                "csv": int(row["CSV"]), "x": float(row["X_mm"]), "y": float(row["Y_mm"]),
                "z": float(row["Z_mm"]), "vpp": float(row["Vpp_mV"]), "p": float(row["Pressure_pp_kPa"]),
            })
    return rows


def unique_points(rows):
    """One row per (x,y,z). The dataset already has a single reading per position
    (the imprecise first readings at the 4 repeated positions were discarded),
    so this just normalizes into the shape the plotters expect. If any position
    ever appears twice again, keep the later (higher-CSV) reading."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["x"], r["y"], r["z"])].append(r)
    out = []
    for (x, y, z), grp in groups.items():
        chosen = sorted(grp, key=lambda g: g["csv"])[-1]
        out.append({"x": x, "y": y, "z": z,
                     "p": chosen["p"], "vpp": chosen["vpp"],
                     "n": len(grp), "csv": chosen["csv"]})
    return out


def plot_2d_map(rows):
    uniq = unique_points(rows)
    x = np.array([r["x"] for r in uniq])
    y = np.array([r["y"] for r in uniq])
    p = np.array([r["p"] for r in uniq])

    fig, ax = plt.subplots(figsize=(9, 6.5))
    tri = Triangulation(x, y)
    interp = LinearTriInterpolator(tri, p)
    xi = np.linspace(x.min(), x.max(), 250)
    yi = np.linspace(y.min(), y.max(), 250)
    Xi, Yi = np.meshgrid(xi, yi)
    Pi = interp(Xi, Yi)

    cf = ax.contourf(Xi, Yi, Pi, levels=20, cmap="inferno")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Peak-to-peak pressure (kPa)")

    ax.scatter(x, y, c=p, cmap="inferno", edgecolors="white", linewidths=1.2, s=70, zorder=3)
    for r in uniq:
        label = f"{r['p']:.0f}"
        ax.annotate(label, (r["x"], r["y"]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7, color="white",
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Aneurysm-channel hydrophone field map, XY plane at Z = 13 mm\n"
                  "2 MHz, 30 Vpp drive, SN 4746",
                  fontsize=11)
    ax.set_aspect("equal")
    ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    out = os.path.join(FIG_DIR, "field_map_2D.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def plot_profiles(rows):
    uniq = unique_points(rows)

    def line(yfix=None, xfix=None):
        if yfix is not None:
            pts = sorted([r for r in uniq if r["y"] == yfix], key=lambda r: r["x"])
            return [r["x"] for r in pts], [r["p"] for r in pts]
        else:
            pts = sorted([r for r in uniq if r["x"] == xfix], key=lambda r: r["y"])
            return [r["y"] for r in pts], [r["p"] for r in pts]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for yfix, color, label in [(9, "#d62728", "Y=9"), (11, "#9467bd", "Y=11")]:
        xs, ps = line(yfix=yfix)
        axes[0].plot(xs, ps, "o-", color=color, label=label)
    axes[0].set_xlabel("X (mm)  [Z=13]")
    axes[0].set_ylabel("Peak-to-peak pressure (kPa)")
    axes[0].set_title("X-scans")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for xfix, color, label in [(9, "#1f77b4", "X=9"), (10, "#2ca02c", "X=10"), (11, "#ff7f0e", "X=11")]:
        ys, ps = line(xfix=xfix)
        axes[1].plot(ys, ps, "o-", color=color, label=label)
    axes[1].set_xlabel("Y (mm)  [Z=13]")
    axes[1].set_title("Y-scans")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("Beam profile cuts through the aneurysm channel "
                  "(2 MHz, 30 Vpp drive)")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "field_profiles.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    data = load_data()
    plot_2d_map(data)
    plot_profiles(data)
