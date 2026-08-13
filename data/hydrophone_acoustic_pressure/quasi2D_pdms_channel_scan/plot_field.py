"""
Plot the Precision Acoustics hydrophone spatial field-mapping data
(1.7 MHz, 20 Vpp transducer drive, hydrophone SN 4746, 50 Ohm termination).

Reproducibility:
- Reads data_table.csv (X_mm, Y_mm, Z_mm, Vpp_mV, Pressure_pp_kPa) in this folder.
- That table was derived from the raw oscilloscope waveforms in raw_csv/
  (Tektronix TBS2072 exports, TIME/CH1 columns) by taking full-trace
  peak-to-peak voltage, then converting to pressure using the calibrated
  hydrophone sensitivity S = 655.63 mV/MPa at 1.7 MHz (linear interpolation
  between the 1 MHz and 2 MHz calibration points from certificate 20251211-03).

IMPORTANT CAVEAT: the pressure values are PROVISIONAL. They assume no
external booster amplifier was in the signal chain. If the Precision
Acoustics HA2 / Hydrophone Booster Amplifier (typical gain 27 dB, G~22.4)
was in line during this scan, all pressure values must be divided by G
before use. This does not change the SHAPE of the field, only its absolute
scale, so the plots below are valid either way -- only the pressure axis
labels/values would need rescaling. See hydrophone_experiment_summary.md
Section 6 for details.

Outputs (written to ./figures/):
- field_map_2D.png       : interpolated pressure map of the XY scan at Z=6 mm
- field_profiles.png     : X, Y and Z line profiles through the peak
- field_scatter_3D.png   : 3D scatter of all 19 measurement points
"""

import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(HERE, "data_table.csv")
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_data(path=DATA_CSV):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "csv": int(row["CSV"]),
                    "x": float(row["X_mm"]),
                    "y": float(row["Y_mm"]),
                    "z": float(row["Z_mm"]),
                    "vpp": float(row["Vpp_mV"]),
                    "p": float(row["Pressure_pp_kPa"]),
                }
            )
    return rows


def plot_2d_map(rows):
    """Interpolated pressure map of the XY scan at Z = 6 mm."""
    z6 = [r for r in rows if r["z"] == 6]
    x = np.array([r["x"] for r in z6])
    y = np.array([r["y"] for r in z6])
    p = np.array([r["p"] for r in z6])

    fig, ax = plt.subplots(figsize=(8, 6.5))

    tri = Triangulation(x, y)
    interp = LinearTriInterpolator(tri, p)
    xi = np.linspace(x.min(), x.max(), 200)
    yi = np.linspace(y.min(), y.max(), 200)
    Xi, Yi = np.meshgrid(xi, yi)
    Pi = interp(Xi, Yi)

    cf = ax.contourf(Xi, Yi, Pi, levels=20, cmap="inferno")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Provisional pressure, p-p (kPa)")

    sc = ax.scatter(x, y, c=p, cmap="inferno", edgecolors="white", linewidths=1.2, s=70, zorder=3)
    for r in z6:
        ax.annotate(f"{r['p']:.0f}", (r["x"], r["y"]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7, color="white")

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Hydrophone field map, XY plane at Z = 6 mm\n"
                  "1.7 MHz, 20 Vpp drive, SN 4746 (provisional, no booster correction)",
                  fontsize=11)
    ax.set_aspect("equal")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(FIG_DIR, "field_map_2D.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def plot_profiles(rows):
    """1D line profiles through the peak: X-scan, Y-scan, Z-scan."""
    x_scan = sorted([r for r in rows if r["y"] == 12 and r["z"] == 6], key=lambda r: r["x"])
    y_scan = sorted([r for r in rows if r["x"] == 10 and r["z"] == 6], key=lambda r: r["y"])
    z_scan = sorted([r for r in rows if r["x"] == 10 and r["y"] == 12], key=lambda r: r["z"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot([r["x"] for r in x_scan], [r["p"] for r in x_scan], "o-", color="#d62728")
    axes[0].set_xlabel("X (mm)  [Y=12, Z=6]")
    axes[0].set_ylabel("Provisional pressure, p-p (kPa)")
    axes[0].set_title("X-scan (lateral)")
    axes[0].grid(alpha=0.3)

    axes[1].plot([r["y"] for r in y_scan], [r["p"] for r in y_scan], "o-", color="#1f77b4")
    axes[1].set_xlabel("Y (mm)  [X=10, Z=6]")
    axes[1].set_title("Y-scan (lateral)")
    axes[1].grid(alpha=0.3)

    axes[2].plot([r["z"] for r in z_scan], [r["p"] for r in z_scan], "o-", color="#2ca02c")
    axes[2].set_xlabel("Z (mm)  [X=10, Y=12]")
    axes[2].set_title("Z-scan (axial/depth)")
    axes[2].grid(alpha=0.3)

    fig.suptitle("Beam profile cuts through the pressure maximum "
                  "(1.7 MHz, 20 Vpp drive, provisional pressures)")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "field_profiles.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def plot_3d_scatter(rows):
    """3D scatter of all measurement points, colour/size by pressure."""
    x = np.array([r["x"] for r in rows])
    y = np.array([r["y"] for r in rows])
    z = np.array([r["z"] for r in rows])
    p = np.array([r["p"] for r in rows])

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(x, y, z, c=p, cmap="inferno", s=60 + 4 * p, edgecolors="k", linewidths=0.5)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.1)
    cbar.set_label("Provisional pressure, p-p (kPa)")

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.invert_zaxis()  # Z increases with depth from transducer
    ax.set_title("3D scatter of hydrophone scan points\n"
                  "1.7 MHz, 20 Vpp drive, SN 4746 (provisional pressures)")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "field_scatter_3D.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    data = load_data()
    plot_2d_map(data)
    plot_profiles(data)
    plot_3d_scatter(data)
