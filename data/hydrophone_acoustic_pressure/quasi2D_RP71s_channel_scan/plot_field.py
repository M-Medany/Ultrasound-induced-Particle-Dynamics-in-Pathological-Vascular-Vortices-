"""
Plot the quasi-2D PDMS channel hydrophone scan measured with the RP Acoustics
RP 71s PVDF hydrophone (1.7 MHz, 20 Vpp drive, x100 amplifier).

Reads data_table.csv (X_mm, Y_mm, Z_mm, Vpp_mV, Pressure_pp_kPa) and renders a
3D channel-box scatter (pressure encoded by marker size + Blues colour).

CAVEAT: pressures use S = 3.7 mV/bar, RP Acoustics' certified sensitivity for
1-100 kHz only (out of band at 1.7 MHz) -- treat kPa as a working estimate,
likely an overestimate. See README.md.
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

xs, ys, zs, p = [], [], [], []
with open(os.path.join(HERE, "data_table.csv")) as f:
    for r in csv.DictReader(f):
        xs.append(float(r["X_mm"])); ys.append(float(r["Y_mm"]))
        zs.append(float(r["Z_mm"])); p.append(float(r["Pressure_pp_kPa"]))
vmin, vmax = min(p), max(p)

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]})
fig = plt.figure(figsize=(13, 11), facecolor="white")
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("white")
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.set_facecolor((0.97, 0.97, 0.96, 1.0))
    axis.pane.set_edgecolor((0.85, 0.85, 0.82, 1.0))
    axis.line.set_color((0.7, 0.7, 0.66, 1.0))
ax.grid(True, color=(0.88, 0.88, 0.85), linewidth=0.5)
ax.tick_params(colors="#52514e", labelsize=15, pad=6)

# PDMS channel box (dimensions still provisional -- see README)
x0, x1, y0, y1, z0, z1 = 6.5, 14.5, 10, 12, 0, 6
HEIGHT_MARKER_Z = 5.5
box_color, edge_color = "#B5D4F4", "#185FA5"
faces = [
    [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)], [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],
    [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)], [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)],
    [(x0,y0,z0),(x0,y1,z0),(x0,y1,z1),(x0,y0,z1)], [(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)]]
for fp in faces:
    ax.add_collection3d(Poly3DCollection([fp], facecolor=box_color, edgecolor=edge_color,
                                         linewidth=0.8, alpha=0.22, zorder=1))
ring = [(x0,y0,HEIGHT_MARKER_Z),(x1,y0,HEIGHT_MARKER_Z),(x1,y1,HEIGHT_MARKER_Z),
        (x0,y1,HEIGHT_MARKER_Z),(x0,y0,HEIGHT_MARKER_Z)]
rx, ry, rz = zip(*ring)
ax.plot(rx, ry, rz, color="#993C1D", linewidth=2, zorder=2)

for x, y, z in zip(xs, ys, zs):
    ax.plot([x, x], [y, y], [0, z], color=(0.75, 0.75, 0.72), linewidth=0.6, zorder=1)

sizes = [90 + (v - vmin) / (vmax - vmin) * 340 for v in p]
sc = ax.scatter(xs, ys, zs, c=p, cmap="Blues", s=sizes, edgecolors="#0c447c",
                linewidths=0.8, vmin=vmin, vmax=vmax, depthshade=True, zorder=3)

legend_handles = [
    Patch(facecolor=box_color, edgecolor=edge_color, alpha=0.5, label="PDMS channel"),
    Line2D([0],[0], color="#993C1D", linewidth=2, label=f"height marker = {HEIGHT_MARKER_Z} mm"),
    Line2D([0],[0], marker="o", linestyle="", markerfacecolor="#378ADD",
           markeredgecolor="#0c447c", markersize=8, label="hydrophone reading"),
]
ax.legend(handles=legend_handles, loc="upper left", fontsize=14, frameon=False, bbox_to_anchor=(0.0, 0.9))

ax.set_xlabel("x (mm)", fontsize=19, color="#2c2c2a", labelpad=20)
ax.set_ylabel("y (mm)", fontsize=19, color="#2c2c2a", labelpad=20)
ax.set_zlabel("z (mm)", fontsize=19, color="#2c2c2a", labelpad=14)
ax.set_xlim(min(x0, min(xs)) - 1, max(x1, max(xs)) + 1)
ax.set_ylim(min(ys) - 1, max(ys) + 1)
ax.set_zlim(0, max(zs) + 1)
ax.view_init(elev=22, azim=-55)

cbar = fig.colorbar(sc, ax=ax, shrink=0.45, pad=0.12, aspect=16)
cbar.set_label("Peak-to-peak pressure (kPa)", fontsize=16, color="#52514e")
cbar.ax.tick_params(labelsize=13, colors="#52514e")
cbar.outline.set_edgecolor((0.85, 0.85, 0.82))

fig.tight_layout(pad=2.0)
out = os.path.join(FIG, "field_scatter_3D.png")
fig.savefig(out, dpi=180, facecolor="white")
print("wrote", out)
