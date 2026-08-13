"""
3D scatter of the verified hydrophone field-mapping data (data_table.csv),
styled to match a companion figure from another experiment on this project
(soft panes, PDMS channel box + height-marker ring, drop lines, Blues
colormap, size+color pressure encoding).

DATA: read from data_table.csv (our 19 verified CSV15-33 points, corrected
per Section 11 of hydrophone_experiment_summary.md). This script does NOT
use the placeholder rows from the reference/style script -- only its look.

CHANNEL BOX geometry (as given by the user, 12 Aug 2026):
  - PDMS block height (z): 5.5 mm, sitting on the transducer face (z=0)
  - Fluid channel height inside the block: ~5 mm -> drawn as a marker ring
    at z=5, same convention as the reference script.
  - Cross-section width (y): 7 mm, "from the foot of the piezo", y = 8 to 15
  - Long-axis width (x): 20 mm, matching the PZT aperture. No offset was
    given for x, so the box is CENTERED ON X=10 (the original scan center,
    see Sections 4/8 of the summary) -> x = 0 to 20.
    ADAPT X0_BOX / X1_BOX BELOW IF THE TRANSDUCER IS NOT CENTERED THERE.

Output: figures/field_scatter_3D_channel_style.png
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(HERE, "data_table.csv")
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---- load OUR verified data (unchanged) ----
rows = []
with open(DATA_CSV) as f:
    for row in csv.DictReader(f):
        rows.append((float(row["X_mm"]), float(row["Y_mm"]), float(row["Z_mm"]),
                      float(row["Pressure_pp_kPa"])))

xs = [r[0] for r in rows]
ys = [r[1] for r in rows]
zs = [r[2] for r in rows]
peak_kpa = [r[3] for r in rows]
vmin, vmax = min(peak_kpa), max(peak_kpa)

try:
    plt.rcParams["font.family"] = "Liberation Sans"
except Exception:
    pass

fig = plt.figure(figsize=(13, 11), facecolor="white")
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("white")

# soft panes + gridlines
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.set_facecolor((0.97, 0.97, 0.96, 1.0))
    axis.pane.set_edgecolor((0.85, 0.85, 0.82, 1.0))
    axis.line.set_color((0.7, 0.7, 0.66, 1.0))
ax.grid(True, color=(0.88, 0.88, 0.85), linewidth=0.5)
ax.tick_params(colors="#52514e", labelsize=17, pad=8)

zbase, ztop = 0, max(zs) + 1

# --- channel block: PDMS, cross-section 7 mm (y) x 20 mm (x, along PZT width) ---
X0_BOX, X1_BOX = 0, 20     # 20 mm wide, centered on X=10 (scan center) -- adjust if needed
Y0_BOX, Y1_BOX = 8, 15     # 7 mm cross-section, "from the foot of the piezo"
Z0_BOX, Z1_BOX = 0, 5.5    # PDMS block height
CHANNEL_HEIGHT_Z = 5       # fluid channel height inside the block (marker ring)

box_color = "#B5D4F4"      # light blue = PDMS (clear/transparent material)
edge_color = "#185FA5"


def face(pts):
    return Poly3DCollection([pts], facecolor=box_color, edgecolor=edge_color,
                             linewidth=0.8, alpha=0.22, zorder=1)


x0, x1, y0, y1, z0, z1 = X0_BOX, X1_BOX, Y0_BOX, Y1_BOX, Z0_BOX, Z1_BOX
faces = [
    [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],  # bottom
    [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],  # top
    [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],  # front
    [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],  # back
    [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],  # left
    [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],  # right
]
for f in faces:
    ax.add_collection3d(face(f))

# height marker outline at z = channel height, through the block
ring = [(x0, y0, CHANNEL_HEIGHT_Z), (x1, y0, CHANNEL_HEIGHT_Z),
         (x1, y1, CHANNEL_HEIGHT_Z), (x0, y1, CHANNEL_HEIGHT_Z), (x0, y0, CHANNEL_HEIGHT_Z)]
rx, ry, rz = zip(*ring)
ax.plot(rx, ry, rz, color="#993C1D", linewidth=2, zorder=2)

# drop lines from each measurement point down to z = 0 (depth cue)
for x, y, z in zip(xs, ys, zs):
    ax.plot([x, x], [y, y], [zbase, z], color=(0.75, 0.75, 0.72), linewidth=0.6, zorder=1)

# scatter: marker size AND color both encode pressure
sizes = [90 + (v - vmin) / (vmax - vmin) * 340 for v in peak_kpa]
sc = ax.scatter(xs, ys, zs, c=peak_kpa, cmap="Blues", s=sizes,
                 edgecolors="#0c447c", linewidths=0.8, vmin=vmin, vmax=vmax,
                 depthshade=True, zorder=3)

legend_handles = [
    Patch(facecolor=box_color, edgecolor=edge_color, alpha=0.5,
          label=f"PDMS channel block (0-{Z1_BOX:g} mm, along x)"),
    Line2D([0], [0], color="#993C1D", linewidth=2,
           label=f"channel height = {CHANNEL_HEIGHT_Z:g} mm"),
    Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#378ADD",
           markeredgecolor="#0c447c", markersize=8, label="hydrophone reading"),
]
ax.legend(handles=legend_handles, loc="upper left", fontsize=16, frameon=False,
          bbox_to_anchor=(0.0, 0.88))

ax.set_xlabel("x (mm)", fontsize=20, color="#2c2c2a", labelpad=22)
ax.set_ylabel("y (mm)", fontsize=20, color="#2c2c2a", labelpad=22)
ax.set_zlabel("z (mm)", fontsize=20, color="#2c2c2a", labelpad=16)
ax.set_xlim(min(x0, min(xs)) - 1, max(x1, max(xs)) + 1)
ax.set_ylim(min(y0, min(ys)) - 1, max(y1, max(ys)) + 1)
ax.set_zlim(zbase, ztop)
ax.set_title("Provisional peak-to-peak pressure vs. channel position\n"
             "1.7 MHz, 20 Vpp drive, SN 4746 (no booster correction)",
             fontsize=22, color="#0b0b0b", pad=32)
ax.view_init(elev=22, azim=-55)

cbar = fig.colorbar(sc, ax=ax, shrink=0.45, pad=0.14, aspect=16)
cbar.set_label("Peak-to-peak pressure (kPa)", fontsize=17, color="#52514e")
cbar.ax.tick_params(labelsize=14, colors="#52514e")
cbar.outline.set_edgecolor((0.85, 0.85, 0.82))

fig.tight_layout(pad=2.0)
out = os.path.join(FIG_DIR, "field_scatter_3D_channel_style.png")
fig.savefig(out, dpi=180, facecolor="white")
print("saved", out)
