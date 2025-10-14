#!/usr/bin/env python3
# make_data_based_figure_two_mb.py
# Panel A: measured velocity (quiver/stream) + two MB trajectories + core circle
# Panel B: analytic Rankine pressure centered at estimated vortex center + trajectories
#
# OUTPUT:
#   - <outbase>_panelA.png/.svg, <outbase>_panelB.png/.svg
#   - (optional) <outbase>_combined.png/.svg

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# ======================== USER INPUT ========================
CSV_PATH   = r"C:\Users\M4\VSCode_Projects\Ultrasound-Swarm-Microbubbles-Navigating-Vortices-to-Target-and-Fill-Aneurysms\Excel_data_velocity_comsol\Velocity_2d_5cm.csv"
# 👉 point this at your two-bubble CSV:
TRAJ_PATH  = r"C:\Users\M4\VSCode_Projects\Ultrasound-Swarm-Microbubbles-Navigating-Vortices-to-Target-and-Fill-Aneurysms\trajectory_two_mb.csv"

GAMMA      = 0.95       # Circulation Γ (same units as your CSV)
RHO        = 1000.0     # Fluid density
A_OVERRIDE = 15         # Core radius a. Set to None to auto-estimate from data

OUTBASE         = "fig_data"    # Base name; files get _panelA/_panelB/_combined suffixes
OUT_DPI         = 600           # Output DPI for PNGs
GRID_N          = 120           # Interpolation resolution
USE_STREAM      = False         # True: streamplot; False: quiver
SAVE_COMBINED   = True          # Also save the original 2-panel combined figure

# Style
FONT_BASE       = 18
TITLE_SIZE      = 18
LABEL_SIZE      = 24
TICK_SIZE       = 20
LEGEND_SIZE     = 18

# Trajectory colors/width
TRAJ1_COLOR     = 'r'    # MB1 red
TRAJ2_COLOR     = 'b'    # MB2 blue
START_COLOR     = 'g'    # green start markers
PATH_LW         = 3.5    # thicker paths
CORE_EDGE_COLOR = 'k'    # core circle black
PANEL_B_CMAP    = plt.cm.gray_r  # black→white gradient for pressure panel
# ====================== END USER INPUT ======================

# --------- global matplotlib font + size (force Arial) ----------
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": FONT_BASE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
})

def estimate_vortex_center(df: pd.DataFrame):
    X = df["x"].to_numpy(float)
    Y = df["y"].to_numpy(float)
    U = df["u"].to_numpy(float)
    V = df["v"].to_numpy(float)

    spd = np.hypot(U, V)
    # Keep mid band and discard slowest 30%
    xlo, xhi = np.percentile(X, [5, 95])
    ylo, yhi = np.percentile(Y, [5, 95])
    slo = np.percentile(spd, 30)
    m = (X >= xlo) & (X <= xhi) & (Y >= ylo) & (Y <= yhi) & (spd > slo)

    if m.sum() < 200:
        j = int(np.argmin(spd))
        return float(X[j]), float(Y[j])

    X = X[m]; Y = Y[m]; U = U[m]; V = V[m]
    Umag = np.hypot(U, V) + 1e-12
    ux, uy = U/Umag, V/Umag

    xs = np.linspace(np.percentile(X, 10), np.percentile(X, 90), 60)
    ys = np.linspace(np.percentile(Y, 10), np.percentile(Y, 90), 60)

    R0 = 0.35 * min(X.ptp(), Y.ptp())
    best_cost, best_xy = 1e99, (np.median(X), np.median(Y))

    for xc in xs:
        dx = X - xc
        for yc in ys:
            dy = Y - yc
            r  = np.hypot(dx, dy) + 1e-12
            rx, ry = dx/r, dy/r
            tx, ty = -ry, rx
            ur = ux*rx + uy*ry
            ut = ux*tx + uy*ty
            w  = np.exp(-(r*r)/(2*R0*R0))
            cost = (w*(ur*ur)).sum()/w.sum() + 0.2*(w*(1.0 - np.abs(ut))).sum()/w.sum()
            if cost < best_cost:
                best_cost, best_xy = cost, (xc, yc)

    return float(best_xy[0]), float(best_xy[1])

def infer_core_radius(df: pd.DataFrame, xc: float, yc: float) -> float:
    X = df["x"].to_numpy(float)
    Y = df["y"].to_numpy(float)
    U = df["u"].to_numpy(float)
    V = df["v"].to_numpy(float)
    r = np.hypot(X - xc, Y - yc)
    s = np.hypot(U, V)
    rlo, rhi = np.percentile(r, [30, 95])
    band = (r >= rlo) & (r <= rhi)
    if band.sum() > 0:
        r_peak = r[band][s[band].argmax()]
    else:
        r_peak = np.median(r)
    return float(r_peak)

def idw_grid(df: pd.DataFrame, gx: np.ndarray, gy: np.ndarray, k: int = 8):
    """Inverse-distance weighted interpolation of u,v onto a regular grid."""
    pts = df[['x','y']].to_numpy(float)
    U = df['u'].to_numpy(float)
    V = df['v'].to_numpy(float)
    tree = cKDTree(pts)
    Xg, Yg = np.meshgrid(gx, gy)
    XY = np.column_stack([Xg.ravel(), Yg.ravel()])
    d, idx = tree.query(XY, k=k)
    if np.isscalar(d):
        d = np.array([d]); idx = np.array([idx])
    w = 1.0/(d + 1e-12)
    w = w / w.sum(axis=1, keepdims=True)
    Ug = (w * U[idx]).sum(axis=1).reshape(Xg.shape)
    Vg = (w * V[idx]).sum(axis=1).reshape(Xg.shape)
    return Xg, Yg, Ug, Vg

def rankine_pressure(Gamma: float, a: float, rho: float,
                     X: np.ndarray, Y: np.ndarray, xc: float, yc: float):
    """Rankine-consistent pressure (low in the core)."""
    xr, yr = X - xc, Y - yc
    r = np.sqrt(xr**2 + yr**2) + 1e-12
    p = np.where(r < a,
                 -(rho*Gamma**2)/(4*np.pi**2*a**2) + (rho*Gamma**2)/(8*np.pi**2)*(r**2/a**4),
                 -(rho*Gamma**2)/(8*np.pi**2)*(1/r**2))
    return p

def plot_panel_A(ax, Xg, Yg, Ug, Vg,
                 tx1, ty1, tx2, ty2,
                 xc, yc, a, xmin, xmax, ymin, ymax, use_stream):
    speed = np.hypot(Ug, Vg)
    if use_stream:
        ax.streamplot(Xg, Yg, Ug, Vg, color=speed, density=1.4, linewidth=1.0)
    else:
        mags = speed + 1e-12
        Un = Ug/mags; Vn = Vg/mags
        ax.quiver(Xg[::3,::3], Yg[::3,::3], Un[::3,::3], Vn[::3,::3],
                  speed[::3,::3], angles='xy', scale=25, alpha=0.9)

    # MB1
    ax.plot(tx1, ty1, '-', lw=PATH_LW, color=TRAJ1_COLOR, label='MB1 path')
    ax.plot(tx1[0], ty1[0], 'o', ms=7, color=START_COLOR, label='start')
    ax.plot(tx1[-1], ty1[-1], 'o', ms=7, color=TRAJ1_COLOR, label='end MB1')

    # MB2 (if provided)
    if tx2 is not None and ty2 is not None:
        ax.plot(tx2, ty2, '-', lw=PATH_LW, color=TRAJ2_COLOR, label='MB2 path')
        ax.plot(tx2[0], ty2[0], 'o', ms=7, color=START_COLOR)
        ax.plot(tx2[-1], ty2[-1], 'o', ms=7, color=TRAJ2_COLOR)

    # core circle
    ax.add_patch(plt.Circle((xc, yc), a, fill=False, lw=2, edgecolor=CORE_EDGE_COLOR))

    ax.set_aspect('equal', 'box')
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.grid(alpha=0.25)
    ax.set_xlabel('x', fontfamily="Arial")
    ax.set_ylabel('y', fontfamily="Arial")
    leg = ax.legend(loc='upper right', frameon=False, ncol=2)
    for txt in leg.get_texts():
        txt.set_fontfamily("Arial")
        txt.set_fontsize(LEGEND_SIZE)

def plot_panel_B(ax, Xg, Yg, gamma, a, rho, xc, yc,
                 tx1, ty1, tx2, ty2, xmin, xmax, ymin, ymax):
    p = rankine_pressure(gamma, a, rho, Xg, Yg, xc, yc)
    c = ax.contourf(Xg, Yg, p, levels=40, cmap=PANEL_B_CMAP)
    # trajectories on top
    ax.plot(tx1, ty1, '-', lw=PATH_LW, color=TRAJ1_COLOR)
    ax.plot(tx1[0], ty1[0], 'o', ms=7, color=START_COLOR)
    ax.plot(tx1[-1], ty1[-1], 'o', ms=7, color=TRAJ1_COLOR)
    if tx2 is not None and ty2 is not None:
        ax.plot(tx2, ty2, '-', lw=PATH_LW, color=TRAJ2_COLOR)
        ax.plot(tx2[0], ty2[0], 'o', ms=7, color=START_COLOR)
        ax.plot(tx2[-1], ty2[-1], 'o', ms=7, color=TRAJ2_COLOR)

    # core circle
    ax.add_patch(plt.Circle((xc, yc), a, color='w', fill=False, lw=2))
    # inward pressure direction field (sampled symmetrically)
    xr = Xg - xc; yr = Yg - yc; rr = np.sqrt(xr**2 + yr**2) + 1e-12
    stride = 8
    ii = np.arange(stride // 2, Xg.shape[0], stride)
    jj = np.arange(stride // 2, Xg.shape[1], stride)
    Xi = Xg[np.ix_(ii, jj)]
    Yi = Yg[np.ix_(ii, jj)]
    Ui = (-xr / rr)[np.ix_(ii, jj)]
    Vi = (-yr / rr)[np.ix_(ii, jj)]
    ax.quiver(Xi, Yi, Ui, Vi, alpha=0.85, scale=20, width=0.004,
              color='white', edgecolor='black', linewidths=0.3)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.grid(alpha=0.25)
    ax.set_xlabel('x', fontfamily="Arial")
    ax.set_ylabel('y', fontfamily="Arial")

def main(csv_path: str, traj_path: str, gamma: float, rho: float,
         a_override, outbase: str, grid_N: int, use_stream: bool,
         save_combined: bool, out_dpi: int):

    # Load velocity field
    if not os.path.isfile(csv_path) or not os.path.isfile(traj_path):
        print("[ERROR] Missing CSV file(s).")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if not {'x','y','u','v'} <= set(df.columns):
        print("[ERROR] Velocity CSV must contain columns: x, y, u, v")
        sys.exit(1)

    xmin, xmax = df['x'].min(), df['x'].max()
    ymin, ymax = df['y'].min(), df['y'].max()

    xc, yc = estimate_vortex_center(df)
    if a_override is None:
        a = infer_core_radius(df, xc, yc)
    else:
        a = float(a_override)

    gx = np.linspace(xmin, xmax, grid_N)
    gy = np.linspace(ymin, ymax, grid_N)
    Xg, Yg, Ug, Vg = idw_grid(df, gx, gy, k=8)

    # -------- read trajectory (supports 1 or 2 bubbles) --------
    traj = pd.read_csv(traj_path)
    cols = set(traj.columns.str.strip())
    if {'x1','y1','x2','y2'} <= cols:
        tx1 = traj['x1'].to_numpy(float); ty1 = traj['y1'].to_numpy(float)
        tx2 = traj['x2'].to_numpy(float); ty2 = traj['y2'].to_numpy(float)
    elif {'x','y'} <= cols:
        tx1 = traj['x'].to_numpy(float);  ty1 = traj['y'].to_numpy(float)
        tx2 = None; ty2 = None
    else:
        print("[ERROR] Trajectory CSV must contain (x,y) or (x1,y1,x2,y2).")
        sys.exit(1)

    # Panel A
    figA, axA = plt.subplots(figsize=(7, 6), constrained_layout=True)
    plot_panel_A(axA, Xg, Yg, Ug, Vg, tx1, ty1, tx2, ty2, xc, yc, a,
                 xmin, xmax, ymin, ymax, use_stream)
    figA.savefig(f'{outbase}_panelA.png', dpi=out_dpi)
    figA.savefig(f'{outbase}_panelA.svg')
    plt.close(figA)

    # Panel B
    figB, axB = plt.subplots(figsize=(7, 6), constrained_layout=True)
    plot_panel_B(axB, Xg, Yg, gamma, a, rho, xc, yc, tx1, ty1, tx2, ty2,
                 xmin, xmax, ymin, ymax)
    figB.savefig(f'{outbase}_panelB.png', dpi=out_dpi)
    figB.savefig(f'{outbase}_panelB.svg')
    plt.close(figB)

    # Combined (optional)
    if save_combined:
        fig, axs = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        plot_panel_A(axs[0], Xg, Yg, Ug, Vg, tx1, ty1, tx2, ty2, xc, yc, a,
                     xmin, xmax, ymin, ymax, use_stream)
        plot_panel_B(axs[1], Xg, Yg, gamma, a, rho, xc, yc, tx1, ty1, tx2, ty2,
                     xmin, xmax, ymin, ymax)
        fig.savefig(f'{outbase}_combined.png', dpi=out_dpi)
        fig.savefig(f'{outbase}_combined.svg')
        plt.close(fig)

    print(f'Center used: (xc, yc)=({xc:.3f}, {yc:.3f})  |  a={a:.3f}')
    print(f'Saved {outbase}_panelA.(png|svg) and {outbase}_panelB.(png|svg)'
          + (f' plus {outbase}_combined.(png|svg)' if save_combined else ''))

if __name__ == '__main__':
    main(CSV_PATH, TRAJ_PATH, GAMMA, RHO, A_OVERRIDE, OUTBASE,
         GRID_N, USE_STREAM, SAVE_COMBINED, OUT_DPI)
