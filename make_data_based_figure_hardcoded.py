#!/usr/bin/env python3
# make_data_based_figure_hardcoded.py
# Two-panel figure from your CSV velocity field and a simulated trajectory.
#
# Panel A: measured velocity (quiver/stream) + bubble trajectory + core circle
# Panel B: analytic Rankine pressure centered at estimated vortex center
#
# HOW TO USE (VS Code / double-click):
#   - Set the paths and parameters in the "USER INPUT" block below
#   - Run the script (no command-line arguments needed)
#
# OUTPUT:
#   - fig_data.png and fig_data.svg (or whatever you set in outbase)
#
# INPUTS:
#   - Velocity CSV with columns: x, y, u, v
#   - Trajectory CSV with columns: x, y (and optionally t)

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


# ======================== USER INPUT ========================
CSV_PATH  = r"C:\Users\M4\VSCode_Projects\Ultrasound-Swarm-Microbubbles-Navigating-Vortices-to-Target-and-Fill-Aneurysms\Excel_data_velocity_comsol\Velocity_2d_5cm.csv"
TRAJ_PATH = r"C:\Users\M4\VSCode_Projects\Ultrasound-Swarm-Microbubbles-Navigating-Vortices-to-Target-and-Fill-Aneurysms\trajectory.csv"

GAMMA      = 0.95       # Circulation Γ (same units as your CSV)
RHO        = 1000.0     # Fluid density
A_OVERRIDE = 15         # Core radius a. Set to None to auto-estimate from data

OUTBASE    = "fig_data" # Output file basename (png and svg will be created)
GRID_N     = 120        # Interpolation resolution (higher = smoother/slower)
USE_STREAM = False      # True: streamplot (pretty, slower). False: quiver arrows
# ====================== END USER INPUT ======================


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


def main(csv_path: str, traj_path: str, gamma: float, rho: float,
         a_override, outbase: str, grid_N: int, use_stream: bool):

    # Sanity checks
    if not os.path.isfile(csv_path):
        print(f"[ERROR] Velocity CSV not found:\n  {csv_path}")
        sys.exit(1)
    if not os.path.isfile(traj_path):
        print(f"[ERROR] Trajectory CSV not found:\n  {traj_path}")
        sys.exit(1)

    # Load data
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if not {'x','y','u','v'} <= set(df.columns):
        print("[ERROR] Velocity CSV must contain columns: x, y, u, v")
        sys.exit(1)

    xmin, xmax = df['x'].min(), df['x'].max()
    ymin, ymax = df['y'].min(), df['y'].max()

    # Vortex center & core radius
    xc, yc = estimate_vortex_center(df)
    if a_override is None:
        a = infer_core_radius(df, xc, yc)
    else:
        a = float(a_override)

    # Interpolate velocity to grid
    gx = np.linspace(xmin, xmax, grid_N)
    gy = np.linspace(ymin, ymax, grid_N)
    Xg, Yg, Ug, Vg = idw_grid(df, gx, gy, k=8)

    # Load trajectory
    traj = pd.read_csv(traj_path)
    if not {'x','y'} <= set(traj.columns):
        print("[ERROR] Trajectory CSV must contain columns: x, y (and optionally t)")
        sys.exit(1)
    tx = traj['x'].to_numpy(float)
    ty = traj['y'].to_numpy(float)

    # Build figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    speed = np.hypot(Ug, Vg)
    if use_stream:
        axs[0].streamplot(Xg, Yg, Ug, Vg, color=speed, density=1.4, linewidth=1.0)
        mappable = None
    else:
        mags = speed + 1e-12
        Un = Ug/mags; Vn = Vg/mags
        q = axs[0].quiver(Xg[::3,::3], Yg[::3,::3], Un[::3,::3], Vn[::3,::3],
                          speed[::3,::3], angles='xy', scale=25, alpha=0.9)
        mappable = q

    axs[0].plot(tx, ty, '-', lw=2, label='MB trajectory')
    axs[0].plot(tx[0], ty[0], 'o', ms=6, label='start')
    axs[0].plot(tx[-1], ty[-1], 'o', ms=6, label='end')
    axs[0].add_patch(plt.Circle((xc, yc), a, fill=False, lw=2))
    axs[0].set_aspect('equal', 'box')
    axs[0].set_xlim(xmin, xmax)
    axs[0].set_ylim(ymin, ymax)
    axs[0].grid(alpha=0.25)
    axs[0].set_title('Measured velocity + MB trajectory')
    axs[0].legend(loc='upper right', frameon=False)
    if mappable is not None:
        fig.colorbar(mappable, ax=axs[0], label='Speed (CSV units/s)')

    # Pressure panel
    p = rankine_pressure(gamma, a, rho, Xg, Yg, xc, yc)
    cs = axs[1].contourf(Xg, Yg, p, levels=40)
    axs[1].plot(tx, ty, 'w-', lw=2)
    axs[1].plot(tx[0], ty[0], 'wo', ms=6)
    axs[1].plot(tx[-1], ty[-1], 'wo', ms=6)
    axs[1].add_patch(plt.Circle((xc, yc), a, color='w', fill=False, lw=2))
    # Inward arrows indicating -∇p direction (schematic)
    xr = Xg - xc; yr = Yg - yc; rr = np.sqrt(xr**2 + yr**2) + 1e-12
    axs[1].quiver(Xg[::8,::8], Yg[::8,::8], -xr[::8,::8]/rr[::8,::8], -yr[::8,::8]/rr[::8,::8],
                  alpha=0.7, scale=20, width=0.004)
    axs[1].set_aspect('equal', 'box')
    axs[1].set_xlim(xmin, xmax)
    axs[1].set_ylim(ymin, ymax)
    axs[1].grid(alpha=0.25)
    axs[1].set_title('Analytic pressure (Rankine) + trajectory')
    fig.colorbar(cs, ax=axs[1], label='Pressure (arb.)')

    # Save
    plt.savefig(f'{outbase}.png', dpi=300)
    plt.savefig(f'{outbase}.svg')
    print(f'Center used: (xc, yc)=({xc:.3f}, {yc:.3f})  |  a={a:.3f}')
    print(f'Saved {outbase}.png and {outbase}.svg')


if __name__ == '__main__':
    # Call main with the hard-coded parameters
    main(CSV_PATH, TRAJ_PATH, GAMMA, RHO, A_OVERRIDE, OUTBASE, GRID_N, USE_STREAM)
