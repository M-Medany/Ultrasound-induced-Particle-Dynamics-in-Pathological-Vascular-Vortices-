#!/usr/bin/env python3
# make_data_based_figure.py
# Two-panel figure from your CSV velocity field and a simulated trajectory.
#
# Panel A: measured velocity (quiver/stream) + bubble trajectory + core circle
# Panel B: analytic Rankine pressure centered at estimated vortex center
#
# Usage:
#   python make_data_based_figure.py --csv path/to/Velocity_2d_5cm.csv \
#       --traj path/to/trajectory.csv --gamma 0.95 --a_override 15 --out fig_data
#
# Notes:
# - The trajectory CSV is expected to have columns: x,y (and optionally t).
#   If you don't have it yet, run your simulator and save x,y over time.
# - If --a_override is omitted, we estimate a from the peak-speed radius band.

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def estimate_vortex_center(df):
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

def infer_core_radius(df, xc, yc):
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

def idw_grid(df, gx, gy, k=8):
    # Inverse-distance weighted interpolation of u,v onto grid
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

def rankine_pressure(Gamma, a, rho, X, Y, xc, yc):
    xr, yr = X - xc, Y - yc
    r = np.sqrt(xr**2 + yr**2) + 1e-12
    p = np.where(r < a,
                 -(rho*Gamma**2)/(4*np.pi**2*a**2) + (rho*Gamma**2)/(8*np.pi**2)*(r**2/a**4),
                 -(rho*Gamma**2)/(8*np.pi**2)*(1/r**2))
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='Path to CSV with columns x,y,u,v')
    ap.add_argument('--traj', required=True, help='Path to trajectory CSV with columns x,y[,t]')
    ap.add_argument('--gamma', type=float, default=0.95, help='Circulation Γ (CSV units)')
    ap.add_argument('--rho', type=float, default=1000.0, help='Density ρ')
    ap.add_argument('--a_override', type=float, default=None, help='Core radius a (if provided, skips inference)')
    ap.add_argument('--out', type=str, default='fig_data', help='Output basename')
    ap.add_argument('--grid_N', type=int, default=120, help='Grid size for interpolation')
    ap.add_argument('--stream', action='store_true', help='Use streamplot (slower) instead of quiver')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip()
    assert {'x','y','u','v'} <= set(df.columns), "CSV must contain x,y,u,v"

    xmin, xmax = df['x'].min(), df['x'].max()
    ymin, ymax = df['y'].min(), df['y'].max()

    # Vortex center & core radius
    xc, yc = estimate_vortex_center(df)
    if args.a_override is None:
        a = infer_core_radius(df, xc, yc)
    else:
        a = float(args.a_override)

    # Interpolate velocity to grid
    gx = np.linspace(xmin, xmax, args.grid_N)
    gy = np.linspace(ymin, ymax, args.grid_N)
    Xg, Yg, Ug, Vg = idw_grid(df, gx, gy, k=8)

    # Load trajectory
    traj = pd.read_csv(args.traj)
    tx = traj['x'].to_numpy(float)
    ty = traj['y'].to_numpy(float)

    # Build figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    speed = np.hypot(Ug, Vg)
    if args.stream:
        axs[0].streamplot(Xg, Yg, Ug, Vg, color=speed, cmap='viridis', density=1.4, linewidth=1.0)
    else:
        # Normalize arrows to show direction; color by speed
        mags = speed + 1e-12
        Un = Ug/mags; Vn = Vg/mags
        q = axs[0].quiver(Xg[::3,::3], Yg[::3,::3], Un[::3,::3], Vn[::3,::3],
                          speed[::3,::3], cmap='viridis', angles='xy', scale=25, alpha=0.9)
        cbar = fig.colorbar(q, ax=axs[0], label='Speed (CSV units/s)')

    axs[0].plot(tx, ty, 'k-', lw=2, label='MB trajectory')
    axs[0].plot(tx[0], ty[0], 'go', ms=6, label='start')
    axs[0].plot(tx[-1], ty[-1], 'ro', ms=6, label='end')
    axs[0].add_patch(plt.Circle((xc, yc), a, color='r', fill=False, lw=2))
    axs[0].set_aspect('equal', 'box')
    axs[0].set_xlim(xmin, xmax)
    axs[0].set_ylim(ymin, ymax)
    axs[0].grid(alpha=0.25)
    axs[0].set_title('Measured velocity + MB trajectory')
    axs[0].legend(loc='upper right', frameon=False)

    # Pressure panel
    p = rankine_pressure(args.gamma, a, args.rho, Xg, Yg, xc, yc)
    cs = axs[1].contourf(Xg, Yg, p, levels=40)
    axs[1].plot(tx, ty, 'w-', lw=2)
    axs[1].plot(tx[0], ty[0], 'go', ms=6)
    axs[1].plot(tx[-1], ty[-1], 'ro', ms=6)
    axs[1].add_patch(plt.Circle((xc, yc), a, color='w', fill=False, lw=2))
    # Inward -∇p arrows (schematic direction toward center)
    xr = Xg - xc; yr = Yg - yc; rr = np.sqrt(xr**2 + yr**2) + 1e-12
    axs[1].quiver(Xg[::8,::8], Yg[::8,::8], -xr[::8,::8]/rr[::8,::8], -yr[::8,::8]/rr[::8,::8],
                  color='white', alpha=0.7, scale=20, width=0.004)
    axs[1].set_aspect('equal', 'box')
    axs[1].set_xlim(xmin, xmax)
    axs[1].set_ylim(ymin, ymax)
    axs[1].grid(alpha=0.25)
    axs[1].set_title('Analytic pressure (Rankine) + trajectory')
    fig.colorbar(cs, ax=axs[1], label='Pressure (arb.)')

    # Save
    outbase = args.out
    plt.savefig(f'{outbase}.png', dpi=300)
    plt.savefig(f'{outbase}.svg')
    print(f'Center used: (xc, yc)=({xc:.3f}, {yc:.3f})  |  a={a:.3f}')
    print(f'Saved {outbase}.png and {outbase}.svg')

if __name__ == '__main__':
    main()
