#!/usr/bin/env python3
# make_rankine_schematic_science.py
# Science-style, small-inset 2-panel schematic:
# (A) velocity streamlines of a Rankine vortex
# (B) pressure field with inward pressure-gradient arrows
#
# Usage example:
#   python make_rankine_schematic_science.py --Gamma 1.0 --a 1.0 --rho 1.0 \
#       --out fig_rankine_inset --width 3.6 --height 1.9 --mono
#
# Outputs:
#   <out>.png (transparent) and <out>.svg (vector), both tightly cropped.

import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------- Physics helpers ----------
def rankine_velocity(Gamma, a, X, Y):
    r = np.sqrt(X**2 + Y**2) + 1e-12
    theta = np.arctan2(Y, X)
    u_theta = np.where(r < a, (Gamma/(2*np.pi))*(r/a**2), (Gamma/(2*np.pi))*(1/r))
    U = -u_theta*np.sin(theta)
    V =  u_theta*np.cos(theta)
    return U, V

def rankine_pressure(Gamma, a, rho, X, Y):
    r = np.sqrt(X**2 + Y**2) + 1e-12
    p = np.where(r < a,
                 -(rho*Gamma**2)/(4*np.pi**2*a**2) + (rho*Gamma**2)/(8*np.pi**2)*(r**2/a**4),
                 -(rho*Gamma**2)/(8*np.pi**2)*(1/r**2))
    return p

def make_trajectory(T=10.0, n=300, r0=2.0, decay=0.2, ang_rate=3.0):
    t = np.linspace(0, T, n)
    r = r0*np.exp(-decay*t)
    th = ang_rate*t
    x = r*np.cos(th)
    y = r*np.sin(th)
    return x, y

# ---------- Styling ----------
def set_science_rc(fontsize=7, fontfamily="Arial"):
    mpl.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": fontsize,
        "font.sans-serif": [fontfamily, "DejaVu Sans", "Helvetica", "Arial"],
        "font.family": "sans-serif",
        "axes.titlesize": fontsize,
        "axes.labelsize": fontsize,
        "axes.linewidth": 0.6,
        "xtick.labelsize": fontsize-1,
        "ytick.labelsize": fontsize-1,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.frameon": False,
        "legend.fontsize": fontsize-1,
        "pdf.fonttype": 42,  # editable text in Illustrator
        "ps.fonttype": 42,
    })

def strip_axes(ax):
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Gamma', type=float, default=1.0, help='Circulation')
    ap.add_argument('--a',     type=float, default=1.0, help='Core radius')
    ap.add_argument('--rho',   type=float, default=1.0, help='Density')
    ap.add_argument('--extent', type=float, default=2.5, help='Half-width of the domain')
    ap.add_argument('--N', type=int, default=200, help='Grid size per axis')
    ap.add_argument('--out', type=str, default='fig_rankine_inset', help='Output basename (no extension)')
    # Inset/Science style controls
    ap.add_argument('--width', type=float, default=3.6, help='Figure width in inches (small inset)')
    ap.add_argument('--height', type=float, default=1.9, help='Figure height in inches (small inset)')
    ap.add_argument('--fontsize', type=float, default=7.0, help='Base font size (pt)')
    ap.add_argument('--panel_weight', type=str, default='bold', choices=['regular','bold'], help='Panel letter weight')
    ap.add_argument('--mono', action='store_true', help='Use grayscale-safe styling')
    ap.add_argument('--transparent', action='store_true', help='Transparent figure background')
    args = ap.parse_args()

    set_science_rc(fontsize=args.fontsize)

    # Domain
    x = np.linspace(-args.extent, args.extent, args.N)
    y = np.linspace(-args.extent, args.extent, args.N)
    X, Y = np.meshgrid(x, y)

    # Fields
    U, V = rankine_velocity(args.Gamma, args.a, X, Y)
    p = rankine_pressure(args.Gamma, args.a, args.rho, X, Y)
    speed = np.hypot(U, V)

    # Trajectory
    tx, ty = make_trajectory()

    # Colormaps (grayscale-friendly by default)
    if args.mono:
        cmap_vel = "gray"
        cmap_p = "gray"
        streamline_color = "k"
        traj_color = "k"
        core_edge = "k"
        start_c = "k"
        end_c = "k"
        arrow_c = "k"
    else:
        # Perceptually uniform, colorblind-friendly
        cmap_vel = "viridis"
        cmap_p = "magma"
        streamline_color = None  # use speed colormap
        traj_color = "white"     # contrasts on pressure map
        core_edge = "white"
        start_c = "white"
        end_c = "white"
        arrow_c = "white"

    fig, axs = plt.subplots(1, 2, figsize=(args.width, args.height))
    # Tight spacing for inset feel
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.06, wspace=0.08)

    # --- Panel A: Velocity streamlines ---
    # Slightly heavier linewidth for downscaling
    lw_stream = 0.9
    density = 1.6 if args.N >= 160 else 1.2
    if streamline_color is None:
        strm = axs[0].streamplot(X, Y, U, V, color=speed, cmap=cmap_vel,
                                 density=density, linewidth=lw_stream, arrowsize=0.8)
    else:
        strm = axs[0].streamplot(X, Y, U, V, color=streamline_color,
                                 density=density, linewidth=lw_stream, arrowsize=0.8)

    # Trajectory and core
    axs[0].plot(tx, ty, '-', color='k' if args.mono else 'k', lw=1.2)
    axs[0].plot(tx[0], ty[0], marker='o', ms=3.5, mfc='none', mec='k')
    axs[0].plot(tx[-1], ty[-1], marker='o', ms=3.5, mfc='k', mec='k')
    axs[0].add_patch(plt.Circle((0, 0), args.a, fill=False, lw=1.2,
                                ec='k' if args.mono else 'crimson'))
    axs[0].set_aspect('equal')
    axs[0].set_xlim(-args.extent, args.extent)
    axs[0].set_ylim(-args.extent, args.extent)
    strip_axes(axs[0])
    axs[0].text(0.01, 0.98, "A", transform=axs[0].transAxes,
                va='top', ha='left', weight=args.panel_weight)

    # --- Panel B: Pressure field + inward ∇p arrows ---
    # Fewer levels = cleaner at small size
    cs = axs[1].contourf(X, Y, p, levels=24, cmap=cmap_p, antialiased=True)
    # Trajectory on top
    axs[1].plot(tx, ty, '-', color=traj_color if not args.mono else 'k', lw=1.2)
    axs[1].plot(tx[0], ty[0], marker='o', ms=3.5,
                mfc='none', mec=start_c if not args.mono else 'k')
    axs[1].plot(tx[-1], ty[-1], marker='o', ms=3.5,
                mfc=end_c if not args.mono else 'k',
                mec=end_c if not args.mono else 'k')
    axs[1].add_patch(plt.Circle((0, 0), args.a, fill=False, lw=1.2,
                                ec=core_edge if not args.mono else 'k'))

    # Inward arrows ~ -∇p ≈ -r̂ (schematic)
    r = np.sqrt(X**2 + Y**2) + 1e-12
    step = max(args.N // 24, 6)  # sparse for clarity
    axs[1].quiver(X[::step, ::step], Y[::step, ::step],
                  -X[::step, ::step]/r[::step, ::step],
                  -Y[::step, ::step]/r[::step, ::step],
                  color=arrow_c if not args.mono else 'k',
                  alpha=0.8, scale=22, width=0.004, headwidth=3, headlength=4)

    axs[1].set_aspect('equal')
    axs[1].set_xlim(-args.extent, args.extent)
    axs[1].set_ylim(-args.extent, args.extent)
    strip_axes(axs[1])
    axs[1].text(0.01, 0.98, "B", transform=axs[1].transAxes,
                va='top', ha='left', weight=args.panel_weight)

    # Minimal colorbar only if not monochrome; small tick labels
    if not args.mono:
        cbar = fig.colorbar(cs, ax=axs[1], fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=args.fontsize-2)
        cbar.set_label("Pressure (arb.)", fontsize=args.fontsize-1)

    # Save tight & (optionally) transparent for insets
    png_kws = dict(bbox_inches="tight", pad_inches=0.01, transparent=args.transparent or True)
    svg_kws = dict(bbox_inches="tight", pad_inches=0.01, transparent=args.transparent or True)
    plt.savefig(f"{args.out}.png", **png_kws)
    plt.savefig(f"{args.out}.svg", **svg_kws)
    print(f"Saved {args.out}.png and {args.out}.svg")

if __name__ == "__main__":
    main()
