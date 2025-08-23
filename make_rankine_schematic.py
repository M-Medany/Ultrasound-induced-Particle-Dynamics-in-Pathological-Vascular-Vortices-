#!/usr/bin/env python3
# make_rankine_schematic.py
# Generates a 2-panel schematic: (A) velocity streamlines of a Rankine vortex
# and (B) pressure field with inward pressure-gradient arrows.
#
# Usage:
#   python make_rankine_schematic.py --Gamma 1.0 --a 1.0 --rho 1.0 --out fig_rankine
#
# Output:
#   fig_rankine.png and fig_rankine.svg

import argparse
import numpy as np
import matplotlib.pyplot as plt

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Gamma', type=float, default=1.0, help='Circulation')
    ap.add_argument('--a',     type=float, default=1.0, help='Core radius')
    ap.add_argument('--rho',   type=float, default=1.0, help='Density')
    ap.add_argument('--extent', type=float, default=2.5, help='Half-width of the domain')
    ap.add_argument('--N', type=int, default=200, help='Grid size per axis')
    ap.add_argument('--out', type=str, default='fig_rankine', help='Output basename (no extension)')
    args = ap.parse_args()

    x = np.linspace(-args.extent, args.extent, args.N)
    y = np.linspace(-args.extent, args.extent, args.N)
    X, Y = np.meshgrid(x, y)

    U, V = rankine_velocity(args.Gamma, args.a, X, Y)
    p = rankine_pressure(args.Gamma, args.a, args.rho, X, Y)

    # Synthetic trajectory for illustration
    tx, ty = make_trajectory()

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Left: velocity streamlines
    speed = np.hypot(U, V)
    axs[0].streamplot(X, Y, U, V, color=speed, cmap='viridis', density=1.6, linewidth=1.0)
    axs[0].plot(tx, ty, 'k-', lw=2, label='MB trajectory')
    axs[0].plot(tx[0], ty[0], 'go', ms=6, label='start')
    axs[0].plot(tx[-1], ty[-1], 'ro', ms=6, label='end')
    axs[0].add_patch(plt.Circle((0, 0), args.a, color='r', fill=False, lw=2))
    axs[0].set_aspect('equal')
    axs[0].set_title('Velocity field (Rankine vortex)')
    axs[0].grid(alpha=0.25)
    axs[0].legend(loc='upper right', frameon=False)

    # Right: pressure field with inward grad p arrows
    cs = axs[1].contourf(X, Y, p, levels=40, cmap='plasma')
    axs[1].plot(tx, ty, 'w-', lw=2)
    axs[1].plot(tx[0], ty[0], 'go', ms=6)
    axs[1].plot(tx[-1], ty[-1], 'ro', ms=6)
    axs[1].add_patch(plt.Circle((0, 0), args.a, color='w', fill=False, lw=2))
    # Inward arrows ~ -∇p direction ≈ -r̂ (schematic)
    r = np.sqrt(X**2 + Y**2) + 1e-12
    axs[1].quiver(X[::8,::8], Y[::8,::8], -X[::8,::8]/r[::8,::8], -Y[::8,::8]/r[::8,::8],
                  color='white', alpha=0.7, scale=20, width=0.004)
    axs[1].set_aspect('equal')
    axs[1].set_title('Pressure field (low at core)')
    axs[1].grid(alpha=0.25)
    cbar = fig.colorbar(cs, ax=axs[1], label='Pressure (arb.)')

    plt.savefig(f'{args.out}.png', dpi=300)
    plt.savefig(f'{args.out}.svg')
    print(f'Saved {args.out}.png and {args.out}.svg')

if __name__ == '__main__':
    main()
