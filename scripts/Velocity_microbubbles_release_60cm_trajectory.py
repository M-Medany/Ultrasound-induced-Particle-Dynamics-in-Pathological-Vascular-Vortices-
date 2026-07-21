# run_mb_normalized60_fixedstep.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from pathlib import Path as _Path
REPO_ROOT = _Path(__file__).resolve().parents[1]


# ===================== USER CONFIG =====================
CSV_PATH = REPO_ROOT / "data" / "comsol" / "Normalized_60_cm.csv"

rho   = 1000.0
CD    = 1000000000000
Gamma = 0.00002            # circulation used only in Rankine pressure
a_override = 50e-6      # e.g. 8.7e-05 (meters) to force core radius
CENTER_OVERRIDE = (1.2e-4, 1.531e-4) # e.g. (2.505e-4, 1.731e-4) to force center


# Initial MB position (you can type microns here; auto-converted to meters)
x0_um, y0_um = 70.0, 150.0
u0, v0 = 0.0, 0.0

# Time control for fixed-step RK4
t0, t1 = 0.0, 2_000_000.0         # total simulated "time"
N_STEPS = 2000                    # total RK4 steps (dt = (t1-t0)/N_STEPS)
PROGRESS_EVERY = 100              # print every N steps

# Plot/frames
output_dir     = "output_images"
frame_interval = 20
skip_interval  = 1
mask_radius_um = 10.0
quiver_scale   = 50

# ============== Globals (single-process) ==============
TREE=None; UVAL=None; VVAL=None
XC=YC=A=GAM=RHO=CDG=None
XMIN=XMAX=YMIN=YMAX=None

# ----------------- helpers -----------------
def um_to_m(x_um, y_um): return x_um*1e-6, y_um*1e-6

def estimate_vortex_center(df):
    X = df["x"].to_numpy(float); Y = df["y"].to_numpy(float)
    U = df["u"].to_numpy(float); V = df["v"].to_numpy(float)
    spd = np.hypot(U, V)
    xlo, xhi = np.percentile(X, [5, 95])
    ylo, yhi = np.percentile(Y, [5, 95])
    slo = np.percentile(spd, 30)
    m = (X>=xlo)&(X<=xhi)&(Y>=ylo)&(Y<=yhi)&(spd>slo)
    if m.sum() < 200:
        j = int(np.argmin(spd)); return float(X[j]), float(Y[j])
    X= X[m]; Y= Y[m]; U= U[m]; V= V[m]
    Um = np.hypot(U,V)+1e-12; ux,uy = U/Um, V/Um
    xs = np.linspace(np.percentile(X,10), np.percentile(X,90), 60)
    ys = np.linspace(np.percentile(Y,10), np.percentile(Y,90), 60)
    R0 = 0.35*min(X.ptp(), Y.ptp())
    best_cost = 1e99; best_xy = (np.median(X), np.median(Y))
    for xc in xs:
        dx = X - xc
        for yc in ys:
            dy = Y - yc
            r = np.hypot(dx,dy)+1e-12
            rx,ry = dx/r, dy/r; tx,ty = -ry, rx
            ur = ux*rx + uy*ry
            ut = ux*tx + uy*ty
            w  = np.exp(-(r*r)/(2*R0*R0))
            cost = (w*(ur*ur)).sum()/w.sum() + 0.2*(w*(1.0-np.abs(ut))).sum()/w.sum()
            if cost < best_cost:
                best_cost, best_xy = cost, (xc,yc)
    return float(best_xy[0]), float(best_xy[1])

def kNN_velocity_field(x, y, k=8, eps=1e-12):
    xq = float(np.clip(x, XMIN, XMAX))
    yq = float(np.clip(y, YMIN, YMAX))
    d, idx = TREE.query((xq, yq), k=k)
    if np.isscalar(d): d=np.array([d]); idx=np.array([idx])
    w = 1.0/(d + eps); w /= w.sum()
    u = float(np.dot(w, UVAL[idx])); v = float(np.dot(w, VVAL[idx]))
    return np.array([u, v], dtype=float)

# Rankine pressure gradient
def dp_dr_rankine(r, Gamma, rho, a):
    if r < a:  return (rho*Gamma**2)/(4*np.pi**2) * (r/(a*a))
    else:      return (rho*Gamma**2)/(4*np.pi**2) * (1.0/(r**3))

def grad_p_rankine(x, y, xc, yc, Gamma, rho, a, eps=1e-15):
    xr, yr = x-xc, y-yc
    r = np.hypot(xr, yr) + eps
    dpr = dp_dr_rankine(r, Gamma, rho, a)
    return np.array([dpr * xr/r, dpr * yr/r], dtype=float)

def rhs(state_vec):
    # state_vec = [x, y, uMBx, uMBy]
    x, y, uMBx, uMBy = [float(v) for v in state_vec]
    uf = kNN_velocity_field(x, y, k=8)
    gradp = grad_p_rankine(x, y, XC, YC, GAM, RHO, A)
    du = uf - np.array([uMBx, uMBy], dtype=float)
    du_mag = float(np.hypot(du[0], du[1]))
    ax = -(3.0/RHO)*gradp[0] + 0.75*CDG*du[0]*du_mag
    ay = -(3.0/RHO)*gradp[1] + 0.75*CDG*du[1]*du_mag
    return np.array([uMBx, uMBy, ax, ay], dtype=float)

def rk4_step(y, dt):
    k1 = rhs(y)
    k2 = rhs(y + 0.5*dt*k1)
    k3 = rhs(y + 0.5*dt*k2)
    k4 = rhs(y + dt*k3)
    yn = y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    # Stop if outside bounds
    if (yn[0] < XMIN) or (yn[0] > XMAX) or (yn[1] < YMIN) or (yn[1] > YMAX):
        yn[0] = float(np.clip(yn[0], XMIN, XMAX))
        yn[1] = float(np.clip(yn[1], YMIN, YMAX))
        return yn, True
    return yn, False

# ----------------- main -----------------
def main():
    print("Reading velocity data from CSV...")
    vf = pd.read_csv(CSV_PATH)
    vf.columns = vf.columns.str.strip()
    for col in ("x","y","u","v"):
        if col not in vf.columns:
            raise ValueError("CSV must have x,y,u,v")

    # --- CSV arrays (use distinct names to avoid collisions) ---
    Xc = vf["x"].to_numpy(float)
    Yc = vf["y"].to_numpy(float)
    Uc = vf["u"].to_numpy(float)
    Vc = vf["v"].to_numpy(float)

    global XMIN,XMAX,YMIN,YMAX,TREE,UVAL,VVAL,XC,YC,A,GAM,RHO,CDG
    XMIN,XMAX = float(Xc.min()), float(Xc.max())
    YMIN,YMAX = float(Yc.min()), float(Yc.max())
    print(f"Loaded {len(vf)} points | domain x:[{XMIN},{XMAX}], y:[{YMIN},{YMAX}]")

    # Build KDTree for interpolation
    TREE = cKDTree(np.column_stack([Xc, Yc])); UVAL = Uc; VVAL = Vc

    # Initial in meters
    x0, y0 = um_to_m(x0_um, y0_um)
    if not (XMIN <= x0 <= XMAX and YMIN <= y0 <= YMAX):
        raise ValueError(f"Initial (x0,y0)=({x0},{y0}) outside CSV domain.")

    # Center + core radius
    print("Estimating vortex center ...")
    xc, yc = estimate_vortex_center(vf)
    if CENTER_OVERRIDE is not None:
        xc, yc = CENTER_OVERRIDE
        print("CENTER_OVERRIDE applied.")
    print(f"Center used: (xc,yc)=({xc:.6g},{yc:.6g})")

    if a_override is None:
        r = np.hypot(Xc - xc, Yc - yc)
        s = np.hypot(Uc, Vc)
        try:
            rlo, rhi = np.percentile(r, [30, 95])
            band = (r >= rlo) & (r <= rhi)
            r_peak = r[band][s[band].argmax()] if band.any() else np.median(r)
        except Exception:
            r_peak = np.median(r)
        A_local = float(r_peak)
        print(f"Core radius a (auto): {A_local:.6g} m")
    else:
        A_local = float(a_override)
        print(f"Core radius a (override): {A_local:.6g} m")

    XC, YC, A, GAM, RHO, CDG = xc, yc, A_local, Gamma, rho, CD

    # ------------ Fixed-step RK4 integration ------------
    dt = (t1 - t0) / float(N_STEPS)
    state = np.array([x0, y0, u0, v0], dtype=float)
    times = np.empty(N_STEPS+1, dtype=float); times[0] = t0
    traj  = np.empty((4, N_STEPS+1), dtype=float); traj[:,0] = state

    print(f"Integrating with fixed-step RK4: N_STEPS={N_STEPS}, dt={dt:.1f} s")
    hit_boundary = False
    for n in range(1, N_STEPS+1):
        state, hit = rk4_step(state, dt)
        times[n] = times[n-1] + dt
        traj[:, n] = state
        if (n % PROGRESS_EVERY) == 0:
            r_now = float(np.hypot(state[0]-XC, state[1]-YC))
            print(f"  step {n}/{N_STEPS}  t={times[n]:.0f}s  r_to_center={r_now*1e6:.1f} µm")
        if hit:
            print(f"  boundary reached at step {n}, t={times[n]:.0f}s")
            hit_boundary = True
            traj  = traj[:, :n+1]
            times = times[:n+1]
            break

    # ----------------- plots (µm, normalized speed) -----------------
    os.makedirs(output_dir, exist_ok=True)

    def save_frames():
        Xu, Yu = Xc*1e6, Yc*1e6
        # Normalized dataset — use magnitude directly for colors
        Xs, Ys = Xu[::skip_interval], Yu[::skip_interval]
        Us, Vs = Uc[::skip_interval], Vc[::skip_interval]
        mags = np.hypot(Us,Vs); mags = np.where(mags==0, 1e-12, mags)
        Un, Vn = Us/mags, Vs/mags
        norm = plt.Normalize(mags.min(), mags.max())
        colors = plt.cm.viridis(norm(mags))

        tx = traj[0]*1e6; ty = traj[1]*1e6
        print(f"Exporting frames every {frame_interval} steps to: {os.path.abspath(output_dir)}")
        for i in range(0, tx.size, frame_interval):
            d = np.hypot(Xs - tx[i], Ys - ty[i])
            mask = d > mask_radius_um
            fig, ax = plt.subplots(figsize=(10,5))
            ax.quiver(Xs[mask], Ys[mask], Un[mask], Vn[mask],
                      color=colors[mask], scale=quiver_scale, alpha=0.9)
            cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
            cbar.set_label('Normalized speed (arb.)')
            ax.plot(tx[:i+1], ty[:i+1], 'k-', lw=2, label='trajectory')
            ax.plot(tx[0], ty[0], 'go', ms=7, label='start')
            ax.plot(tx[i], ty[i], 'ro', ms=6)
            core = plt.Circle((XC*1e6, YC*1e6), A*1e6, edgecolor='r', facecolor='none', lw=2)
            ax.add_patch(core)
            ax.set_title('Velocity Vector Plot with MB trajectory')
            ax.set_xlabel('X (µm)'); ax.set_ylabel('Y (µm)')
            ax.set_aspect('equal','box'); ax.grid(alpha=0.3)
            ax.set_xlim(Xu.min(), Xu.max()); ax.set_ylim(Yu.min(), Yu.max())
            if i == 0: ax.legend(loc='upper right')
            plt.savefig(os.path.join(output_dir, f"frame_{i:04d}.png"), dpi=220)
            plt.close(fig)

    def save_static():
        Xu, Yu = Xc*1e6, Yc*1e6
        Xs, Ys = Xu[::skip_interval], Yu[::skip_interval]
        Us, Vs = Uc[::skip_interval], Vc[::skip_interval]
        mags = np.hypot(Us,Vs); mags = np.where(mags==0, 1e-12, mags)
        Un, Vn = Us/mags, Vs/mags
        norm = plt.Normalize(mags.min(), mags.max())
        colors = plt.cm.viridis(norm(mags))

        fig, ax = plt.subplots(figsize=(10,5))
        ax.quiver(Xs, Ys, Un, Vn, color=colors, scale=quiver_scale, alpha=0.9)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
        cbar.set_label('Normalized speed (arb.)')
        tx = traj[0]*1e6; ty = traj[1]*1e6
        ax.plot(tx, ty, 'k-', lw=2)
        ax.plot(tx[0], ty[0], 'go', ms=7)
        ax.plot(tx[-1], ty[-1], 'ro', ms=7)
        core = plt.Circle((XC*1e6, YC*1e6), A*1e6, edgecolor='r', facecolor='none', lw=2)
        ax.add_patch(core)
        ax.set_title('Velocity Vector Plot with MB trajectory (static)')
        ax.set_xlabel('X (µm)'); ax.set_ylabel('Y (µm)')
        ax.set_aspect('equal','box'); ax.grid(alpha=0.3)
        ax.set_xlim(Xu.min(), Xu.max()); ax.set_ylim(Yu.min(), Yu.max())
        out = os.path.join(output_dir, "velocity_Trajectory_plot.png")
        plt.savefig(out, dpi=300); plt.close(fig)
        print(f"Saved: {out}")

    save_frames()
    save_static()
    if hit_boundary:
        print("Stopped early due to boundary; consider moving x0,y0 or lowering dt.")
    print("Done. (Make video with ffmpeg if you like.)")

if __name__ == "__main__":
    main()
