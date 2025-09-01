# spawn_safe_mb_two.py — two microbubbles, robust center + core radius
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
from multiprocessing import get_context, cpu_count
from PIL import ImageFile

# ---------------- Pillow/PNG robustness (Windows) ----------------
# Bump encoder buffer to avoid _idat/fileno issues when saving many PNGs
ImageFile.MAXBLOCK = 1 << 24  # 16 MB

# ===================== USER CONFIG =====================
CSV_PATH = r"C:\Users\M4\VSCode_Projects\Ultrasound-Swarm-Microbubbles-Navigating-Vortices-to-Target-and-Fill-Aneurysms\Excel_data_velocity_comsol\Velocity_2d_5cm.csv"

rho = 1000.0
CD  = 5
Gamma = 0.95          # circulation (in your CSV units)
a_override = 15       # set None to auto-detect core radius from peak speed
CENTER_OVERRIDE = None  # e.g. (3835.0, 335.0)

# Initial states (must be inside CSV domain) — CSV UNITS
x0_1, y0_1 = 3930.0, 150.0
u0_1, v0_1 = 0.0, 0.0

x0_2, y0_2 = 4050.0, 190.0
u0_2, v0_2 = 0.0, 0.0

# Time
t_span = (0.0, 2_000_000.0)
rtol, atol = 1e-6, 1e-9
chunk_steps_target = 400  # ~steps per chunk

# Output frames (for video)
output_dir = "output_images"
frame_interval = 20
skip_interval  = 10
mask_radius    = 10.0     # CSV units (white halo around MBs)
quiver_scale   = 30

# Colors & styles
TRAJ1_COLOR = 'r'       # MB1: red
TRAJ2_COLOR = 'b'       # MB2: blue
CORE_EDGE_COLOR = 'k'   # core circle: black
PATH_LW = 3.5           # thicker paths

# ---- Units: show cm/s on colorbar (CSV is mm/s) ----
CSV_VEL_UNITS = 'm/s'     # what your CSV velocities are in
DISPLAY_VEL_UNITS = 'cm/s' # what you want to display
VMAG_SCALE = 100 if (CSV_VEL_UNITS == 'm/s' and DISPLAY_VEL_UNITS == 'cm/s') else 1.0
CBAR_LABEL = f'Velocity Magnitude ({DISPLAY_VEL_UNITS})'

# ---- Trajectory-only export (no background) ----
DRAW_CORE_CIRCLE = True      # keep black circle
HIDE_AXES = True             # hide ticks & spines for a clean look
TRAJ_ONLY_FIGSIZE = (4, 4)   # smaller, compact
TRAJ_ONLY_DPI = 600          # crisp output
TRAJ_BG_TRANSPARENT = True   # transparent background (set False for white)
START_MARKER_COLOR = 'g'     # green start markers like your panel
CURR_MARKER_SIZE = 7

# ---- Plot look & resolution ----
FIGSIZE = (10, 5)
DPI_FRAME = 400     # per-frame PNG for video (try 300–400)
DPI_STATIC = 600    # high-res static figure
TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 18
TICK_FONTSIZE  = 14
LEGEND_FONTSIZE = 12

# Robustness toggles
SAVE_JPG_FRAMES = False      # set True to avoid PNG path entirely
CLEAR_OUTPUT_DIR = False     # set True to wipe old frames before writing

# ============== GLOBALS (initialized in workers) ==============
TREE = None
UVAL = None
VVAL = None
XC = None
YC = None
A = None
GAM = None
RHO = None
CDG = None
RTOL = None
ATOL = None
CHUNK_STEPS_TARGET = None
XMIN = None
XMAX = None
YMIN = None
YMAX = None

# ----------------- Physics helpers (top-level) -----------------
def estimate_vortex_center(df):
    """
    Robust center finder for ANY units.
    Trims by percentiles and minimizes radial component of velocity.
    """
    X = df["x"].to_numpy(float)
    Y = df["y"].to_numpy(float)
    U = df["u"].to_numpy(float)
    V = df["v"].to_numpy(float)

    spd = np.hypot(U, V)
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
            rx, ry = dx/r, dy/r           # radial unit
            tx, ty = -ry, rx              # tangential unit
            ur = ux*rx + uy*ry            # radial alignment
            ut = ux*tx + uy*ty            # tangential alignment
            w  = np.exp(-(r*r)/(2*R0*R0)) # Gaussian window
            cost = (w*(ur*ur)).sum()/w.sum() + 0.2*(w*(1.0 - np.abs(ut))).sum()/w.sum()
            if cost < best_cost:
                best_cost, best_xy = cost, (xc, yc)

    return float(best_xy[0]), float(best_xy[1])

def kNN_velocity_field(x, y, k=8, eps=1e-12):
    d, idx = TREE.query((x, y), k=k)
    if np.isscalar(d):
        d   = np.array([d]); idx = np.array([idx])
    w = 1.0 / (d + eps); w /= w.sum()
    u = np.dot(w, UVAL[idx]); v = np.dot(w, VVAL[idx])
    return np.array([u, v])

def dp_dr_rankine(r, Gamma, rho, a):
    if r < a:
        return (rho * Gamma**2) / (4*np.pi**2) * (r / (a*a))
    else:
        return (rho * Gamma**2) / (4*np.pi**2) * (1.0 / (r**3))

def grad_p_rankine(x, y, xc, yc, Gamma, rho, a, eps=1e-15):
    xr, yr = x - xc, y - yc
    r = np.hypot(xr, yr) + eps
    dpr = dp_dr_rankine(r, Gamma, rho, a)
    return dpr * np.array([xr/r, yr/r])

# ------------------------ Dynamics -----------------------------
def rhs(t, Y):
    # unpack both microbubbles
    x1, y1, u1, v1, x2, y2, u2, v2 = Y

    # MB1
    uf1 = kNN_velocity_field(x1, y1, k=8)
    gradp1 = grad_p_rankine(x1, y1, XC, YC, GAM, RHO, A)
    du1 = uf1 - np.array([u1, v1])
    du1_mag = np.hypot(du1[0], du1[1])
    ax1 = -(3.0/RHO) * gradp1[0] + 0.75 * CDG * du1[0] * du1_mag
    ay1 = -(3.0/RHO) * gradp1[1] + 0.75 * CDG * du1[1] * du1_mag

    # MB2
    uf2 = kNN_velocity_field(x2, y2, k=8)
    gradp2 = grad_p_rankine(x2, y2, XC, YC, GAM, RHO, A)
    du2 = uf2 - np.array([u2, v2])
    du2_mag = np.hypot(du2[0], du2[1])
    ax2 = -(3.0/RHO) * gradp2[0] + 0.75 * CDG * du2[0] * du2_mag
    ay2 = -(3.0/RHO) * gradp2[1] + 0.75 * CDG * du2[1] * du2_mag

    return [u1, v1, ax1, ay1, u2, v2, ax2, ay2]

def out_of_bounds(t, Y):
    x1, y1 = Y[0], Y[1]
    x2, y2 = Y[4], Y[5]
    return min(
        x1 - XMIN, XMAX - x1, y1 - YMIN, YMAX - y1,
        x2 - XMIN, XMAX - x2, y2 - YMIN, YMAX - y2
    )
out_of_bounds.terminal = True
out_of_bounds.direction = -1

def solve_ode_chunk(t0, t1, Yinit):
    span = t1 - t0
    max_step = max(1e-9, span / CHUNK_STEPS_TARGET)
    sol = solve_ivp(
        rhs, [t0, t1], Yinit, method="RK45",
        rtol=RTOL, atol=ATOL, max_step=max_step,
        dense_output=True, events=out_of_bounds
    )
    return sol.t, sol.y

# ------------- Worker initializer (runs in each child) ----------
def init_worker(points, u_values, v_values, xc, yc, a, gamma, rho, cd,
                rtol, atol, chunk_steps_target,
                xmin, xmax, ymin, ymax):
    global TREE, UVAL, VVAL, XC, YC, A, GAM, RHO, CDG, RTOL, ATOL, CHUNK_STEPS_TARGET
    global XMIN, XMAX, YMIN, YMAX
    TREE = cKDTree(points)
    UVAL = u_values
    VVAL = v_values
    XC, YC = xc, yc
    A, GAM, RHO, CDG = a, gamma, rho, cd
    RTOL, ATOL = rtol, atol
    CHUNK_STEPS_TARGET = chunk_steps_target
    XMIN, XMAX, YMIN, YMAX = xmin, xmax, ymin, ymax

# ============================== MAIN ===========================
def main():
    print(f"Reading velocity data ...\nCSV: {CSV_PATH}")
    vf = pd.read_csv(CSV_PATH)
    vf.columns = vf.columns.str.strip()
    assert {'x','y','u','v'} <= set(vf.columns), "CSV must contain x,y,u,v"

    points = vf[['x','y']].to_numpy(float)
    u_values = vf['u'].to_numpy(float)
    v_values = vf['v'].to_numpy(float)

    xmin, xmax = points[:,0].min(), points[:,0].max()
    ymin, ymax = points[:,1].min(), points[:,1].max()
    print(f"Domain x:[{xmin},{xmax}], y:[{ymin},{ymax}]  (CSV units)")

    # Validate both initial positions
    if not (xmin <= x0_1 <= xmax and ymin <= y0_1 <= ymax):
        raise ValueError(f"Initial MB1 (x0,y0)=({x0_1},{y0_1}) outside CSV domain.")
    if not (xmin <= x0_2 <= xmax and ymin <= y0_2 <= ymax):
        raise ValueError(f"Initial MB2 (x0,y0)=({x0_2},{y0_2}) outside CSV domain.")

    print("Estimating vortex center ...")
    xc, yc = estimate_vortex_center(vf)
    if CENTER_OVERRIDE is not None:
        xc, yc = CENTER_OVERRIDE
        print("CENTER_OVERRIDE applied.")
    print(f"Center (used): (xc,yc)=({xc:.3f},{yc:.3f})")

    # Sanity: if center is outside domain, fall back
    if not (xmin <= xc <= xmax and ymin <= yc <= ymax):
        j = int(np.argmin(np.hypot(u_values, v_values)))
        xc, yc = float(points[j,0]), float(points[j,1])  # <-- fixed bracket
        print(f"Center fallback to slowest point: (xc,yc)=({xc:.3f},{yc:.3f})")

    # Core radius
    if a_override is None:
        r = np.hypot(vf["x"].to_numpy(float) - xc, vf["y"].to_numpy(float) - yc)
        s = np.hypot(u_values, v_values)
        rlo, rhi = np.percentile(r, [30, 95])
        band = (r >= rlo) & (r <= rhi)
        if band.sum() > 0:
            r_peak = r[band][s[band].argmax()]
        else:
            r_peak = np.median(r)
        A_local = float(r_peak)
        print(f"Core radius a from peak speed: {A_local:.3f}")
    else:
        A_local = float(a_override)
        print(f"Core radius a (override): {A_local:.3f}")

    # Output dir housekeeping
    os.makedirs(output_dir, exist_ok=True)
    if CLEAR_OUTPUT_DIR:
        for fn in os.listdir(output_dir):
            fp = os.path.join(output_dir, fn)
            try:
                if os.path.isfile(fp):
                    os.remove(fp)
                elif os.path.isdir(fp):
                    shutil.rmtree(fp)
            except Exception as e:
                print(f"Warning: could not delete {fp}: {e}")

    # Build chunks
    num_chunks = cpu_count()
    edges = np.linspace(t_span[0], t_span[1], num_chunks + 1)
    intervals = [(edges[i], edges[i+1]) for i in range(num_chunks)]

    # Spawn-safe pool and initializer
    ctx = get_context("spawn")
    results = []
    with ctx.Pool(processes=num_chunks,
                  initializer=init_worker,
                  initargs=(points, u_values, v_values, xc, yc, A_local,
                            Gamma, rho, CD, rtol, atol, chunk_steps_target,
                            xmin, xmax, ymin, ymax)) as pool:
        # first chunk
        Yprev = [x0_1, y0_1, u0_1, v0_1, x0_2, y0_2, u0_2, v0_2]
        results.append(pool.apply_async(solve_ode_chunk,
                        args=(intervals[0][0], intervals[0][1], Yprev)))
        # chain chunks
        for i in range(1, num_chunks):
            t, y = results[i-1].get()
            print(f"Chunk {i}/{num_chunks} done, t∈[{intervals[i-1][0]}, {intervals[i-1][1]}]")
            Yprev = [y[0,-1], y[1,-1], y[2,-1], y[3,-1],
                     y[4,-1], y[5,-1], y[6,-1], y[7,-1]]
            results.append(pool.apply_async(solve_ode_chunk,
                            args=(intervals[i][0], intervals[i][1], Yprev)))

        final_t = []
        final_y_list = []
        for i, r in enumerate(results):
            t, y = r.get()
            final_t.extend(t)
            final_y_list.append(y)
            print(f"Collected chunk {i+1}/{num_chunks}")

    final_y = np.concatenate(final_y_list, axis=1)  # rows: [x1,y1,u1,v1,x2,y2,u2,v2]

    # Save trajectory CSV
    traj_out = "trajectory_two_mb.csv"
    traj_df = pd.DataFrame({
        "x1": final_y[0], "y1": final_y[1], "uMBx1": final_y[2], "uMBy1": final_y[3],
        "x2": final_y[4], "y2": final_y[5], "uMBx2": final_y[6], "uMBy2": final_y[7],
    })
    traj_df.to_csv(traj_out, index=False)
    print(f"Saved trajectory: {traj_out}")



    # ============== Frames (two MBs) ==============
    def save_frames():
        X = vf['x'].to_numpy(float); Y = vf['y'].to_numpy(float)
        U = vf['u'].to_numpy(float); V = vf['v'].to_numpy(float)
        Xs = X[::skip_interval]; Ys = Y[::skip_interval]
        Us = U[::skip_interval]; Vs = V[::skip_interval]

        # magnitude in CSV units, then convert for display (cm/s)
        mags_raw = np.hypot(Us, Vs)
        mags_raw = np.where(mags_raw == 0, 1e-12, mags_raw)
        Un = Us / mags_raw
        Vn = Vs / mags_raw

        mags_disp = mags_raw * VMAG_SCALE
        norm = plt.Normalize(mags_disp.min(), mags_disp.max())
        colors = plt.cm.viridis(norm(mags_disp))

        tx1, ty1 = final_y[0], final_y[1]
        tx2, ty2 = final_y[4], final_y[5]
        n = tx1.size

        for i in range(0, n, frame_interval):
            d1 = np.hypot(Xs - tx1[i], Ys - ty1[i])
            d2 = np.hypot(Xs - tx2[i], Ys - ty2[i])
            mask = (d1 > mask_radius) & (d2 > mask_radius)

            fig, ax = plt.subplots(figsize=FIGSIZE)
            ax.quiver(Xs[mask], Ys[mask], Un[mask], Vn[mask],
                      color=colors[mask], scale=quiver_scale, alpha=0.9)

            cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
            cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
            cbar.set_label(CBAR_LABEL, fontsize=LABEL_FONTSIZE)

            # thicker, fixed-color paths
            ax.plot(tx1[:i+1], ty1[:i+1], lw=PATH_LW, color=TRAJ1_COLOR, label='MB1 path')
            ax.plot(tx2[:i+1], ty2[:i+1], lw=PATH_LW, color=TRAJ2_COLOR, label='MB2 path')

            # starts
            ax.plot(tx1[0], ty1[0], marker='o', ms=7, linestyle='None', color=TRAJ1_COLOR, label='start MB1')
            ax.plot(tx2[0], ty2[0], marker='o', ms=7, linestyle='None', color=TRAJ2_COLOR, label='start MB2')

            # current positions (hollow)
            ax.plot(tx1[i], ty1[i], marker='o', linestyle='None',
                    markerfacecolor='none', markeredgecolor=TRAJ1_COLOR, ms=7)
            ax.plot(tx2[i], ty2[i], marker='o', linestyle='None',
                    markerfacecolor='none', markeredgecolor=TRAJ2_COLOR, ms=7)

            # black core circle, no label
            core = plt.Circle((xc, yc), A_local, edgecolor=CORE_EDGE_COLOR, facecolor='none', lw=2)
            ax.add_patch(core)

            ax.set_title('Velocity field + Two MB trajectories', fontsize=TITLE_FONTSIZE)
            ax.set_xlabel('x (CSV units)', fontsize=LABEL_FONTSIZE)
            ax.set_ylabel('y (CSV units)', fontsize=LABEL_FONTSIZE)
            ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
            ax.set_aspect('equal', 'box'); ax.grid(alpha=0.3)
            ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
            if i == 0:
                ax.legend(loc='upper right', ncol=2, prop={'size': LEGEND_FONTSIZE})

            fig.tight_layout()

            try:
                if SAVE_JPG_FRAMES:
                    out_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
                    fig.savefig(out_path, format="jpg", dpi=DPI_FRAME, bbox_inches="tight")
                else:
                    out_path = os.path.join(output_dir, f"frame_{i:04d}.png")
                    fig.savefig(out_path, format="png", dpi=DPI_FRAME, bbox_inches="tight",
                                pil_kwargs={"compress_level": 1})
            except Exception as e:
                # Fallback to JPG if PNG write trips on Pillow
                fallback_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
                print(f"PNG save failed at frame {i} ({e}). Falling back to JPG: {fallback_path}")
                fig.savefig(fallback_path, format="jpg", dpi=DPI_FRAME, bbox_inches="tight")
            finally:
                plt.close(fig)

        print(f"Frames saved to: {os.path.abspath(output_dir)}")
        print("Make video:\n  ffmpeg -framerate 30 -i frame_%04d." + ("jpg" if SAVE_JPG_FRAMES else "png") + " -c:v libx264 -pix_fmt yuv420p output_video.mp4")

    def save_static():
        X = vf['x'].to_numpy(float); Y = vf['y'].to_numpy(float)
        U = vf['u'].to_numpy(float); V = vf['v'].to_numpy(float)
        Xs = X[::skip_interval]; Ys = Y[::skip_interval]
        Us = U[::skip_interval]; Vs = V[::skip_interval]

        mags_raw = np.hypot(Us, Vs)
        mags_raw = np.where(mags_raw == 0, 1e-12, mags_raw)
        Un = Us / mags_raw
        Vn = Vs / mags_raw

        mags_disp = mags_raw * VMAG_SCALE
        norm = plt.Normalize(mags_disp.min(), mags_disp.max())
        colors = plt.cm.viridis(norm(mags_disp))

        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.quiver(Xs, Ys, Un, Vn, color=colors, scale=quiver_scale, alpha=0.9)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
        cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
        cbar.set_label(CBAR_LABEL, fontsize=LABEL_FONTSIZE)

        tx1, ty1 = final_y[0], final_y[1]
        tx2, ty2 = final_y[4], final_y[5]
        ax.plot(tx1, ty1, lw=PATH_LW, color=TRAJ1_COLOR, label='MB1 path')
        ax.plot(tx2, ty2, lw=PATH_LW, color=TRAJ2_COLOR, label='MB2 path')
        ax.plot(tx1[0],  ty1[0],  marker='o', ms=7, linestyle='None', color=TRAJ1_COLOR)
        ax.plot(tx2[0],  ty2[0],  marker='o', ms=7, linestyle='None', color=TRAJ2_COLOR)
        ax.plot(tx1[-1], ty1[-1], marker='o', linestyle='None',
                markerfacecolor='none', markeredgecolor=TRAJ1_COLOR, ms=8)
        ax.plot(tx2[-1], ty2[-1], marker='o', linestyle='None',
                markerfacecolor='none', markeredgecolor=TRAJ2_COLOR, ms=8)

        core = plt.Circle((xc, yc), A_local, edgecolor=CORE_EDGE_COLOR, facecolor='none', lw=2)
        ax.add_patch(core)

        ax.set_title('Velocity field + Two MB trajectories (static)', fontsize=TITLE_FONTSIZE)
        ax.set_xlabel('x (CSV units)', fontsize=LABEL_FONTSIZE)
        ax.set_ylabel('y (CSV units)', fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
        ax.set_aspect('equal', 'box'); ax.grid(alpha=0.3)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.legend(loc='upper right', ncol=2, prop={'size': LEGEND_FONTSIZE})

        fig.tight_layout()
        try:
            plt.savefig("velocity_Trajectory_two_mb.png", format="png",
                        dpi=DPI_STATIC, bbox_inches="tight",
                        pil_kwargs={"compress_level": 1})
        except Exception as e:
            print(f"PNG save failed for static ({e}). Saving JPG fallback.")
            plt.savefig("velocity_Trajectory_two_mb.jpg", format="jpg",
                        dpi=DPI_STATIC, bbox_inches="tight")
        finally:
            plt.close(fig)
        print("Saved: velocity_Trajectory_two_mb.(png/jpg)")

    # ---- Run the plotting ----
    save_frames()
    save_static()

# Windows/VS Code needs this guard for multiprocessing
if __name__ == "__main__":
    # Globals used in workers
    RHO = rho
    CDG = CD
    GAM = Gamma
    RTOL = rtol
    ATOL = atol
    CHUNK_STEPS_TARGET = chunk_steps_target
    main()
