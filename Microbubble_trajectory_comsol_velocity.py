# spawn_safe_mb_two.py — two microbubbles, trajectory-only export
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
ImageFile.MAXBLOCK = 1 << 24  # 16 MB buffer for PNG encoder

# ===================== USER CONFIG =====================
CSV_PATH = r"C:\Users\M4\VSCode_Projects\Ultrasound-Swarm-Microbubbles-Navigating-Vortices-to-Target-and-Fill-Aneurysms\Excel_data_velocity_comsol\Velocity_2d_5cm.csv"

rho   = 1000.0
CD    = 5.0
Gamma = 0.95           # circulation (CSV units)
a_override = 15.0      # set None to auto-detect core radius
CENTER_OVERRIDE = None # e.g. (3835.0, 335.0)

# Initial states (must be inside CSV domain) — CSV UNITS
x0_1, y0_1 = 3930.0, 150.0; u0_1, v0_1 = 0.0, 0.0
x0_2, y0_2 = 4050.0, 190.0; u0_2, v0_2 = 0.0, 0.0

# Integration time
t_span = (0.0, 2_000_000.0)
rtol, atol = 1e-6, 1e-9
chunk_steps_target = 400  # ~steps per chunk

# Output
output_dir      = "output_images"
CLEAR_OUTPUT_DIR = False  # wipe old frames first
# ---- Trajectory-only export (no background) ----
DRAW_CORE_CIRCLE      = True
HIDE_AXES             = True
TRAJ_ONLY_FIGSIZE     = (4, 4)
TRAJ_ONLY_DPI         = 600
TRAJ_BG_TRANSPARENT   = True
START_MARKER_COLOR    = 'g'   # green starts
TRAJ1_COLOR           = 'r'   # MB1 red
TRAJ2_COLOR           = 'b'   # MB2 blue
CORE_EDGE_COLOR       = 'k'   # circle black
PATH_LW               = 3.5   # thicker paths
frame_interval        = 20    # step between frames
SAVE_STEP_FRAMES      = True  # set False to only save the single static image
SAVE_JPG_FALLBACK     = True  # fallback to JPG if PNG save fails

# ============== GLOBALS (initialized in workers) ==============
TREE = None; UVAL=None; VVAL=None
XC=None; YC=None; A=None; GAM=None; RHO=None; CDG=None
RTOL=None; ATOL=None; CHUNK_STEPS_TARGET=None
XMIN=None; XMAX=None; YMIN=None; YMAX=None

# ----------------- Helpers -----------------
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
    X = X[m]; Y = Y[m]; U = U[m]; V = V[m]
    Umag = np.hypot(U, V) + 1e-12
    ux, uy = U/Umag, V/Umag
    xs = np.linspace(np.percentile(X,10), np.percentile(X,90), 60)
    ys = np.linspace(np.percentile(Y,10), np.percentile(Y,90), 60)
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
            cost = (w*(ur*ur)).sum()/w.sum() + 0.2*(w*(1.0-np.abs(ut))).sum()/w.sum()
            if cost < best_cost: best_cost, best_xy = cost, (xc, yc)
    return float(best_xy[0]), float(best_xy[1])

def kNN_velocity_field(x, y, k=8, eps=1e-12):
    d, idx = TREE.query((x, y), k=k)
    if np.isscalar(d): d = np.array([d]); idx = np.array([idx])
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

# ---------------- Dynamics ----------------
def rhs(t, Y):
    x1,y1,u1,v1,x2,y2,u2,v2 = Y
    # MB1
    uf1 = kNN_velocity_field(x1, y1); gradp1 = grad_p_rankine(x1, y1, XC, YC, GAM, RHO, A)
    du1 = uf1 - np.array([u1, v1]); du1_mag = np.hypot(du1[0], du1[1])
    ax1 = -(3.0/RHO)*gradp1[0] + 0.75*CDG*du1[0]*du1_mag
    ay1 = -(3.0/RHO)*gradp1[1] + 0.75*CDG*du1[1]*du1_mag
    # MB2
    uf2 = kNN_velocity_field(x2, y2); gradp2 = grad_p_rankine(x2, y2, XC, YC, GAM, RHO, A)
    du2 = uf2 - np.array([u2, v2]); du2_mag = np.hypot(du2[0], du2[1])
    ax2 = -(3.0/RHO)*gradp2[0] + 0.75*CDG*du2[0]*du2_mag
    ay2 = -(3.0/RHO)*gradp2[1] + 0.75*CDG*du2[1]*du2_mag
    return [u1, v1, ax1, ay1, u2, v2, ax2, ay2]

def out_of_bounds(t, Y):
    x1,y1,x2,y2 = Y[0],Y[1],Y[4],Y[5]
    return min(x1 - XMIN, XMAX - x1, y1 - YMIN, YMAX - y1,
               x2 - XMIN, XMAX - x2, y2 - YMIN, YMAX - y2)
out_of_bounds.terminal = True
out_of_bounds.direction = -1

def solve_ode_chunk(t0, t1, Yinit):
    span = t1 - t0
    max_step = max(1e-9, span / CHUNK_STEPS_TARGET)
    sol = solve_ivp(rhs, [t0, t1], Yinit, method="RK45",
                    rtol=RTOL, atol=ATOL, max_step=max_step,
                    dense_output=True, events=out_of_bounds)
    return sol.t, sol.y

def init_worker(points, u_values, v_values, xc, yc, a, gamma, rho, cd,
                rtol, atol, chunk_steps_target, xmin, xmax, ymin, ymax):
    global TREE, UVAL, VVAL, XC, YC, A, GAM, RHO, CDG, RTOL, ATOL, CHUNK_STEPS_TARGET
    global XMIN, XMAX, YMIN, YMAX
    TREE = cKDTree(points); UVAL = u_values; VVAL = v_values
    XC, YC = xc, yc; A, GAM, RHO, CDG = a, gamma, rho, cd
    RTOL, ATOL = rtol, atol; CHUNK_STEPS_TARGET = chunk_steps_target
    XMIN, XMAX, YMIN, YMAX = xmin, xmax, ymin, ymax

# ============================== MAIN ===========================
def main():
    print(f"Reading velocity data ...\nCSV: {CSV_PATH}")
    vf = pd.read_csv(CSV_PATH); vf.columns = vf.columns.str.strip()
    assert {'x','y','u','v'} <= set(vf.columns), "CSV must contain x,y,u,v"

    points = vf[['x','y']].to_numpy(float)
    u_values = vf['u'].to_numpy(float)
    v_values = vf['v'].to_numpy(float)

    xmin, xmax = points[:,0].min(), points[:,0].max()
    ymin, ymax = points[:,1].min(), points[:,1].max()
    print(f"Domain x:[{xmin},{xmax}], y:[{ymin},{ymax}]  (CSV units)")

    if not (xmin <= x0_1 <= xmax and ymin <= y0_1 <= ymax):
        raise ValueError(f"Initial MB1 (x0,y0)=({x0_1},{y0_1}) outside CSV domain.")
    if not (xmin <= x0_2 <= xmax and ymin <= y0_2 <= ymax):
        raise ValueError(f"Initial MB2 (x0,y0)=({x0_2},{y0_2}) outside CSV domain.")

    print("Estimating vortex center ...")
    xc, yc = estimate_vortex_center(vf)
    if CENTER_OVERRIDE is not None:
        xc, yc = CENTER_OVERRIDE; print("CENTER_OVERRIDE applied.")
    print(f"Center (used): (xc,yc)=({xc:.3f},{yc:.3f})")

    # sanity fallback
    if not (xmin <= xc <= xmax and ymin <= yc <= ymax):
        j = int(np.argmin(np.hypot(u_values, v_values)))
        xc, yc = float(points[j,0]), float(points[j,1])
        print(f"Center fallback to slowest point: (xc,yc)=({xc:.3f},{yc:.3f})")

    # core radius
    if a_override is None:
        r = np.hypot(vf["x"].to_numpy(float) - xc, vf["y"].to_numpy(float) - yc)
        s = np.hypot(u_values, v_values)
        rlo, rhi = np.percentile(r, [30, 95]); band = (r>=rlo)&(r<=rhi)
        r_peak = r[band][s[band].argmax()] if band.sum() > 0 else np.median(r)
        A_local = float(r_peak); print(f"Core radius a from peak speed: {A_local:.3f}")
    else:
        A_local = float(a_override); print(f"Core radius a (override): {A_local:.3f}")

    # output directory
    os.makedirs(output_dir, exist_ok=True)
    if CLEAR_OUTPUT_DIR:
        for fn in os.listdir(output_dir):
            fp = os.path.join(output_dir, fn)
            try:
                os.remove(fp) if os.path.isfile(fp) else shutil.rmtree(fp)
            except Exception as e:
                print(f"Warning: could not delete {fp}: {e}")

    # build chunks
    num_chunks = cpu_count()
    edges = np.linspace(t_span[0], t_span[1], num_chunks + 1)
    intervals = [(edges[i], edges[i+1]) for i in range(num_chunks)]

    # pool
    ctx = get_context("spawn")
    results = []
    with ctx.Pool(processes=num_chunks,
                  initializer=init_worker,
                  initargs=(points, u_values, v_values, xc, yc, A_local,
                            Gamma, rho, CD, rtol, atol, chunk_steps_target,
                            xmin, xmax, ymin, ymax)) as pool:
        Yprev = [x0_1, y0_1, u0_1, v0_1, x0_2, y0_2, u0_2, v0_2]
        results.append(pool.apply_async(solve_ode_chunk, args=(intervals[0][0], intervals[0][1], Yprev)))
        for i in range(1, num_chunks):
            t, y = results[i-1].get()
            print(f"Chunk {i}/{num_chunks} done, t∈[{intervals[i-1][0]}, {intervals[i-1][1]}]")
            Yprev = [y[0,-1], y[1,-1], y[2,-1], y[3,-1], y[4,-1], y[5,-1], y[6,-1], y[7,-1]]
            results.append(pool.apply_async(solve_ode_chunk, args=(intervals[i][0], intervals[i][1], Yprev)))
        final_t = []; final_y_list = []
        for i, r in enumerate(results):
            t, y = r.get(); final_t.extend(t); final_y_list.append(y)
            print(f"Collected chunk {i+1}/{num_chunks}")

    final_y = np.concatenate(final_y_list, axis=1)  # [x1,y1,u1,v1,x2,y2,u2,v2]

    # ---- SAVE TRAJECTORY CSV ----
    traj_out = "trajectory_two_mb.csv"
    traj_df = pd.DataFrame({
        "x1": final_y[0], "y1": final_y[1], "uMBx1": final_y[2], "uMBy1": final_y[3],
        "x2": final_y[4], "y2": final_y[5], "uMBx2": final_y[6], "uMBy2": final_y[7],
    })
    traj_df.to_csv(traj_out, index=False)
    print(f"Saved trajectory CSV: {os.path.abspath(traj_out)}")
    print(f"Output dir: {os.path.abspath(output_dir)}")

    # ============== TRAJECTORY-ONLY EXPORTS ==============
    tx1, ty1 = final_y[0], final_y[1]
    tx2, ty2 = final_y[4], final_y[5]
    n = tx1.size

    # 1) SINGLE COMPACT IMAGE (no background)
    fig, ax = plt.subplots(figsize=TRAJ_ONLY_FIGSIZE)
    ax.plot(tx1, ty1, color=TRAJ1_COLOR, lw=PATH_LW)
    ax.plot(tx2, ty2, color=TRAJ2_COLOR, lw=PATH_LW)
    ax.plot(tx1[0],  ty1[0],  'o', ms=7, color=START_MARKER_COLOR)
    ax.plot(tx2[0],  ty2[0],  'o', ms=7, color=START_MARKER_COLOR)
    ax.plot(tx1[-1], ty1[-1], 'o', ms=7, color=TRAJ1_COLOR)
    ax.plot(tx2[-1], ty2[-1], 'o', ms=7, color=TRAJ2_COLOR)
    if DRAW_CORE_CIRCLE:
        ax.add_patch(plt.Circle((xc, yc), A_local, edgecolor=CORE_EDGE_COLOR, facecolor='none', lw=2))
    ax.set_aspect('equal', 'box'); ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    if HIDE_AXES: ax.axis('off')
    fig.tight_layout(pad=0)
    try:
        fig.savefig("trajectory_only.png", dpi=TRAJ_ONLY_DPI, bbox_inches='tight',
                    pad_inches=0, transparent=TRAJ_BG_TRANSPARENT,
                    pil_kwargs={"compress_level": 1})
        print(f"Saved: {os.path.abspath('trajectory_only.png')}")
    except Exception as e:
        if SAVE_JPG_FALLBACK:
            fig.savefig("trajectory_only.jpg", dpi=TRAJ_ONLY_DPI, bbox_inches='tight', pad_inches=0)
            print(f"PNG failed ({e}). Saved JPG: {os.path.abspath('trajectory_only.jpg')}")
        else:
            raise
    # also export vector (crisp & tiny)
    fig.savefig("trajectory_only.svg", bbox_inches='tight', pad_inches=0, transparent=TRAJ_BG_TRANSPARENT)
    print(f"Saved: {os.path.abspath('trajectory_only.svg')}")
    plt.close(fig)

    # 2) OPTIONAL: PER-STEP TRAJECTORY-ONLY FRAMES
    if SAVE_STEP_FRAMES:
        for i in range(0, n, frame_interval):
            fig, ax = plt.subplots(figsize=TRAJ_ONLY_FIGSIZE)
            ax.plot(tx1[:i+1], ty1[:i+1], color=TRAJ1_COLOR, lw=PATH_LW)
            ax.plot(tx2[:i+1], ty2[:i+1], color=TRAJ2_COLOR, lw=PATH_LW)
            ax.plot(tx1[0], ty1[0], 'o', ms=7, color=START_MARKER_COLOR)
            ax.plot(tx2[0], ty2[0], 'o', ms=7, color=START_MARKER_COLOR)
            ax.plot(tx1[i], ty1[i], 'o', ms=7, color=TRAJ1_COLOR)
            ax.plot(tx2[i], ty2[i], 'o', ms=7, color=TRAJ2_COLOR)
            if DRAW_CORE_CIRCLE:
                ax.add_patch(plt.Circle((xc, yc), A_local, edgecolor=CORE_EDGE_COLOR, facecolor='none', lw=2))
            ax.set_aspect('equal', 'box'); ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
            if HIDE_AXES: ax.axis('off')
            fig.tight_layout(pad=0)
            out_png = os.path.join(output_dir, f"trajonly_{i:04d}.png")
            try:
                fig.savefig(out_png, dpi=TRAJ_ONLY_DPI, bbox_inches='tight',
                            pad_inches=0, transparent=TRAJ_BG_TRANSPARENT,
                            pil_kwargs={"compress_level": 1})
            except Exception as e:
                if SAVE_JPG_FALLBACK:
                    out_jpg = os.path.join(output_dir, f"trajonly_{i:04d}.jpg")
                    fig.savefig(out_jpg, dpi=TRAJ_ONLY_DPI, bbox_inches='tight', pad_inches=0)
                    print(f"PNG failed at frame {i} ({e}). Saved JPG fallback: {out_jpg}")
                else:
                    raise
            finally:
                plt.close(fig)
        print("Trajectory-only frames written to:", os.path.abspath(output_dir))

# ---- Windows/VS Code needs this guard for multiprocessing
if __name__ == "__main__":
    # Globals for workers
    RHO = rho; CDG = CD; GAM = Gamma
    RTOL = rtol; ATOL = atol; CHUNK_STEPS_TARGET = chunk_steps_target
    main()
