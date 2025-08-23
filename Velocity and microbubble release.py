# spawn_safe_mb.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
from multiprocessing import get_context, cpu_count

# ===================== USER CONFIG =====================
CSV_PATH = r"C:\Users\M4\VSCode_Projects\Ultrasound-Swarm-Microbubbles-Navigating-Vortices-to-Target-and-Fill-Aneurysms\Excel_data_velocity_comsol\Velocity_2d_5cm.csv"

rho = 1000.0      # density
CD  = 2         # quadratic drag coeff
Gamma = 0.1    # circulation (CSV-units^2 / time)
a_override = None # e.g., set 60.0 to force; otherwise auto
CENTER_OVERRIDE = None
# Initial state (must be inside CSV domain)
x0, y0 = 3900, 170.0
u0, v0 = 0.0, 0.0

# Time
t_span = (0.0, 2_000_000.0)
rtol, atol = 1e-6, 1e-9
chunk_steps_target = 400  # ~steps per chunk

# Output frames (for video)
output_dir = "output_images"
frame_interval = 20
skip_interval  = 10
mask_radius    = 10.0     # CSV units
quiver_scale   = 30

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
    X = df["x"].values; Y = df["y"].values
    U = df["u"].values; V = df["v"].values

    # mask out slow + top/bottom strips
    speed = np.hypot(U, V)
    ymin, ymax = Y.min(), Y.max()
    m = (Y > ymin + 25) & (Y < ymax - 25) & (speed > np.percentile(speed, 30))
    X = X[m]; Y = Y[m]; U = U[m]; V = V[m]
    if X.size < 200:
        # fallback to centroid if masking got too aggressive
        return float(np.median(df["x"])), float(np.median(df["y"]))

    # candidate grid (dense enough but fast)
    xs = np.linspace(np.percentile(X, 5), np.percentile(X, 95), 60)
    ys = np.linspace(np.percentile(Y, 5), np.percentile(Y, 95), 60)

    # unit velocity directions
    Umag = np.hypot(U, V) + 1e-12
    ux = U / Umag
    uy = V / Umag

    # window radius for weighting (favor nearby points)
    R0 = 0.35 * min(X.ptp(), Y.ptp())
    best_cost, best_xy = 1e99, (X.mean(), Y.mean())

    for xc in xs:
        dx = X - xc
        for yc in ys:
            dy = Y - yc
            r = np.hypot(dx, dy) + 1e-12
            rx, ry = dx/r, dy/r                    # radial unit
            tx, ty = -ry, rx                       # tangential unit
            # angle-only costs: suppress radial component, reward tangential alignment
            ur = ux*rx + uy*ry                     # cos(angle to radial)
            ut = ux*tx + uy*ty                     # cos(angle to tangential)
            w  = np.exp(-(r*r)/(2*R0*R0))          # Gaussian window
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

def rhs(t, Y):
    # Uses globals set by init_worker
    x, y, uMBx, uMBy = Y
    uf = kNN_velocity_field(x, y, k=8)
    gradp = grad_p_rankine(x, y, XC, YC, GAM, RHO, A)
    dxdt, dydt = uMBx, uMBy
    du = uf - np.array([uMBx, uMBy]); du_mag = np.hypot(du[0], du[1])
    # Pressure acceleration with correct sign + quadratic drag
    ax = -(3.0/RHO) * gradp[0] + 0.75 * CDG * du[0] * du_mag
    ay = -(3.0/RHO) * gradp[1] + 0.75 * CDG * du[1] * du_mag
    return [dxdt, dydt, ax, ay]

def out_of_bounds(t, Y):
    # positive inside, negative once any margin is crossed
    x, y = Y[0], Y[1]
    return min(x - XMIN, XMAX - x, y - YMIN, YMAX - y)
out_of_bounds.terminal = True
out_of_bounds.direction = -1


def solve_ode_chunk(t0, t1, Yinit):
    span = t1 - t0
    max_step = max(1e-9, span / CHUNK_STEPS_TARGET)
    sol = solve_ivp(
        rhs, [t0, t1], Yinit, method="RK45",
        rtol=RTOL, atol=ATOL, max_step=max_step,
        dense_output=True, events=out_of_bounds   # <<< here
    )
    return sol.t, sol.y


# ------------- Worker initializer (runs in each child) ----------
def init_worker(points, u_values, v_values, xc, yc, a, gamma, rho, cd,
                rtol, atol, chunk_steps_target,
                xmin, xmax, ymin, ymax):      # <<< add these
    global TREE, UVAL, VVAL, XC, YC, A, GAM, RHO, CDG, RTOL, ATOL, CHUNK_STEPS_TARGET
    global XMIN, XMAX, YMIN, YMAX            # <<< add these
    TREE = cKDTree(points)
    UVAL = u_values
    VVAL = v_values
    XC, YC = xc, yc
    A, GAM, RHO, CDG = a, gamma, rho, cd
    RTOL, ATOL = rtol, atol
    CHUNK_STEPS_TARGET = chunk_steps_target
    XMIN, XMAX, YMIN, YMAX = xmin, xmax, ymin, ymax   # <<< add these




# ============================== MAIN ===========================
def main():
    print("Reading velocity data ...")
    vf = pd.read_csv(CSV_PATH)
    vf.columns = vf.columns.str.strip()
    assert {'x','y','u','v'} <= set(vf.columns), "CSV must contain x,y,u,v"

    points = vf[['x','y']].values
    u_values = vf['u'].values
    v_values = vf['v'].values

    xmin, xmax = points[:,0].min(), points[:,0].max()
    ymin, ymax = points[:,1].min(), points[:,1].max()
    print(f"Domain x:[{xmin},{xmax}], y:[{ymin},{ymax}]  (CSV units)")

    if not (xmin <= x0 <= xmax and ymin <= y0 <= ymax):
        raise ValueError(f"Initial (x0,y0)=({x0},{y0}) outside CSV domain.")

    print("Estimating vortex center ...")
    xc, yc = estimate_vortex_center(vf)

    # manual override (set CENTER_OVERRIDE at the top next to your other config)
    if CENTER_OVERRIDE is not None:
        xc, yc = CENTER_OVERRIDE

    print(f"Center (used): (xc,yc)=({xc:.3f},{yc:.3f})")


    if a_override is None:
        r = np.hypot(vf["x"].values - xc, vf["y"].values - yc)
        s = np.hypot(vf["u"].values, vf["v"].values)
        m = (vf["y"].values > vf["y"].min() + 25) & (s > np.percentile(s, 30))
        r_peak = r[m][s[m].argmax()]
        A_local = float(r_peak)
        print(f"Core radius a from peak speed: {A_local:.3f}")
    else:
        A_local = float(a_override)
        print(f"Core radius a (override): {A_local:.3f}")

    os.makedirs(output_dir, exist_ok=True)

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
                            Gamma, rho, CD, rtol, atol, chunk_steps_target, xmin, xmax, ymin, ymax)) as pool:
        # first
        Yprev = [x0, y0, u0, v0]
        results.append(pool.apply_async(solve_ode_chunk, args=(intervals[0][0], intervals[0][1], Yprev)))
        # chain
        for i in range(1, num_chunks):
            t, y = results[i-1].get()
            print(f"Chunk {i}/{num_chunks} done, t∈[{intervals[i-1][0]}, {intervals[i-1][1]}]")
            Yprev = [y[0,-1], y[1,-1], y[2,-1], y[3,-1]]
            results.append(pool.apply_async(solve_ode_chunk, args=(intervals[i][0], intervals[i][1], Yprev)))

        final_t = []
        final_y_list = []
        for i, r in enumerate(results):
            t, y = r.get()
            final_t.extend(t)
            final_y_list.append(y)
            print(f"Collected chunk {i+1}/{num_chunks}")

    final_y = np.concatenate(final_y_list, axis=1)  # rows [x,y,uMBx,uMBy]

    # ============== Frames (same look as yours) ==============
    def save_frames():
        X = vf['x'].values; Y = vf['y'].values
        U = vf['u'].values; V = vf['v'].values
        Xs = X[::skip_interval]; Ys = Y[::skip_interval]
        Us = U[::skip_interval]; Vs = V[::skip_interval]
        mags = np.hypot(Us,Vs); mags = np.where(mags==0, 1e-12, mags)
        Un = Us/mags; Vn = Vs/mags
        norm = plt.Normalize(mags.min(), mags.max())
        colors = plt.cm.viridis(norm(mags))

        tx, ty = final_y[0], final_y[1]
        n = tx.size
        for i in range(0, n, frame_interval):
            d = np.hypot(Xs - tx[i], Ys - ty[i])
            mask = d > mask_radius
            fig, ax = plt.subplots(figsize=(10,5))
            ax.quiver(Xs[mask], Ys[mask], Un[mask], Vn[mask],
                      color=colors[mask], scale=quiver_scale, alpha=0.9)
            cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
            cbar.set_label('Velocity Magnitude (CSV units/s)')

            ax.plot(tx[:i+1], ty[:i+1], 'k-', lw=2, label='trajectory')
            ax.plot(tx[0], ty[0], 'go', ms=7, label='start')
            ax.plot(tx[i], ty[i], 'ro', ms=5)

            core = plt.Circle((xc, yc), A_local, edgecolor='r', facecolor='none', lw=2)
            ax.add_patch(core)
            ax.set_title('Velocity field + MB trajectory')
            ax.set_xlabel('x (CSV units)'); ax.set_ylabel('y (CSV units)')
            ax.set_aspect('equal', 'box'); ax.grid(alpha=0.3)
            ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
            if i == 0:
                ax.legend(loc='upper right')

            out_path = os.path.join(output_dir, f"frame_{i:04d}.png")
            plt.savefig(out_path, dpi=200); plt.close(fig)
        print(f"Frames saved to: {os.path.abspath(output_dir)}")
        print("Make video:\n  ffmpeg -framerate 30 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p output_video.mp4")

    def save_static():
        X = vf['x'].values; Y = vf['y'].values
        U = vf['u'].values; V = vf['v'].values
        Xs = X[::skip_interval]; Ys = Y[::skip_interval]
        Us = U[::skip_interval]; Vs = V[::skip_interval]
        mags = np.hypot(Us,Vs); mags = np.where(mags==0, 1e-12, mags)
        Un = Us/mags; Vn = Vs/mags
        norm = plt.Normalize(mags.min(), mags.max())
        colors = plt.cm.viridis(norm(mags))

        fig, ax = plt.subplots(figsize=(10,5))
        ax.quiver(Xs, Ys, Un, Vn, color=colors, scale=quiver_scale, alpha=0.9)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
        cbar.set_label('Velocity Magnitude (CSV units/s)')
        tx, ty = final_y[0], final_y[1]
        ax.plot(tx, ty, 'k-', lw=2)
        ax.plot(tx[0], ty[0], 'go', ms=7)
        ax.plot(tx[-1], ty[-1], 'ro', ms=7)
        core = plt.Circle((xc, yc), A_local, edgecolor='r', facecolor='none', lw=2)
        ax.add_patch(core)
        ax.set_title('Velocity field + MB trajectory (static)')
        ax.set_xlabel('x (CSV units)'); ax.set_ylabel('y (CSV units)')
        ax.set_aspect('equal', 'box'); ax.grid(alpha=0.3)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        plt.savefig("velocity_Trajectory_plot.png", dpi=300); plt.close(fig)
        print("Saved: velocity_Trajectory_plot.png")

    save_frames()
    save_static()

# Windows/VS Code needs this guard for multiprocessing
if __name__ == "__main__":
    main()
