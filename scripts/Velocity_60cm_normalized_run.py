import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree

from pathlib import Path as _Path
REPO_ROOT = _Path(__file__).resolve().parents[1]


# ----------- User configuration -----------
CSV_PATH = REPO_ROOT / "data" / "comsol" / "Normalized_Velocity_60_cm_Full.csv"
CSV_SKIPROWS = 7
CSV_COLUMNS = ["x", "y", "z", "u", "v", "w"]
POSITION_SCALE = 1.0   # set to 1e3 if CSV is in meters and you want mm
VELOCITY_SCALE = 1.0   # set to 100 if CSV is in m/s and you want cm/s
START_POS_ABS = None   # e.g. (0.56, 0.12); None uses quantiles below
START_POS_QUANTILES = (0.65, 0.25)
XVEL0, YVEL0 = 0.0, 0.0
T_SPAN = (0.0, 200_000.0)

# plotting
OUTPUT_DIR = "output_images"
STATIC_FIG = "velocity_trajectory_60cm.png"
FIGSIZE = (10, 5)
DPI = 400
QUIVER_SKIP = 5
QUIVER_MAX = 3000
QUIVER_Y_MIN = None
QUIVER_SCALE = 30
QUIVER_SEED = 0
TITLE = "60 cm/s velocity field + MB trajectory"
X_LABEL = "Position x"
Y_LABEL = "Position y"

# physics
RHO = 1000.0
CD = 10.0
GAMMA = 10.0
CORE_RADIUS_OVERRIDE = None


def load_velocity_field():
    try:
        df = pd.read_csv(CSV_PATH)
        df.columns = df.columns.str.strip()
        if not {"x", "y", "u", "v"} <= set(df.columns):
            raise ValueError("Missing expected columns")
    except Exception:
        df = pd.read_csv(CSV_PATH, skiprows=CSV_SKIPROWS, names=CSV_COLUMNS)
        df = df.replace(r"^\s+$", np.nan, regex=True).dropna(subset=["x", "y", "u", "v"])

    df = df.astype(float)
    df["x"] *= POSITION_SCALE
    df["y"] *= POSITION_SCALE
    df["u"] *= VELOCITY_SCALE
    df["v"] *= VELOCITY_SCALE
    df = df.drop_duplicates(subset=["x", "y"])
    df = df.reset_index(drop=True)
    return df


def estimate_vortex_center(df):
    X = df["x"].to_numpy()
    Y = df["y"].to_numpy()
    U = df["u"].to_numpy()
    V = df["v"].to_numpy()

    spd = np.hypot(U, V)
    xlo, xhi = np.percentile(X, [10, 90])
    ylo, yhi = np.percentile(Y, [10, 90])
    slo = np.percentile(spd, 40)
    mask = (X >= xlo) & (X <= xhi) & (Y >= ylo) & (Y <= yhi) & (spd > slo)

    if mask.sum() < 150:
        j = int(np.argmin(spd))
        return float(X[j]), float(Y[j])

    X = X[mask]
    Y = Y[mask]
    U = U[mask]
    V = V[mask]

    Umag = np.hypot(U, V) + 1e-12
    ux = U / Umag
    uy = V / Umag

    xs = np.linspace(np.percentile(X, 15), np.percentile(X, 85), 60)
    ys = np.linspace(np.percentile(Y, 15), np.percentile(Y, 85), 60)
    R0 = 0.35 * min(X.ptp(), Y.ptp())

    best_cost = 1e99
    best_xy = (np.median(X), np.median(Y))
    for xc in xs:
        dx = X - xc
        for yc in ys:
            dy = Y - yc
            r = np.hypot(dx, dy) + 1e-12
            rx = dx / r
            ry = dy / r
            tx = -ry
            ty = rx

            ur = ux * rx + uy * ry
            ut = ux * tx + uy * ty

            w = np.exp(-(r * r) / (2 * R0 * R0))
            wsum = w.sum()
            cost = (w * (ur * ur)).sum() / wsum + 0.25 * (w * (1.0 - np.abs(ut))).sum() / wsum

            if cost < best_cost:
                best_cost = cost
                best_xy = (xc, yc)

    return float(best_xy[0]), float(best_xy[1])


def rankine_pressure_gradient(x, y, xc, yc, gamma, rho, a):
    xr, yr = x - xc, y - yc
    r = np.hypot(xr, yr) + 1e-12
    if r < a:
        dpr = (rho * gamma ** 2) / (4 * np.pi ** 2) * (r / (a * a))
    else:
        dpr = (rho * gamma ** 2) / (4 * np.pi ** 2) * (1.0 / (r ** 3))
    return dpr * np.array([xr / r, yr / r])


def select_quiver_indices(X, Y, stride=1, y_min=None, max_count=None, seed=0):
    idx = np.arange(X.size)
    if y_min is not None:
        idx = idx[Y[idx] >= y_min]
    if stride > 1:
        idx = idx[::stride]
    if max_count is not None and idx.size > max_count:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=max_count, replace=False))
    return idx


def microbubble_rhs(t, Y, tree, u_val, v_val, xc, yc, gamma, rho, core_radius):
    x, y, u_bx, u_by = Y
    dist, idx = tree.query((x, y))
    idx = np.atleast_1d(idx)
    weights = 1.0 / (np.atleast_1d(dist) + 1e-12)
    weights /= weights.sum()
    uf = np.array([np.dot(weights, u_val[idx]), np.dot(weights, v_val[idx])])

    grad_p = rankine_pressure_gradient(x, y, xc, yc, gamma, rho, core_radius)

    du = uf - np.array([u_bx, u_by])
    du_mag = np.hypot(du[0], du[1])
    ax = -(3.0 / rho) * grad_p[0] + 0.75 * CD * du[0] * du_mag
    ay = -(3.0 / rho) * grad_p[1] + 0.75 * CD * du[1] * du_mag

    return [u_bx, u_by, ax, ay]


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    vf = load_velocity_field()
    print(f"Loaded {len(vf)} velocity samples.")

    xmin, xmax = vf["x"].min(), vf["x"].max()
    ymin, ymax = vf["y"].min(), vf["y"].max()
    print(f"x range: [{xmin}, {xmax}], y range: [{ymin}, {ymax}]")

    xc, yc = estimate_vortex_center(vf)
    print(f"Estimated vortex center: ({xc}, {yc})")

    if CORE_RADIUS_OVERRIDE is None:
        r = np.hypot(vf["x"] - xc, vf["y"] - yc)
        idx = np.argmax(np.hypot(vf["u"], vf["v"]))
        core_radius = float(r.iloc[idx])
    else:
        core_radius = CORE_RADIUS_OVERRIDE
    print(f"Core radius: {core_radius}")

    if START_POS_ABS is not None:
        x0, y0 = START_POS_ABS
    else:
        qx, qy = START_POS_QUANTILES
        x0 = float(vf["x"].quantile(qx))
        y0 = float(vf["y"].quantile(qy))
    print(f"Initial position: ({x0}, {y0})")

    if not (xmin <= x0 <= xmax and ymin <= y0 <= ymax):
        raise ValueError("Initial position lies outside the velocity domain.")

    points = vf[["x", "y"]].to_numpy()
    tree = cKDTree(points)
    u_val = vf["u"].to_numpy()
    v_val = vf["v"].to_numpy()

    y0_state = [x0, y0, XVEL0, YVEL0]
    sol = solve_ivp(
        microbubble_rhs,
        T_SPAN,
        y0_state,
        args=(tree, u_val, v_val, xc, yc, GAMMA, RHO, core_radius),
        method="RK45",
        rtol=1e-6,
        atol=1e-9,
        max_step=(T_SPAN[1] - T_SPAN[0]) / 400,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    traj = pd.DataFrame({"x": sol.y[0], "y": sol.y[1], "u": sol.y[2], "v": sol.y[3], "t": sol.t})
    traj_path = "trajectory_60cm.csv"
    traj.to_csv(traj_path, index=False)
    print(f"Saved trajectory to {os.path.abspath(traj_path)}")

    keep = select_quiver_indices(
        vf["x"].to_numpy(),
        vf["y"].to_numpy(),
        stride=max(1, QUIVER_SKIP),
        y_min=QUIVER_Y_MIN,
        max_count=QUIVER_MAX,
        seed=QUIVER_SEED,
    )

    X = vf["x"].to_numpy()[keep]
    Y = vf["y"].to_numpy()[keep]
    U = vf["u"].to_numpy()[keep]
    V = vf["v"].to_numpy()[keep]

    mag = np.hypot(U, V)
    mag = np.where(mag == 0, 1e-12, mag)
    Un = U / mag
    Vn = V / mag

    norm = plt.Normalize(mag.min(), mag.max())
    colors = plt.cm.viridis(norm(mag))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.quiver(X, Y, Un, Vn, color=colors, scale=QUIVER_SCALE, alpha=0.85)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax)
    cbar.set_label("Velocity magnitude (scaled)")

    ax.plot(sol.y[0], sol.y[1], color="k", lw=2.5)
    ax.plot(sol.y[0][0], sol.y[1][0], marker="o", color="g", ms=7)
    ax.plot(sol.y[0][-1], sol.y[1][-1], marker="o", markerfacecolor="none", markeredgecolor="r", ms=8)

    core = plt.Circle((xc, yc), core_radius, edgecolor="k", facecolor="none", lw=2)
    ax.add_patch(core)

    ax.set_title(TITLE)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_aspect("equal", "box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, STATIC_FIG)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
