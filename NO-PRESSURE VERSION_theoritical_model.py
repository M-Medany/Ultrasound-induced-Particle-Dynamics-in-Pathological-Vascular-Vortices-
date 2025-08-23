# --- NO-PRESSURE VERSION ------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# ========= CONFIG =========
CSV_PATH = r"C:\Users\M4\VSCode_Projects\Ultrasound-Swarm-Microbubbles-Navigating-Vortices-to-Target-and-Fill-Aneurysms\Excel_data_velocity_comsol\Normalized_60_cm.csv"  # <-- set this
# velocity_data = pd.read_csv(r'C:\Users\M4\VSCode_Projects\Ultrasound-Swarm-Microbubbles-Navigating-Vortices-to-Target-and-Fill-Aneurysms\Excel_data_velocity_comsol\Normalized_60_cm.csv')


TAU = 5e-3             # relaxation [s]; try 1e-3 .. 2e-2
KNN_K = 8              # k-NN for smoothing interpolation
MAX_STEP = 1e-4        # ODE max time step [s]
T_SPAN = (0.0, 0.2)    # total simulation time [s]
X0, Y0 = 75e-6, 150e-6 # initial position [m] (adjust)
U0, V0 = 0.0, 0.0      # initial bubble velocity [m/s]
SKIP = 6               # quiver downsampling for plotting
# =========================

# ---- Load velocity field (SI units already in your CSV) ----
vf = pd.read_csv(CSV_PATH)
vf.columns = vf.columns.str.strip()
assert {'x','y','u','v'} <= set(vf.columns), "CSV must contain x,y,u,v"

points = vf[['x','y']].values
u_vals = vf['u'].values
v_vals = vf['v'].values
tree = cKDTree(points)

# Domain bounds from data (used for exit event and plotting axes)
XMIN, XMAX = float(vf['x'].min()), float(vf['x'].max())
YMIN, YMAX = float(vf['y'].min()), float(vf['y'].max())

# ---- Interpolated fluid velocity with k-NN smoothing ----
def velocity_field(x, y, k=KNN_K, eps=1e-12):
    d, idx = tree.query((x, y), k=k)
    if np.isscalar(d):  # when k=1
        d = np.array([d]); idx = np.array([idx])
    w = 1.0 / (d + eps)
    w /= w.sum()
    u = np.dot(w, u_vals[idx])
    v = np.dot(w, v_vals[idx])
    return np.array([u, v])

# ---- Dynamics (no pressure term) ----
def rhs_no_pressure(t, Y):
    x, y, ubx, uby = Y
    uf = velocity_field(x, y)
    dxdt, dydt = ubx, uby
    dubxdt = (uf[0] - ubx) / TAU
    dubydt = (uf[1] - uby) / TAU
    return [dxdt, dydt, dubxdt, dubydt]

# ---- Exit event (stop when bubble leaves the data rectangle) ----
def out_of_bounds(t, Y):
    x, y = Y[0], Y[1]
    return min(x - XMIN, XMAX - x, y - YMIN, YMAX - y)
out_of_bounds.terminal = True
out_of_bounds.direction = -1  # positive->negative crossing

# ---- Run integration ----
Y_init = [X0, Y0, U0, V0]
sol = solve_ivp(rhs_no_pressure, T_SPAN, Y_init,
                method="RK45", rtol=1e-6, atol=1e-9,
                max_step=MAX_STEP, events=out_of_bounds)

# ---- Plot quiver + trajectory (µm, cm/s for readability) ----
fig, ax = plt.subplots(figsize=(11, 5))
sub = vf.iloc[::SKIP].copy()
speed = np.hypot(sub['u'], sub['v'])
q = ax.quiver(sub['x']*1e6, sub['y']*1e6,
              sub['u']*100, sub['v']*100,
              speed, cmap='viridis', angles='xy', scale=2000)
cbar = plt.colorbar(q, ax=ax)
cbar.set_label('Velocity magnitude (m/s)')

traj_x = sol.y[0]*1e6
traj_y = sol.y[1]*1e6
ax.plot(traj_x, traj_y, 'k-', lw=2, label="trajectory")
ax.plot(traj_x[0], traj_y[0], 'go', ms=8, label="start")

ax.set_title('Bubble advection with strong drag (no pressure)')
ax.set_xlabel('x (µm)'); ax.set_ylabel('y (µm)')
ax.set_xlim(XMIN*1e6, XMAX*1e6); ax.set_ylim(YMIN*1e6, YMAX*1e6)
ax.set_aspect('equal', 'box'); ax.grid(alpha=0.3); ax.legend()
plt.savefig("No_Pressure_bubble_trajectory.png", dpi=300, bbox_inches='tight')
plt.show()
# ------------------------------------------------------------------------------
