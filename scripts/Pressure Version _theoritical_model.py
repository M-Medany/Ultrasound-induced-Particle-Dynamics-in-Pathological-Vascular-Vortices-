import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool, cpu_count

from pathlib import Path as _Path
REPO_ROOT = _Path(__file__).resolve().parents[1]


# ===================== Constants (SI) =====================
# Choose physically consistent values with your CSV (meters, m/s)
Gamma = 4      # [m^2/s] circulation (set to your case; your old "3" assumed different units)
rho   = 1000.0    # [kg/m^3]
a     = 0.5    # [m] vortex core radius ~ O(100 µm) for your 450×360 µm domain
CD    = 10
p_inf = 1e5

# =================== Output directory ====================
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

# ================== Read velocity CSV ====================
print("Reading velocity data from CSV...")
file_path = REPO_ROOT / "data" / "comsol" / "Velocity_2d_5cm.csv"
velocity_data = pd.read_csv(file_path)
velocity_data.columns = velocity_data.columns.str.strip()
print(f"Columns: {velocity_data.columns.tolist()}")

# KDTree points (meters) and values (m/s)
points  = velocity_data[['x','y']].values
u_values = velocity_data['u'].values
v_values = velocity_data['v'].values
print(f"Loaded {len(points)} points for interpolation.")

print("Building KDTree...")
tree = cKDTree(points)

# ============ k-NN weighted velocity interpolation ============
def velocity_field(x, y, tree, u_values, v_values, k=8, eps=1e-12):
    d, idx = tree.query((x, y), k=k)
    if np.isscalar(d):  # when k=1
        d   = np.array([d])
        idx = np.array([idx])
    w = 1.0 / (d + eps)
    w /= w.sum()
    u = np.dot(w, u_values[idx])
    v = np.dot(w, v_values[idx])
    return np.array([u, v])

# ================= Analytic Rankine ∇p ==================
def dp_dr_rankine(r, Gamma, rho, a):
    # from your p(r): dp/dr = rho*Gamma^2/(4*pi^2) * ( r/a^2  if r<a  else 1/r^3 )
    if r < a:
        return (rho * Gamma**2) / (4*np.pi**2) * (r / (a*a))
    else:
        return (rho * Gamma**2) / (4*np.pi**2) * (1.0 / (r**3))

def pressure_gradient(x, y, Gamma, rho, a, eps=1e-15):
    r = np.hypot(x, y) + eps
    dpr = dp_dr_rankine(r, Gamma, rho, a)
    # ∇p = (dp/dr) * (x/r, y/r)
    return dpr * np.array([x/r, y/r])

# =========== Dynamics (your exact governing eq.) ===========
def microbubble_dynamics(t, Y, tree, u_values, v_values, Gamma, rho, a):
    x, y, u_MBx, u_MBy = Y
    u = velocity_field(x, y, tree, u_values, v_values, k=8)
    grad_p = pressure_gradient(x, y, Gamma, rho, a)

    dxdt = u_MBx
    dydt = u_MBy
    du_vec = u - np.array([u_MBx, u_MBy])
    du_mag = np.hypot(du_vec[0], du_vec[1])

    # Your sign and form:
    du_MBx_dt = (3.0/rho) * grad_p[0] + 0.75 * CD * du_vec[0] * du_mag
    du_MBy_dt = (3.0/rho) * grad_p[1] + 0.75 * CD * du_vec[1] * du_mag
    return [dxdt, dydt, du_MBx_dt, du_MBy_dt]

# ================== Initial conditions (SI) ==================1111111
# Your old [70,150] were µm—convert to meters so ODE matches CSV units11
x0 = [50e-6, 100e-6]    # [m]
u_MB0 = [0.0, 0.0]      # [m/s]
initial_conditions = x0 + u_MB0

# ================ Time span (seconds) =======================
t_span = [0.0, 0.10]    # simulate 0.3 s (adjust as you like)

# ============ Per-chunk ODE solver (kept) ===================
def solve_ode_chunk(t_chunk, initial_conditions, tree, u_values, v_values, Gamma, rho, a):
    sol = solve_ivp(lambda t, y: microbubble_dynamics(t, y, tree, u_values, v_values, Gamma, rho, a),
                    [t_chunk[0], t_chunk[1]], initial_conditions,
                    method='RK45', dense_output=True, rtol=1e-6, atol=1e-9, max_step=1e-4)
    return sol.t, sol.y

if __name__ == '__main__':
    num_chunks = cpu_count()
    time_chunks = np.linspace(t_span[0], t_span[1], num_chunks + 1)
    time_intervals = [(time_chunks[i], time_chunks[i+1]) for i in range(num_chunks)]

    initial_conditions_list = [initial_conditions] + [[None]]*(num_chunks-1)

    results = []
    with Pool(processes=num_chunks) as pool:
        for i in range(num_chunks):
            if i == 0:
                results.append(pool.apply_async(
                    solve_ode_chunk, args=(time_intervals[i], initial_conditions, tree, u_values, v_values, Gamma, rho, a)))
            else:
                prev_t, prev_y = results[i-1].get()   # keep your chaining
                initial_conditions_list[i] = [prev_y[0,-1], prev_y[1,-1], prev_y[2,-1], prev_y[3,-1]]
                results.append(pool.apply_async(
                    solve_ode_chunk, args=(time_intervals[i], initial_conditions_list[i], tree, u_values, v_values, Gamma, rho, a)))

        final_t = []
        final_y_list = []
        for r in results:
            t, y = r.get()
            final_t.extend(t)
            final_y_list.append(y)

    final_y = np.concatenate(final_y_list, axis=1)  # rows: [x,y,uMBx,uMBy], cols: time steps

    # ================== Plot & save frames ==================
    def read_and_plot_data(file_path, skip_interval=1, scale_factor=50, frame_interval=10):
        try:
            data = pd.read_csv(file_path)
            data.columns = data.columns.str.strip()

            # Scale only for plotting (axes in µm, arrows in cm/s)
            data_plot = data.copy()
            data_plot['x'] *= 1e6
            data_plot['y'] *= 1e6
            data_plot['u'] *= 100.0
            data_plot['v'] *= 100.0

            X = data_plot['x'].values[::skip_interval]
            Y = data_plot['y'].values[::skip_interval]
            U = data_plot['u'].values[::skip_interval]
            V = data_plot['v'].values[::skip_interval]

            mags = np.hypot(U, V)
            mags = np.where(mags == 0, 1e-12, mags)
            U_n = U / mags
            V_n = V / mags

            norm = plt.Normalize(mags.min(), mags.max())
            colors = plt.cm.viridis(norm(mags))

            # Convert trajectory to µm for overlay & masking
            traj_x_um = final_y[0, :] * 1e6
            traj_y_um = final_y[1, :] * 1e6

            mask_radius_um = 10.0  # µm

            for i in range(0, len(traj_x_um), frame_interval):
                # mask vectors near the current particle (for visual clarity)
                d = np.hypot(X - traj_x_um[i], Y - traj_y_um[i])
                mask = d > mask_radius_um

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.quiver(X[mask], Y[mask], U_n[mask], V_n[mask], color=colors[mask],
                          scale=scale_factor, alpha=1.0)
                cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
                cbar.set_label('Velocity Magnitude (m/s)')

                # Trajectory up to frame i (µm)
                ax.plot(traj_x_um[:i+1], traj_y_um[:i+1], 'k-', linewidth=2)
                ax.plot(traj_x_um[0], traj_y_um[0], 'go', markerfacecolor='g', markersize=8)

                # Particle marker + white halo
                cx, cy = traj_x_um[i], traj_y_um[i]
                circle = plt.Circle((cx, cy), 1.5*mask_radius_um, color='white', alpha=0.95)
                ax.add_patch(circle)
                ax.plot(cx, cy, 'ro', markersize=6)

                ax.set_title('Velocity Vector Plot with Transparent Arrows')
                ax.set_xlabel('X (µm)'); ax.set_ylabel('Y (µm)')
                ax.axis('equal'); ax.grid(True, alpha=0.3)

                plt.savefig(f"{output_dir}/frame_{i:04d}.png", dpi=300)
                plt.show(block=True)
                plt.close(fig)

        except Exception as e:
            print(f"Plotting error: {e}")

    read_and_plot_data(file_path, frame_interval=20)  # Save every 20th solver step

    # ffmpeg -framerate 30 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p output_video.mp4
