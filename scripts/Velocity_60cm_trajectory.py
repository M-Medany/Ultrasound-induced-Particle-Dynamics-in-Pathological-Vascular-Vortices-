import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import os

from pathlib import Path as _Path
REPO_ROOT = _Path(__file__).resolve().parents[1]


# Constants
Gamma = 10
rho = 1000
a = 50
CD = 10
p_inf = 1e5

# Create output directory for images if necessary
output_dir = 'output_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read velocity data from CSV
print("Reading velocity data from CSV...")
# file_path = REPO_ROOT / "data" / "comsol" / "Normalized_Velocity_60_cm_Full.csv"
file_path = REPO_ROOT / "data" / "comsol" / "Normalized_Velocity_60_cm_Full.csv"

velocity_data = pd.read_csv(file_path, skiprows=7, names=['x', 'y', 'z', 'u', 'v', 'w'])

# Clean data (direct, without dropping rows)
velocity_data.columns = velocity_data.columns.str.strip()
velocity_data.fillna(0, inplace=True)
print(f"Columns in the data: {velocity_data.columns.tolist()}")
print(f"Loaded {len(velocity_data)} points for interpolation.")

# Prepare data for interpolation
points = velocity_data[['x', 'y']].values
u_values = velocity_data['u'].values
v_values = velocity_data['v'].values

# Build KDTree
print("Building KDTree for fast interpolation...")
tree = cKDTree(points)

# Define velocity field using KDTree
def velocity_field(x, y, tree, u_values, v_values):
    dist, idx = tree.query((x, y))
    u_theta = u_values[idx]
    v_theta = v_values[idx]
    return np.array([u_theta, v_theta])

# Define the pressure field
def pressure_field(r):
    if r < a:
        return p_inf - Gamma ** 2 / (4 * np.pi ** 2) * rho / a ** 2 + (rho * Gamma ** 2 * r ** 2) / (8 * np.pi ** 2 * a ** 2)
    else:
        return p_inf - Gamma ** 2 / (8 * np.pi ** 2) * rho / r ** 2

# Define pressure gradient
def pressure_gradient(x, y, r):
    dr = 0.01
    p_r_plus = pressure_field(r + dr)
    p_r_minus = pressure_field(r - dr)
    dp_dr = (p_r_plus - p_r_minus) / (2 * dr)
    
    # Prevent division by zero by checking if r is close to zero
    if r < 1e-10:
        return np.array([0.0, 0.0])  # No gradient if r is effectively zero
    
    grad_p = np.array([dp_dr * (x / r), dp_dr * (y / r)])
    
    if not np.all(np.isfinite(grad_p)):
        raise ValueError(f"Invalid pressure gradient: {grad_p} for r: {r}, x: {x}, y: {y}")
    
    return grad_p

# Microbubble dynamics
def microbubble_dynamics(t, Y, tree, u_values, v_values):
    x, y, u_MBx, u_MBy = Y
    r = np.sqrt(x ** 2 + y ** 2)
    
    if not np.isfinite(r):
        raise ValueError(f"Invalid radius r: {r}, derived from x: {x}, y: {y}")
    
    u = velocity_field(x, y, tree, u_values, v_values)
    grad_p = pressure_gradient(x, y, r)
    
    dxdt = u_MBx
    dydt = u_MBy
    du_MBx_dt = (3 / rho) * grad_p[0] + 3 / 4 * CD * (u[0] - u_MBx) * abs(u[0] - u_MBx)
    du_MBy_dt = (3 / rho) * grad_p[1] + 3 / 4 * CD * (u[1] - u_MBy) * abs(u[1] - u_MBy)
    
    return [dxdt, dydt, du_MBx_dt, du_MBy_dt]

# Initial conditions
x0 = [110, 100]
u_MB0 = [0, 0]
initial_conditions = x0 + u_MB0

# Time span
t_span = [0, 200000]

# Solve ODE chunk
def solve_ode_chunk(t_chunk, initial_conditions, tree, u_values, v_values):
    solution = solve_ivp(lambda t, y: microbubble_dynamics(t, y, tree, u_values, v_values),
                         [t_chunk[0], t_chunk[1]], initial_conditions, method='RK45', dense_output=True)
    return solution.t, solution.y

if __name__ == '__main__':
    num_chunks = 1  # Adjust this for more CPU parallelism
    time_chunks = np.linspace(t_span[0], t_span[1], num_chunks + 1)
    time_intervals = [(time_chunks[i], time_chunks[i + 1]) for i in range(num_chunks)]
    
    initial_conditions_list = [initial_conditions]
    for i in range(1, num_chunks):
        initial_conditions_list.append([None])

    with Pool(processes=num_chunks) as pool:
        results = []
        for i in range(num_chunks):
            if i == 0:
                results.append(pool.apply_async(solve_ode_chunk, args=(time_intervals[i], initial_conditions, tree, u_values, v_values)))
            else:
                prev_t, prev_y = results[i - 1].get()
                initial_conditions_list[i] = [prev_y[0, -1], prev_y[1, -1], prev_y[2, -1], prev_y[3, -1]]
                results.append(pool.apply_async(solve_ode_chunk, args=(time_intervals[i], initial_conditions_list[i], tree, u_values, v_values)))

        final_t = []
        final_y = []
        for result in results:
            t, y = result.get()
            final_t.extend(t)
            final_y.append(y)

    final_y = np.concatenate(final_y, axis=1)
    print("ODE solver finished.")

    # Plotting results
    def read_and_plot_data(file_path, skip_interval=1, scale_factor=30, distance_threshold=0.002):
        try:
            # Load data
            data = pd.read_csv(file_path)
            data.columns = data.columns.str.strip()
            data['x'] *= 1e6
            data['y'] *= 1e6
            data['u'] *= 100
            data['v'] *= 100

            # Extract data for plotting
            X = data['x'].values
            Y = data['y'].values
            U = data['u'].values
            V = data['v'].values

            # Skipping data points
            X_skipped = X[::skip_interval]
            Y_skipped = Y[::skip_interval]
            U_skipped = U[::skip_interval]
            V_skipped = V[::skip_interval]

            # Calculate magnitudes for normalization
            magnitudes = np.sqrt(U_skipped**2 + V_skipped**2)
            magnitudes = np.where(magnitudes == 0, 1e-10, magnitudes)

            U_normalized = U_skipped / magnitudes
            V_normalized = V_skipped / magnitudes

            # Normalize magnitudes for coloring
            norm = plt.Normalize(magnitudes.min(), magnitudes.max())
            colors = plt.cm.jet(norm(magnitudes))

            # Filter out velocity vectors near the trajectory
            trajectory_points = np.vstack((final_y[0], final_y[1])).T
            trajectory_tree = cKDTree(trajectory_points)
            distances, _ = trajectory_tree.query(np.vstack((X_skipped, Y_skipped)).T)
            mask = distances > distance_threshold

            X_filtered = X_skipped[mask]
            Y_filtered = Y_skipped[mask]
            U_filtered = U_normalized[mask]
            V_filtered = V_normalized[mask]
            colors_filtered = colors[mask]

            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.quiver(X_filtered, Y_filtered, U_filtered, V_filtered, color=colors_filtered, scale=scale_factor, alpha=0.5)
            cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax)
            cbar.set_label('Velocity Magnitude')
            ax.set_title('Velocity Vector Plot with Jet Color Map')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.axis('equal')
            ax.grid(True, alpha=0.3)

            # Plot trajectory on top of the velocity field
            ax.plot(final_y[0], final_y[1], 'k-', linewidth=2)
            ax.plot(final_y[0, 0], final_y[1, 0], 'go', markerfacecolor='g', markersize=8)  # Start point
            ax.plot(final_y[0, -1], final_y[1, -1], 'ro', markerfacecolor='r', markersize=8)  # End point

            plt.show()

        except FileNotFoundError:
            print(f"Error: The file at {file_path} was not found.")
        except KeyError as e:
            print(f"Error: Missing expected column in the CSV file - {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    # Example usage
    file_path = REPO_ROOT / "data" / "comsol" / "Normalized_Velocity_60_cm_Full.csv"
    read_and_plot_data(file_path)
