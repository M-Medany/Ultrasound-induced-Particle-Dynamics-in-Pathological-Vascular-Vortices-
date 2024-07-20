import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from numba import njit
from tqdm import tqdm  # Progress bar

# Constants
Gamma = 4 # 6.5
rho = 1000
a = 50
CD = 50  # 1.0
p_inf = 1e5

# Read velocity data from CSV
print("Reading velocity data from CSV...")
velocity_data = pd.read_csv(r'C:\Users\mmabo\V_Code\New folder\Aneurysm_filling\Normalized_Velocity_2d_5cm.csv')

# Clean column names if necessary
velocity_data.columns = velocity_data.columns.str.strip()
print(f"Columns in the data: {velocity_data.columns.tolist()}")

# Extract data for interpolation
points = velocity_data[['x', 'y']].values
u_values = velocity_data['u'].values
v_values = velocity_data['v'].values
print(f"Loaded {len(points)} points for interpolation.")

# Build a KDTree for fast interpolation
print("Building KDTree for fast interpolation...")
tree = cKDTree(points)


# Define the velocity field using KDTree
def velocity_field(x, y, tree, u_values, v_values):
    dist, idx = tree.query((x, y))
    u_theta = u_values[idx]
    v_theta = v_values[idx]
    return np.array([u_theta, v_theta])


# Define the pressure field
@njit
def pressure_field(r):
    if r < a:
        print(f"a={a}")
        return p_inf + Gamma ** 2 / (4 * np.pi ** 2) * rho / a ** 2 + (rho * Gamma ** 2 * r ** 2) / (
                    8 * np.pi ** 2 * a ** 2)
    else:
        print(f"r={r}")
        return p_inf + Gamma ** 2 / (8 * np.pi ** 2) * rho / r ** 2


# Define the pressure gradient
@njit
def pressure_gradient(x, y, r, theta):
    dr = 0.1  # finite difference for gradient approximation
    p_r_plus = pressure_field(r + dr)
    p_r_minus = pressure_field(r - dr)
    dp_dr = (p_r_plus - p_r_minus) / (2 * dr)
    grad_p = np.array([dp_dr * (x / r), dp_dr * (y / r)])
    return grad_p


# Dynamics function equivalent to MATLAB's microbubbleDynamics
def microbubble_dynamics(t, Y, tree, u_values, v_values):
    x, y, u_MBx, u_MBy = Y
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    u = velocity_field(x, y, tree, u_values, v_values)
    grad_p = pressure_gradient(x, y, r, theta)

    # Differential equations for velocity
    dxdt = u_MBx
    dydt = u_MBy
    du_MBx_dt = (3 / rho) * grad_p[0] + 3 / 4 * CD * (u[0] - u_MBx) * abs(u[0] - u_MBx)
    du_MBy_dt = (3 / rho) * grad_p[1] + 3 / 4 * CD * (u[1] - u_MBy) * abs(u[1] - u_MBy)

    return [dxdt, dydt, du_MBx_dt, du_MBy_dt]


# Initial conditions
x0 = [130, 100]  # Adjusted initial conditions
u_MB0 = [0, 0]
initial_conditions = x0 + u_MB0  # Combine lists

# Time span for the simulation
t_span = [0,40000]

# Function to solve a part of the ODE
def solve_ode_chunk(t_chunk, initial_conditions, tree, u_values, v_values):
    print(f"Starting chunk from {t_chunk[0]} to {t_chunk[1]}")
    solution = solve_ivp(lambda t, y: microbubble_dynamics(t, y, tree, u_values, v_values),
                         [t_chunk[0], t_chunk[1]], initial_conditions, method='RK45', dense_output=True)
    print(f"Finished chunk from {t_chunk[0]} to {t_chunk[1]}")
    return solution


if __name__ == '__main__':
    print("Starting ODE solver...")

    # Split the time span into chunks ensuring each chunk has at least two points
    num_chunks = 18
    time_chunks = np.linspace(t_span[0], t_span[1], num_chunks + 1)
    time_intervals = [(time_chunks[i], time_chunks[i + 1]) for i in range(num_chunks)]

    # Solve ODE in parallel with progress bar
    with Pool(processes=num_chunks) as pool:
        solutions = list(tqdm(pool.starmap(solve_ode_chunk,
                                           [(time_intervals[i], initial_conditions, tree, u_values, v_values) for i in
                                            range(num_chunks)]), total=num_chunks))

    # Combine the solutions
    t_points = np.concatenate([sol.t for sol in solutions])
    y_combined = np.concatenate([sol.y for sol in solutions], axis=1)

    print("ODE solver finished.")

    # Plotting results
    plt.figure(figsize=(10, 5))
    u_plot = np.array([velocity_field(y_combined[0, i], y_combined[1, i], tree, u_values, v_values) for i in
                       range(y_combined.shape[1])])
    plt.quiver(y_combined[0], y_combined[1], u_plot[:, 0], u_plot[:, 1], color='r')
    plt.title('Velocity Field u Over Trajectory')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.grid(True, alpha=0.3)

    plt.figure(figsize=(10, 5))
    grad_p_plot = np.array([pressure_gradient(y_combined[0, i], y_combined[1, i],
                                              np.sqrt(y_combined[0, i] ** 2 + y_combined[1, i] ** 2),
                                              np.arctan2(y_combined[1, i], y_combined[0, i])) for i in
                            range(y_combined.shape[1])])
    plt.quiver(y_combined[0], y_combined[1], grad_p_plot[:, 0], grad_p_plot[:, 1], color='b')
    plt.title('Pressure Gradient Over Trajectory')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    plt.figure(figsize=(10, 5))
    plt.plot(y_combined[0], y_combined[1], 'b-', linewidth=2)
    plt.plot(y_combined[0, 0], y_combined[1, 0], 'go', markerfacecolor='g', markersize=8)  # Start point
    plt.plot(y_combined[0, -1], y_combined[1, -1], 'ro', markerfacecolor='r', markersize=8)  # End point
    plt.title('Microbubble Trajectory in 2D')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    plt.show()
