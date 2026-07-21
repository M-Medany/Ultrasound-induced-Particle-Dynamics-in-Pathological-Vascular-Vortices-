import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from multiprocessing import Pool, cpu_count

from pathlib import Path as _Path
REPO_ROOT = _Path(__file__).resolve().parents[1]


# Constants
Gamma = 10
rho = 1000
a = 50
CD = 1.2
p_inf = 1e5

# Read velocity data from CSV
print("Reading velocity data from CSV...")
velocity_data = pd.read_csv(REPO_ROOT / "data" / "comsol" / "Normalized_60_cm.csv")

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
def pressure_field(r):
    if r < a:
        return p_inf - Gamma ** 2 / (4 * np.pi ** 2) * rho / a ** 2 + (rho * Gamma ** 2 * r ** 2) / (8 * np.pi ** 2 * a ** 2)
    else:
        return p_inf - Gamma ** 2 / (8 * np.pi ** 2) * rho / r ** 2

# Define the pressure gradient
def pressure_gradient(x, y, r):
    dr = 0.01  # finite difference for gradient approximation
    p_r_plus = pressure_field(r + dr)
    p_r_minus = pressure_field(r - dr)
    dp_dr = (p_r_plus - p_r_minus) / (2 * dr)
    grad_p = np.array([dp_dr * (x / r), dp_dr * (y / r)])
    return grad_p

# Dynamics function equivalent to MATLAB's microbubbleDynamics
def microbubble_dynamics(t, Y, tree, u_values, v_values):
    x, y, u_MBx, u_MBy = Y
    r = np.sqrt(x ** 2 + y ** 2)
    u = velocity_field(x, y, tree, u_values, v_values)
    grad_p = pressure_gradient(x, y, r)
    
    dxdt = u_MBx
    dydt = u_MBy
    du_MBx_dt = (3 / rho) * grad_p[0] + 3 / 4 * CD * (u[0] - u_MBx) * abs(u[0] - u_MBx)
    du_MBy_dt = (3 / rho) * grad_p[1] + 3 / 4 * CD * (u[1] - u_MBy) * abs(u[1] - u_MBy)
    
    return [dxdt, dydt, du_MBx_dt, du_MBy_dt]

# Initial conditions
x0 = [130, 50]
u_MB0 = [0, 0]
initial_conditions = x0 + u_MB0

# Time span for the simulation
t_span = [0, 3000000]

# Function to solve a part of the ODE
def solve_ode_chunk(t_chunk, initial_conditions, tree, u_values, v_values):
    solution = solve_ivp(lambda t, y: microbubble_dynamics(t, y, tree, u_values, v_values),
                         [t_chunk[0], t_chunk[1]], initial_conditions, method='RK45', dense_output=True)
    return solution.t, solution.y

if __name__ == '__main__':
    num_chunks = cpu_count()
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

    # Ensure the final_y is correctly processed
    final_y = np.array(final_y)

    # Create a figure for the animation
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'k-', linewidth=2)  # Changed line color to black ('k-')
    start_point, = ax.plot([], [], 'go', markerfacecolor='g', markersize=8)
    end_point, = ax.plot([], [], 'ro', markerfacecolor='r', markersize=8)

    # Set axis limits
    ax.set_xlim(velocity_data['x'].min(), velocity_data['x'].max())
    ax.set_ylim(velocity_data['y'].min(), velocity_data['y'].max())

    def init():
        line.set_data([], [])
        start_point.set_data([final_y[0, 0]], [final_y[1, 0]])  # Start point
        end_point.set_data([], [])  # Clear end point initially
        return line, start_point, end_point

    def update(frame):
        line.set_data(final_y[0, :frame], final_y[1, :frame])
        end_point.set_data([final_y[0, frame - 1]], [final_y[1, frame - 1]])  # Update end point
        return line, start_point, end_point

    ani = animation.FuncAnimation(fig, update, frames=len(final_t), init_func=init, blit=True)

    # Save the animation using Pillow
    ani.save('trajectory_animation.gif', fps=30, writer='pillow')

    plt.show()

    # Plotting results
    plt.figure(figsize=(10, 5))
    u_plot = np.array([velocity_field(final_y[0, i], final_y[1, i], tree, u_values, v_values) for i in range(len(final_t))])
    plt.quiver(final_y[0], final_y[1], u_plot[:, 0], u_plot[:, 1], color='r')
    plt.title('Velocity Field u Over Trajectory')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.grid(True, alpha=0.3)

    plt.figure(figsize=(10, 5))
    grad_p_plot = np.array([pressure_gradient(final_y[0, i], final_y[1, i], np.sqrt(final_y[0, i] ** 2 + final_y[1, i] ** 2)) for i in range(len(final_t))])
    plt.quiver(final_y[0], final_y[1], grad_p_plot[:, 0], grad_p_plot[:, 1], color='b')
    plt.title('Pressure Gradient Over Trajectory')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    plt.figure(figsize=(10, 5))
    plt.plot(final_y[0], final_y[1], 'b-', linewidth=2)
    plt.plot(final_y[0, 0], final_y[1, 0], 'go', markerfacecolor='g', markersize=8)  # Start point
    plt.plot(final_y[0, -1], final_y[1, -1], 'ro', markerfacecolor='r', markersize=8)  # End point
    plt.title('Microbubble Trajectory in 2D')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    plt.show()
