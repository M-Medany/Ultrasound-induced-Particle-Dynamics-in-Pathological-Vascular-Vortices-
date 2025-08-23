import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Constants
Gamma = 10
rho = 1000
a = 0.005
CD = 1
p_inf = 1e5

# Read velocity data from CSV
print("Reading velocity data from CSV...")
# velocity_data = pd.read_csv(r'C:\Users\mmabo\V_Code\New folder\Aneurysm_filling\Excel_data_velocity_comsol\Normalized_Velocity_2d_5cm.csv')
velocity_data = pd.read_csv(r'C:\Users\M4\VSCode_Projects\Ultrasound-Swarm-Microbubbles-Navigating-Vortices-to-Target-and-Fill-Aneurysms\Excel_data_velocity_comsol\Normalized_Velocity_2d_5cm.csv')

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
t_span = [0, 2000000]

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

    # Plotting results
    def read_and_plot_data(file_path, skip_interval=10, scale_factor=30):
        try:
            # Load data
            data = pd.read_csv(file_path)

            # Clean column names if necessary
            data.columns = data.columns.str.strip()

            # Extract data for plotting
            X = data['x'].values
            Y = data['y'].values
            U = data['u'].values
            V = data['v'].values

            # Skipping data points to reduce density
            X_skipped = X[::skip_interval]
            Y_skipped = Y[::skip_interval]
            U_skipped = U[::skip_interval]
            V_skipped = V[::skip_interval]

            # Calculate magnitudes for normalization
            magnitudes = np.sqrt(U_skipped**2 + V_skipped**2)

            # Avoid division by zero by adding a small number to magnitudes
            magnitudes = np.where(magnitudes == 0, 1e-10, magnitudes)

            U_normalized = U_skipped / magnitudes
            V_normalized = V_skipped / magnitudes

            # Normalize magnitudes for coloring
            norm = plt.Normalize(magnitudes.min(), magnitudes.max())
            colors = plt.cm.viridis(norm(magnitudes))

            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 5))
            quiver = ax.quiver(X_skipped, Y_skipped, U_normalized, V_normalized, color=colors, scale=scale_factor, alpha=0.5)
            cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
            cbar.set_label('Velocity Magnitude')
            ax.set_title('Velocity Vector Plot with Transparent Arrows')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.axis('equal')
            ax.grid(True, alpha=0.3)

            # Plot trajectory on top of the velocity field with a mask
            mask_radius = 10  # Adjust the radius of the mask as needed
            mask = np.full_like(X_skipped, True, dtype=bool)

            for i in range(len(final_y[0])):
                distances = np.sqrt((X_skipped - final_y[0, i])**2 + (Y_skipped - final_y[1, i])**2)
                mask = np.logical_and(mask, distances > mask_radius)

            X_masked = X_skipped[mask]
            Y_masked = Y_skipped[mask]
            U_masked = U_normalized[mask]
            V_masked = V_normalized[mask]

            # Create a new quiver plot with the masked data
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.quiver(X_masked, Y_masked, U_masked, V_masked, color=colors[mask], scale=scale_factor, alpha=0.5)
            cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
            cbar.set_label('Velocity Magnitude')
            ax.set_title('Velocity Vector Plot with Transparent Arrows')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.axis('equal')
            ax.grid(True, alpha=0.3)

            # Plot the trajectory
            ax.plot(final_y[0], final_y[1], 'r-', linewidth=2, color='black')
            ax.plot(final_y[0, 0], final_y[1, 0], 'go', markerfacecolor='g', markersize=8)  # Start point
            ax.plot(final_y[0, -1], final_y[1, -1], 'ro', markerfacecolor='r', markersize=8)  # End point
            plt.savefig('velocity_Trajectory_plot.png', dpi=300)
            plt.show()

        except FileNotFoundError:
            print(f"Error: The file at {file_path} was not found.")
        except KeyError as e:
            print(f"Error: Missing expected column in the CSV file - {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    # Example usage
    # file_path = r'C:\Users\mmabo\V_Code\New folder\Aneurysm_filling\Excel_data_velocity_comsol\Normalized_Velocity_2d_5cm.csv'
    file_path = r'C:\Users\M4\VSCode_Projects\Ultrasound-Swarm-Microbubbles-Navigating-Vortices-to-Target-and-Fill-Aneurysms\Excel_data_velocity_comsol\Normalized_Velocity_2d_5cm.csv'

    read_and_plot_data(file_path)
