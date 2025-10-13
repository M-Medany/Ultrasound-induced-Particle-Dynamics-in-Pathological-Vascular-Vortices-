import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_and_plot_data(file_path, skip_interval=2, scale_factor=80,
                       position_scale=1e6, velocity_scale=100.0):
    try:
        # Load data
        data = pd.read_csv(file_path)

        # Clean column names if necessary and accept common aliases
        data.columns = data.columns.str.strip()
        lower = {c.lower(): c for c in data.columns}
        def pick(*candidates):
            for cand in candidates:
                if cand in lower:
                    return lower[cand]
            return None

        x_col = pick('x', 'pos_x', 'position x (m)', 'x (m)')
        y_col = pick('y', 'pos_y', 'position y (m)', 'y (m)')
        u_col = pick('u', 'ux', 'vx', 'u2')
        v_col = pick('v', 'uy', 'vy', 'v2')
        if None in (x_col, y_col, u_col, v_col):
            missing = [name for name, col in zip(['x', 'y', 'u', 'v'], (x_col, y_col, u_col, v_col)) if col is None]
            raise KeyError(missing[0])

        # Convert into display units before extracting arrays
        data[x_col] = data[x_col].astype(float) * position_scale
        data[y_col] = data[y_col].astype(float) * position_scale
        data[u_col] = data[u_col].astype(float) * velocity_scale
        data[v_col] = data[v_col].astype(float) * velocity_scale

        # Extract data for plotting (already scaled)
        X = data[x_col].to_numpy()
        Y = data[y_col].to_numpy()
        U = data[u_col].to_numpy()
        V = data[v_col].to_numpy()
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
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        quiver = ax.quiver(X_skipped, Y_skipped, U_normalized, V_normalized, color=colors, scale=scale_factor)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
        cbar.set_label('Velocity Magnitude')
        ax.set_title('Velocity Vector Plot with Minimized Arrows')
        pos_unit = 'µm' if np.isclose(position_scale, 1e6) else ''
        if pos_unit:
            ax.set_xlabel(f'X Coordinate ({pos_unit})')
            ax.set_ylabel(f'Y Coordinate ({pos_unit})')
        else:
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
        ax.axis('equal')
        # ax.grid(True, alpha=0.3)
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except KeyError as e:
        print(f"Error: Missing expected column in the CSV file - {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage
# file_path = r'C:\Users\mmabo\V_Code\New folder\Aneurysm_filling\Velocity_2d_5cm.csv'  # Use a raw string for Windows paths
# file_path = r'C:\Users\mmabo\V_Code\New folder\Aneurysm_filling\Excel_data_velocity_comsol\Normalized_60_cm.csv'  # Use a raw string for Windows paths
file_path = r'C:\Users\M4\VSCode_Projects\Ultrasound-Swarm-Microbubbles-Navigating-Vortices-to-Target-and-Fill-Aneurysms\Excel_data_velocity_comsol\Normalized_Velocity_60_cm_Full.csv'

# file_path = r'C:\Users\mmabo\V_Code\New folder\Aneurysm_filling\Normalized_Velocity_2d_5cm.csv'  # Use a raw string for Windows paths



read_and_plot_data(file_path)
