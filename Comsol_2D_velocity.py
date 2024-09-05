import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_and_plot_data(file_path, skip_interval=1, scale_factor=80):
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

        # Convert x and y to micrometers
        data['x'] *= 1e6
        data['y'] *= 1e6

        # Convert velocity to cm/s
        data['u'] *= 100
        data['v'] *= 100
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
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except KeyError as e:
        print(f"Error: Missing expected column in the CSV file - {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage
# file_path = r'C:\Users\mmabo\V_Code\New folder\Aneurysm_filling\Velocity_2d_5cm.csv'  # Use a raw string for Windows paths
file_path = r'C:\Users\mmabo\V_Code\New folder\Aneurysm_filling\Excel_data_velocity_comsol\Normalized_60_cm.csv'  # Use a raw string for Windows paths
# file_path = r'C:\Users\mmabo\V_Code\New folder\Aneurysm_filling\Normalized_Velocity_2d_5cm.csv'  # Use a raw string for Windows paths



read_and_plot_data(file_path)
