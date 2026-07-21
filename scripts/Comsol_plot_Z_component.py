import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path as _Path
REPO_ROOT = _Path(__file__).resolve().parents[1]


# Define the function to read and plot 2D velocity vectors
def read_and_plot_data_2d(file_path, skip_interval=2, scale_factor=70):
    try:
        # Define the column names
        column_names = ['x', 'y', 'z', 'u2', 'v2', 'w2']

        # Load data, skipping the metadata lines
        data = pd.read_csv(file_path, skiprows=0, names=column_names)  # Adjust skiprows as needed

        # Clean column names if necessary
        data.columns = data.columns.str.strip()

        # Convert columns to numeric, coercing errors to NaN
        data['x'] = pd.to_numeric(data['x'], errors='coerce')
        data['y'] = pd.to_numeric(data['y'], errors='coerce')
        data['u2'] = pd.to_numeric(data['u2'], errors='coerce')
        data['v2'] = pd.to_numeric(data['v2'], errors='coerce')

        # Drop rows with NaN values that couldn't be converted
        data.dropna(subset=['x', 'y', 'u2', 'v2'], inplace=True)

        # Convert x and y to micrometers
        data['x'] *= 1e6
        data['y'] *= 1e6

        # Convert velocity to cm/s
        data['u2'] *= 100
        data['v2'] *= 100

        # Extract data for plotting
        X = data['x'].values
        Y = data['y'].values
        U = data['u2'].values
        V = data['v2'].values

        # Skipping data points to reduce density
        X_skipped = X[::skip_interval]
        Y_skipped = Y[::skip_interval]
        U_skipped = U[::skip_interval]
        V_skipped = V[::skip_interval]

        # Calculate magnitudes for coloring
        magnitudes = np.sqrt(U_skipped**2 + V_skipped**2)

        # Normalize direction vectors for consistent arrow length
        magnitudes_non_zero = np.where(magnitudes == 0, 1e-10, magnitudes)
        U_normalized = U_skipped / magnitudes_non_zero
        V_normalized = V_skipped / magnitudes_non_zero

        # Use the original magnitudes for coloring
        norm = plt.Normalize(magnitudes.min(), magnitudes.max())
        colors = plt.cm.viridis(norm(magnitudes))

        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        quiver = ax.quiver(X_skipped, Y_skipped, U_normalized, V_normalized, color=colors, scale=scale_factor)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
        cbar.set_label('Velocity Magnitude (cm/s)')
        ax.set_title('2D Velocity Vector Plot with Minimized Arrows')
        ax.set_xlabel('X Coordinate (micrometers)')
        ax.set_ylabel('Y Coordinate (micrometers)')
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
file_path = REPO_ROOT / "data" / "comsol" / "Velocity_60_cm_Full_scaled.csv"  # Use the provided file path
read_and_plot_data_2d(file_path)
