import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_and_plot_data(file_path):
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
    skip_interval = 20  # Adjust as needed to reduce arrow density
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

    # Scale for visibility, adjust the scale factor as needed
    scale_factor = 100  # Adjust based on your specific visualization needs to shorten arrows

    # Create the plot
    plt.figure(figsize=(10, 8), dpi=300)
    plt.quiver(X_skipped, Y_skipped, U_normalized, V_normalized, scale=scale_factor)
    plt.title('Velocity Vector Plot with Minimized Arrows')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.axis('equal')
    plt.show()

# Example usage
file_path = r'C:\Users\mmabo\V_Code\New folder\Aneurysm_filling\Velocity_2d_5cm.csv'  # Use a raw string for Windows paths
read_and_plot_data(file_path)
