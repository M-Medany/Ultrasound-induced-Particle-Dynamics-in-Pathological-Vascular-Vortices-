import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the CSV file
df = pd.read_csv('Excel_data/PIV_Velocity_Vortex_center.csv')
df.columns = ['Distance', 'Velocity', 'StdDev']

# Set up the 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Data for plotting
distance = df['Distance']
velocity = df['Velocity']
std_dev = df['StdDev']

# Plotting points with error bars along the z-axis (Velocity)
for i in range(len(distance)):
    # Plotting each point and its standard deviation
    ax.errorbar(distance[i], 25, velocity[i], zerr=std_dev[i], fmt='o', color='blue', ecolor=(1, 0.8, 0.8), label='Velocity data' if i == 0 else "")

# Labeling
ax.set_xlabel('Distance')
ax.set_ylabel('Fixed Value (0 to 50, centered at 25)')
ax.set_zlabel('Velocity')

# Set Y-axis limits to represent the span from 0 to 50
ax.set_ylim([0, 50])

# Title and legend
ax.set_title('3D Visualization of Velocity by Distance with Std Deviation at Central Y=25')
ax.legend()

# Set the view angle for better viewing perspective
ax.view_init(elev=10, azim=240)  # Adjust these angles as needed to get the best view

# Show the plot
plt.show()


