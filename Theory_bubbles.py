import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
Gamma = 4.0 #6.5
rho = 1000
a = 0.05
CD = 100 # 1.0
p_inf = 1e5

# Define the velocity field
def velocity_field(r, theta):
    if r < a:
        u_theta = Gamma / (2 * np.pi) * (r / a**2)
    else:
        u_theta = Gamma / (2 * np.pi) * 1 / r
    return np.array([-u_theta * np.sin(theta), u_theta * np.cos(theta)])

# Define the pressure field
def pressure_field(r):
    if r < a:
        return p_inf + Gamma**2 / (4 * np.pi**2) * rho / a**2 + (rho * Gamma**2 * r**2) / (8 * np.pi**2 * a**2) 
    else:
        return p_inf + Gamma**2 / (8 * np.pi**2) * rho / r**2

# Define the pressure gradient
def pressure_gradient(x, y, r, theta):
    dr = 0.01  # finite difference for gradient approximation
    p_r_plus = pressure_field(r + dr)
    p_r_minus = pressure_field(r - dr)
    dp_dr = (p_r_plus - p_r_minus) / (2 * dr)
    grad_p = np.array([dp_dr * (x / r), dp_dr * (y / r)])
    return grad_p

# Dynamics function equivalent to MATLAB's microbubbleDynamics
def microbubble_dynamics(t, Y):
    x, y, u_MBx, u_MBy = Y
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    u = velocity_field(r, theta)
    grad_p = pressure_gradient(x, y, r, theta)
    
    # Differential equations for velocity
    dxdt = u_MBx
    dydt = u_MBy
    du_MBx_dt = (3 / rho) * grad_p[0] + 3/4 * CD * (u[0] - u_MBx) * abs(u[0] - u_MBx)
    du_MBy_dt = (3 / rho) * grad_p[1] + 3/4 * CD * (u[1] - u_MBy) * abs(u[1] - u_MBy)
    return [dxdt, dydt, du_MBx_dt, du_MBy_dt]

# Initial conditions
x0 = [-2, -2]
u_MB0 = [0, 0]
initial_conditions = x0 + u_MB0  # Combine lists

# Time span for the simulation
t_span = [0, 100]

# Solve the ODE
solution = solve_ivp(microbubble_dynamics, [t_span[0], t_span[1]], initial_conditions, method='RK45', dense_output=True)

# Plotting results
t_points = np.linspace(t_span[0], t_span[1], 100)
y = solution.sol(t_points)

# Plot velocity field
plt.figure(figsize=(10, 5))
u_plot = np.array([velocity_field(np.sqrt(y[0, i]**2 + y[1, i]**2), np.arctan2(y[1, i], y[0, i])) for i in range(len(t_points))])
plt.quiver(y[0], y[1], u_plot[:, 0], u_plot[:, 1], color='r')
plt.title('Velocity Field u Over Trajectory')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.grid(True, alpha=0.3)

# Plot pressure gradient
plt.figure(figsize=(10, 5))
grad_p_plot = np.array([pressure_gradient(y[0, i], y[1, i], np.sqrt(y[0, i]**2 + y[1, i]**2), np.arctan2(y[1, i], y[0, i])) for i in range(len(t_points))])
plt.quiver(y[0], y[1], grad_p_plot[:, 0], grad_p_plot[:, 1], color='b')
plt.title('Pressure Gradient Over Trajectory')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.grid(True, alpha=0.3)
plt.axis('equal')

# Plot trajectory
plt.figure(figsize=(10, 5))
plt.plot(y[0], y[1], 'b-', linewidth=2)
plt.plot(y[0, 0], y[1, 0], 'go', markerfacecolor='g', markersize=8)  # Start point
plt.plot(y[0, -1], y[1, -1], 'ro', markerfacecolor='r', markersize=8)  # End point
plt.title('Microbubble Trajectory in 2D')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.grid(True, alpha=0.3)
plt.axis('equal')

plt.show()
