# vortex_mb.py
from dataclasses import dataclass
import numpy as np
from numpy import pi
from scipy.integrate import solve_ivp


@dataclass
class Params:
    # Fluid & bubble
    rho: float = 1000.0        # fluid density [kg/m^3]
    mu: float = 1.0e-3         # dynamic viscosity [Pa·s]
    rho_bubble: float = 1.2    # bubble gas density [kg/m^3] (approx air)
    R: float = 5e-6            # bubble radius [m]
    Cd: float = 0.47           # quadratic drag coefficient (sphere ~0.47)
    use_quadratic_drag: bool = True  # True: quadratic drag; False: Stokes
    include_buoyancy: bool = False   # set True if gravity matters
    g: float = 9.81            # gravity [m/s^2]
    include_pressure_force: bool = True  # include -V ∇P in full mode

    # Vortex (Lamb–Oseen core-like piecewise model you specified)
    Gamma: float = 1e-4        # circulation [m^2/s]
    a: float = 1e-3            # vortex core radius [m]
    eps_r0: float = 1e-12      # small epsilon to avoid r=0 singularity

    # Acoustic (simple standing-wave placeholder for primary radiation)
    include_radiation: bool = False
    P0: float = 1e4            # acoustic pressure amplitude [Pa]
    k_ac: float = 200.0        # acoustic wavenumber [1/m]
    omega: float = 2*pi*100e3  # acoustic angular frequency [rad/s]
    k_rad: float = 1.0/(2*1000.0*(1500.0**2))  # ~1/(2 ρ c^2) ≈ 1/(2ρc^2) [1/Pa]
    # NOTE: k_rad is a “lumped” factor to shape F_rad ~ -k_rad ∇⟨p_a^2⟩; tune as needed.

    # Secondary Bjerknes (stub; set nonzero only if modeling multi-bubble interactions)
    include_bjerknes: bool = False
    FB_const: float = 0.0  # placeholder magnitude [N], set via custom function if needed

    # Integration control
    mode: str = "reduced"   # "reduced" (your original), or "full"


# -------------------------
# Vortex kinematics & pressure
# -------------------------

def vortex_velocity(xy: np.ndarray, p: Params) -> np.ndarray:
    x, y = xy
    r = np.hypot(x, y)
    if r < p.eps_r0:
        return np.array([0.0, 0.0])

    # piecewise u_theta
    if r < p.a:
        u_theta = p.Gamma/(2*pi) * (r/p.a**2)
    else:
        u_theta = p.Gamma/(2*pi) * (1.0/r)

    # polar -> Cartesian: u = (-uθ sinθ, uθ cosθ) = (-uθ*y/r, uθ*x/r)
    return np.array([-u_theta * (y/r), u_theta * (x/r)])


def pressure_gradient(xy: np.ndarray, p: Params) -> np.ndarray:
    """
    ∇p for the piecewise pressure you provided:
    r < a:  p = p∞ - (Γ^2 ρ)/(4π^2 a^2) + (ρ Γ^2 / (8π^2)) (r^2 / a^2)
            ⇒ ∂p/∂r = (ρ Γ^2 / (4π^2 a^2)) r
            ⇒ ∇p = (ρ Γ^2 / (4π^2 a^2)) [x, y]
    r > a:  p = p∞ - (Γ^2 ρ)/(8π^2 r^2)
            ⇒ ∂p/∂r = (ρ Γ^2)/(4π^2 r^3)
            ⇒ ∇p = (ρ Γ^2)/(4π^2 r^4) [x, y]
    """
    x, y = xy
    r2 = x*x + y*y
    r = np.sqrt(r2 + 1e-300)  # safe sqrt
    rho, Gamma, a = p.rho, p.Gamma, p.a

    if r < a:
        coeff = rho * Gamma**2 / (4 * pi**2 * a**2)
        return coeff * np.array([x, y])
    else:
        coeff = rho * Gamma**2 / (4 * pi**2 * (r**4 + 1e-300))
        return coeff * np.array([x, y])


# -------------------------
# Acoustic radiation (simple, tunable)
# -------------------------

def acoustic_p_sq_avg(xy: np.ndarray, t: float, p: Params) -> float:
    """
    Standing-wave model along x: p_a(x,t) = 2 P0 cos(k x) cos(ω t)
    Time-average of p_a^2: <p_a^2> = P0^2 cos^2(k x).
    """
    x, _ = xy
    return (p.P0**2) * (np.cos(p.k_ac * x)**2)


def F_radiation(xy: np.ndarray, t: float, p: Params) -> np.ndarray:
    if not p.include_radiation:
        return np.zeros(2)
    # F_rad ≈ -k_rad ∇⟨p_a^2⟩
    x, y = xy
    dp2dx = -2 * p.P0**2 * p.k_ac * np.cos(p.k_ac * x) * np.sin(p.k_ac * x)  # d/dx cos^2 = -sin(2kx) * k
    return -p.k_rad * np.array([dp2dx, 0.0])


# -------------------------
# Other forces for "full" mode
# -------------------------

def F_drag(u_fluid: np.ndarray, u_b: np.ndarray, p: Params) -> np.ndarray:
    rel = u_fluid - u_b
    speed = np.linalg.norm(rel)
    if speed < 1e-30:
        return np.zeros(2)
    if p.use_quadratic_drag:
        A = pi * p.R**2
        return 0.5 * p.rho * p.Cd * A * speed * rel
    else:
        # Stokes (valid at very low Re): 6πμR (u - v)
        return 6 * pi * p.mu * p.R * rel


def F_buoyancy(p: Params) -> np.ndarray:
    if not p.include_buoyancy:
        return np.zeros(2)
    V = (4/3) * pi * p.R**3
    # Upward along +y
    return np.array([0.0, (p.rho - p.rho_bubble) * p.g * V])


def F_neg_gradP(xy: np.ndarray, p: Params) -> np.ndarray:
    if not p.include_pressure_force:
        return np.zeros(2)
    V = (4/3) * pi * p.R**3
    return -V * pressure_gradient(xy, p)


def F_bjerknes_stub(p: Params) -> np.ndarray:
    # Placeholder; supply your own multi-bubble model if needed
    if not p.include_bjerknes or p.FB_const == 0.0:
        return np.zeros(2)
    # Example: constant pull along +x (purely illustrative)
    return np.array([p.FB_const, 0.0])


# -------------------------
# ODE right-hand sides
# -------------------------

def rhs_reduced(t: float, state: np.ndarray, p: Params) -> np.ndarray:
    x, y, ux, uy = state
    xy = np.array([x, y])
    u_fluid = vortex_velocity(xy, p)
    grad_p = pressure_gradient(xy, p)
    u_b = np.array([ux, uy])
    # Your reduced model:
    acc = (3.0/p.rho) * grad_p + 0.75 * p.Cd * (u_fluid - u_b) * np.linalg.norm(u_fluid - u_b)
    return np.array([ux, uy, acc[0], acc[1]])


def rhs_full(t: float, state: np.ndarray, p: Params) -> np.ndarray:
    x, y, ux, uy = state
    xy = np.array([x, y])
    u_fluid = vortex_velocity(xy, p)
    u_b = np.array([ux, uy])

    # Effective mass (bubble + added mass of surrounding fluid)
    V = (4/3) * pi * p.R**3
    m_b = p.rho_bubble * V
    m_added = 0.5 * p.rho * V
    m_eff = m_b + m_added

    # Forces
    F_total = np.zeros(2)
    F_total += F_drag(u_fluid, u_b, p)
    F_total += F_buoyancy(p)
    F_total += F_neg_gradP(xy, p)
    F_total += F_radiation(xy, t, p)
    F_total += F_bjerknes_stub(p)

    acc = F_total / (m_eff + 1e-300)
    return np.array([ux, uy, acc[0], acc[1]])


def integrate_mb(
    p: Params,
    x0: float, y0: float,
    ux0: float, uy0: float,
    t_span=(0.0, 0.5),
    atol=1e-9, rtol=1e-7, max_step=np.inf
):
    state0 = np.array([x0, y0, ux0, uy0])
    if p.mode == "reduced":
        fun = lambda t, s: rhs_reduced(t, s, p)
    elif p.mode == "full":
        fun = lambda t, s: rhs_full(t, s, p)
    else:
        raise ValueError("Params.mode must be 'reduced' or 'full'")

    sol = solve_ivp(fun, t_span, state0, atol=atol, rtol=rtol, max_step=max_step, dense_output=True)
    return sol


# ------------- quick demo -------------
if __name__ == "__main__":
    p = Params(
        rho=1000.0, mu=1e-3, rho_bubble=1.2,
        R=5e-6, Cd=0.47,
        Gamma=1e-4, a=1e-3,
        mode="reduced",  # switch to "full" to include forces
        include_radiation=False,
        include_buoyancy=False
    )

    # Start slightly off-center, zero initial velocity
    sol = integrate_mb(p, x0=2e-4, y0=0.0, ux0=0.0, uy0=0.0, t_span=(0, 0.1))
    print(f"Integration success: {sol.success}, message: {sol.message}")
    print(f"Final state at t={sol.t[-1]:.5f} s:", sol.y[:, -1])
