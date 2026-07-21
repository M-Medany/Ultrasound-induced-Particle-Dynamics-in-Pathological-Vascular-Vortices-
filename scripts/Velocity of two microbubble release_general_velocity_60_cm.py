# ================================================================
# two_mb_center_swirlstrong_autoroi_FAST.py
# - Center by swirling strength (λ_ci) in an auto-expanding ROI
# - Rankine (a, Γ) calibrated inside ROI + edge-aware clipping
# - Time horizon = N_SWIRL_PERIODS * T_swirl  <<<<< key freeze fix
# - Plots in µm/cm·s⁻¹ and saves diagnostic_center.png
# ================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
from multiprocessing import get_context, cpu_count
from PIL import ImageFile

from pathlib import Path as _Path
REPO_ROOT = _Path(__file__).resolve().parents[1]

ImageFile.MAXBLOCK = 1 << 24

# ===================== USER CONFIG =====================

CSV_PATH = REPO_ROOT / "data" / "comsol" / "Normalized_Velocity_60_cm_Full.csv"

# CSV units & normalization
CSV_POS_UNITS = 'm'          # {'m','mm','um'}
CSV_VEL_UNITS = 'auto'       # {'m/s','cm/s','mm/s','unitless','auto'}
AUTOSCALE_TARGET_MAX_CM_S = 60.0
AUTOSCALE_REGION = 'global'  # or 'inlet'

# Axes / colorbar units
DISPLAY_POS_UNITS = 'um'
DISPLAY_VEL_UNITS = 'cm/s'

# Starting ROI near the swirl (µm). Will auto-expand if sparse.
CENTER_ROI_UM = (80.0, 180.0, 110.0, 190.0)  # xlo, xhi, ylo, yhi  ← aims near (x≈110,y≈140)
ROI_MIN_POINTS = 400
ROI_EXPAND_STEP_UM = 20.0
ROI_MAX_EXPANDS = 10
Y_EXCLUDE_BELOW_FRAC = 0.25   # used only for global fallback

# Rankine core clipping
A_EDGE_FRACTION_MAX = 0.50    # tighter core
A_EDGE_FRACTION_MIN = 0.05

# Releases (absolute µm near the swirl)
MB1_START_ABS_UM = (100.0, 120.0)
MB2_START_ABS_UM = (140.0, 160.0)

# Physics (SI)
rho = 1000.0
CD  = 4.0             # slightly lower drag → easier spiral-in
PRESSURE_GAIN = 1.30  # >1 strengthens inward -∇p
KNN_K = 12

# ---------- TIME HORIZON (freeze fix) ----------
# Integrate for a few swirl periods instead of millions of seconds
N_SWIRL_PERIODS = 8          # total horizon = N_SWIRL_PERIODS * T_swirl
MAX_SWIRL_STEPS_PER_PERIOD = 60
# -----------------------------------------------

# Solver tolerances
rtol, atol = 1e-6, 1e-9

# Plotting
FIGSIZE = (8,5)
DPI_STATIC = 600
TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 14
TICK_FONTSIZE  = 12
LEGEND_FONTSIZE= 11
skip_interval  = 1
quiver_scale   = 80
SHOW_GRID = True

# ===================== GLOBALS (workers) =====================
TREE=None; UVAL=None; VVAL=None
XC=None; YC=None; A=None
GAM=None; RHO=None; CDG=None
RTOL=None; ATOL=None
XMIN=None; XMAX=None; YMIN=None; YMAX=None
MAX_STEP_LIMIT=np.inf
NPTS=0

# ===================== Unit helpers ====================
def L_to_m(u): return {'m':1.0,'mm':1e-3,'um':1e-6}[u]
def m_to_L(u): return {'m':1.0,'mm':1e3,'um':1e6}[u]
def to_mps(u): return {'m/s':1.0,'cm/s':1e-2,'mm/s':1e-3}[u]
def mps_to(u): return {'m/s':1.0,'cm/s':100.0,'mm/s':1000.0}[u]

# ===================== CSV reading =====================
def standardize_columns(df):
    df.columns = df.columns.str.strip()
    lower = {c.lower(): c for c in df.columns}
    def pick(*cands):
        for c in cands:
            if c in lower: return lower[c]
        return None
    x = pick('x','pos_x','position x (m)','x (m)','x_mm')
    y = pick('y','pos_y','position y (m)','y (m)','y_mm')
    u = pick('u','ux','vx','u2')
    v = pick('v','uy','vy','v2')
    if None in (x,y,u,v): raise KeyError("CSV must have x,y,u,v (common aliases allowed).")
    return df.rename(columns={x:'x',y:'y',u:'u',v:'v'})

def read_csv_field(path):
    raw = pd.read_csv(path); raw = standardize_columns(raw)
    x = raw['x'].astype(float)*L_to_m(CSV_POS_UNITS)
    y = raw['y'].astype(float)*L_to_m(CSV_POS_UNITS)
    if CSV_VEL_UNITS in ('m/s','cm/s','mm/s'):
        u = raw['u'].astype(float)*to_mps(CSV_VEL_UNITS)
        v = raw['v'].astype(float)*to_mps(CSV_VEL_UNITS)
    else:
        u = raw['u'].astype(float).to_numpy()
        v = raw['v'].astype(float).to_numpy()
        s = np.hypot(u,v)
        looks_unitless = (np.nanmax(s) <= 1.5) or (np.nanmedian(s) < 1e-2)
        if looks_unitless or CSV_VEL_UNITS in ('unitless','auto'):
            y_sorted=np.sort(y)
            if AUTOSCALE_REGION=='inlet' and len(y_sorted)>0:
                y_cut=y_sorted[int(0.10*len(y_sorted))]; region=(y<=y_cut)
            else:
                region=np.isfinite(s)
            ref=np.nanpercentile(s[region],99.5); ref = np.nanmax(s) if not np.isfinite(ref) or ref<=0 else ref
            scale=(AUTOSCALE_TARGET_MAX_CM_S/100.0)/ref
            u*=scale; v*=scale
            print(f"[Autoscale] normalized CSV → {scale:.6g} m/s per unit (99.5%ile → {AUTOSCALE_TARGET_MAX_CM_S} cm/s)")
    vf = pd.DataFrame({'x':x,'y':y,'u':u,'v':v}).drop_duplicates(subset=['x','y'])
    spd = np.hypot(vf['u'],vf['v'])
    print(f"Speed stats (SI): min={spd.min():.3g} m/s, med={np.median(spd):.3g} m/s, max={spd.max():.3g} m/s")
    return vf

# ===================== ROI helpers =====================
def clamp_roi_to_domain(roi_m, xmin, xmax, ymin, ymax):
    xlo,xhi,ylo,yhi = roi_m
    xlo = max(xmin, min(xlo, xmax)); xhi = max(xmin, min(xhi, xmax))
    ylo = max(ymin, min(ylo, ymax)); yhi = max(ymin, min(yhi, ymax))
    return (min(xlo,xhi), max(xlo,xhi), min(ylo,yhi), max(ylo,yhi))

def ensure_roi_points(roi_m, X, Y, xmin, xmax, ymin, ymax,
                      min_pts=400, step_m=20e-6, max_expands=10):
    xlo,xhi,ylo,yhi = roi_m
    for e in range(max_expands+1):
        sel = (X>=xlo) & (X<=xhi) & (Y>=ylo) & (Y<=yhi)
        n = int(sel.sum())
        if n >= min_pts:
            if e>0: print(f"[ROI] Expanded {e}× → {n} points.")
            return (xlo,xhi,ylo,yhi), sel, True
        xlo = max(xmin, xlo - step_m); xhi = min(xmax, xhi + step_m)
        ylo = max(ymin, ylo - step_m); yhi = min(ymax, yhi + step_m)
    sel = (X>=xlo) & (X<=xhi) & (Y>=ylo) & (Y<=yhi)
    print(f"[ROI] Max expansions reached; points in ROI: {int(sel.sum())}")
    return (xlo,xhi,ylo,yhi), sel, (int(sel.sum()) >= min_pts)

# ===================== Swirling-strength center =====================
def center_from_swirlstrength(df, idx_sel, k=12):
    X = df['x'].to_numpy(float)[idx_sel]
    Y = df['y'].to_numpy(float)[idx_sel]
    U = df['u'].to_numpy(float)[idx_sel]
    V = df['v'].to_numpy(float)[idx_sel]
    pts = np.column_stack([X,Y]); tree = cKDTree(pts)
    k = min(k, pts.shape[0])
    if pts.shape[0] < 80:
        return float(np.median(X)), float(np.median(Y))

    lam_ci = np.zeros(pts.shape[0])
    for i in range(pts.shape[0]):
        d, nb = tree.query(pts[i], k=k)
        dx = pts[nb,0]-pts[i,0]; dy = pts[nb,1]-pts[i,1]
        du = U[nb]-U[i];         dv = V[nb]-V[i]
        A  = np.column_stack([dx,dy]); w = 1.0/(d+1e-12); W=np.diag(w)
        try:
            gu = np.linalg.lstsq(W@A, W@du, rcond=None)[0]  # [ux, uy]
            gv = np.linalg.lstsq(W@A, W@dv, rcond=None)[0]  # [vx, vy]
            ux,uy = gu[0], gu[1]; vx,vy = gv[0], gv[1]
            tr = ux + vy
            det = ux*vy - uy*vx
            disc = det - 0.25*(tr*tr)
            lam_ci[i] = np.sqrt(max(disc, 0.0))
        except Exception:
            lam_ci[i] = 0.0

    speed = np.hypot(U, V)
    s_ref = max(np.percentile(speed, 85), 1e-6)
    weight = lam_ci * np.exp(-(speed/s_ref)**2)  # suppress pure shear

    q = np.percentile(weight[weight>0], 95) if np.any(weight>0) else 0
    core = weight >= q
    if core.sum() < 50:
        q = np.percentile(weight[weight>0], 98) if np.any(weight>0) else 0
        core = weight >= q
    if core.sum() == 0:
        i0 = int(np.argmax(weight)); return float(X[i0]), float(Y[i0])

    w = weight[core] + 1e-12
    xc = float(np.average(X[core], weights=w))
    yc = float(np.average(Y[core], weights=w))
    return xc, yc

def estimate_center_global(df, y_exclude_frac=0.25):
    X = df["x"].to_numpy(float); Y = df["y"].to_numpy(float)
    U = df["u"].to_numpy(float); V = df["v"].to_numpy(float)
    xmin,xmax = X.min(), X.max(); ymin,ymax = Y.min(), Y.max()
    mask = np.ones_like(X, dtype=bool)
    if y_exclude_frac is not None and 0 < y_exclude_frac < 1:
        ythr = ymin + y_exclude_frac*(ymax - ymin)
        mask &= (Y >= ythr)
    X = X[mask]; Y = Y[mask]; U = U[mask]; V = V[mask]
    spd = np.hypot(U, V)
    if X.size < 200:
        j = int(np.argmin(spd)); return float(X[j]), float(Y[j])
    Umag = spd + 1e-12; ux, uy = U/Umag, V/Umag
    xs = np.linspace(np.percentile(X,10), np.percentile(X,90), 60)
    ys = np.linspace(np.percentile(Y,10), np.percentile(Y,90), 60)
    R0 = 0.35 * min(X.ptp(), Y.ptp())
    best_cost, best_xy = 1e99, (np.median(X), np.median(Y))
    for xc in xs:
        dx = X - xc
        for yc in ys:
            dy = Y - yc
            r  = np.hypot(dx,dy) + 1e-12
            rx, ry = dx/r, dy/r
            tx, ty = -ry, rx
            ur = ux*rx + uy*ry
            ut = ux*tx + uy*ty
            w  = np.exp(-(r*r)/(2*R0*R0))
            cost = (w*(ur*ur)).sum()/w.sum() + 0.2*(w*(1.0-np.abs(ut))).sum()/w.sum()
            if cost < best_cost:
                best_cost, best_xy = cost, (xc, yc)
    return float(best_xy[0]), float(best_xy[1])

# ===================== Rankine calibration =====================
def estimate_rankine_in_selection(df, xc, yc, idx_sel,
                                  a_edge_frac_max=0.50, a_edge_frac_min=0.05):
    X=df['x'].to_numpy(float)[idx_sel]; Y=df['y'].to_numpy(float)[idx_sel]
    U=df['u'].to_numpy(float)[idx_sel]; V=df['v'].to_numpy(float)[idx_sel]
    if X.size < 200:  # fallback
        X=df['x'].to_numpy(float); Y=df['y'].to_numpy(float)
        U=df['u'].to_numpy(float); V=df['v'].to_numpy(float)
    dx = X - xc; dy = Y - yc; r = np.hypot(dx,dy); rs = np.where(r<1e-12,1e-12,r)
    rx,ry = dx/rs, dy/rs
    vt = -U*ry + V*rx

    nbins=48
    bins = np.linspace(r.min(), r.max(), nbins+1)
    which = np.digitize(r, bins) - 1
    vt_abs_med = np.full(nbins,np.nan); r_med = np.full(nbins,np.nan)
    for i in range(nbins):
        m = (which==i)
        if m.sum()>=20:
            vt_abs_med[i]=np.median(np.abs(vt[m])); r_med[i]=np.median(r[m])
    i_peak = int(np.nanargmax(vt_abs_med))
    a_raw  = float(r_med[i_peak]) if np.isfinite(r_med[i_peak]) else float(np.median(r))
    vt_a   = float(vt_abs_med[i_peak]) if np.isfinite(vt_abs_med[i_peak]) else float(np.median(np.abs(vt)))

    xmin,xmax = df['x'].min(), df['x'].max()
    ymin,ymax = df['y'].min(), df['y'].max()
    r_edge = min(xc-xmin, xmax-xc, yc-ymin, ymax-yc)
    a_min = a_edge_frac_min*r_edge; a_max = a_edge_frac_max*r_edge
    a_est = float(np.clip(a_raw, a_min, a_max))

    shell = (r>=0.9*a_est) & (r<=1.1*a_est)
    if shell.sum()>=40:
        vt_a = float(np.median(np.abs(vt[shell])))

    r_outer_hi = min(2.5*a_est, 0.95*r_edge)
    outer = (r > 1.15*a_est) & (r < r_outer_hi)
    if outer.sum() < 80:
        outer = (r > a_est) & (r < r_outer_hi)
    gamma_vals = 2*np.pi*r[outer]*vt[outer]
    if gamma_vals.size==0: gamma_vals = 2*np.pi*r*vt
    Gamma_est = float(np.median(gamma_vals))
    T_swirl = 2*np.pi*a_est / max(vt_a, 1e-12)
    return a_est, Gamma_est, T_swirl

# ===================== Dynamics =============================
def kNN_velocity_field(x,y,k):
    d, idx = TREE.query((x,y), k=min(k,NPTS))
    if np.isscalar(d): d=np.array([d]); idx=np.array([idx])
    w = 1.0/(d+1e-12); w/=w.sum()
    u=np.dot(w,UVAL[idx]); v=np.dot(w,VVAL[idx])
    return np.array([u,v])

def dp_dr_rankine(r, Gamma, rho, a):
    if r < a: return (rho*Gamma**2)/(4*np.pi**2) * (r/(a*a))
    return (rho*Gamma**2)/(4*np.pi**2) * (1.0/(r**3))

def grad_p_rankine(x,y,xc,yc,Gamma,rho,a):
    xr,yr=x-xc,y-yc; r=np.hypot(xr,yr)+1e-15
    dpr=dp_dr_rankine(r,Gamma,rho,a)
    return dpr*np.array([xr/r, yr/r])

def rhs(t,Y):
    x1,y1,u1,v1,x2,y2,u2,v2=Y
    uf1=kNN_velocity_field(x1,y1,KNN_K); g1=grad_p_rankine(x1,y1,XC,YC,GAM,RHO,A)
    du1=uf1-np.array([u1,v1]); du1m=np.hypot(*du1)
    ax1 = -PRESSURE_GAIN*(3.0/RHO)*g1[0] + 0.75*CDG*du1[0]*du1m
    ay1 = -PRESSURE_GAIN*(3.0/RHO)*g1[1] + 0.75*CDG*du1[1]*du1m
    uf2=kNN_velocity_field(x2,y2,KNN_K); g2=grad_p_rankine(x2,y2,XC,YC,GAM,RHO,A)
    du2=uf2-np.array([u2,v2]); du2m=np.hypot(*du2)
    ax2 = -PRESSURE_GAIN*(3.0/RHO)*g2[0] + 0.75*CDG*du2[0]*du2m
    ay2 = -PRESSURE_GAIN*(3.0/RHO)*g2[1] + 0.75*CDG*du2[1]*du2m
    return [u1,v1,ax1,ay1,u2,v2,ax2,ay2]

def out_of_bounds(t,Y):
    x1,y1=Y[0],Y[1]; x2,y2=Y[4],Y[5]
    return min(x1-XMIN, XMAX-x1, y1-YMIN, YMAX-y1, x2-XMIN, XMAX-x2, y2-YMIN, YMAX-y2)
out_of_bounds.terminal=True; out_of_bounds.direction=-1

def reach_core(t,Y):
    r1=np.hypot(Y[0]-XC,Y[1]-YC); r2=np.hypot(Y[4]-XC,Y[5]-YC)
    return min(r1,r2) - 0.15*A
reach_core.terminal=True; reach_core.direction=-1

# ===================== Diagnostics =====================
def save_center_diagnostic(vf, roi_m, xc, yc, a):
    X=vf['x'].to_numpy(float); Y=vf['y'].to_numpy(float)
    U=vf['u'].to_numpy(float); V=vf['v'].to_numpy(float)
    pts=np.column_stack([X,Y]); tree=cKDTree(pts); k=min(10,pts.shape[0])
    lam=np.zeros(pts.shape[0])
    for i in range(0,pts.shape[0],10):
        d,idx=tree.query(pts[i],k=k)
        dx=pts[idx,0]-pts[i,0]; dy=pts[idx,1]-pts[i,1]
        du=U[idx]-U[i]; dv=V[idx]-V[i]
        A=np.column_stack([dx,dy]); w=1.0/(d+1e-12); W=np.diag(w)
        try:
            gu=np.linalg.lstsq(W@A, W@du, rcond=None)[0]
            gv=np.linalg.lstsq(W@A, W@dv, rcond=None)[0]
            ux,uy=gu[0],gu[1]; vx,vy=gv[0],gv[1]
            tr=ux+vy; det=ux*vy-uy*vx; disc=det-0.25*(tr*tr)
            lam[i]=np.sqrt(max(disc,0.0))
        except: lam[i]=0.0
    DISP_L=m_to_L(DISPLAY_POS_UNITS)
    fig,ax=plt.subplots(figsize=(6,5),dpi=220)
    sc=ax.scatter((X[::10]*DISP_L),(Y[::10]*DISP_L),c=lam[::10],cmap='magma',s=6)
    plt.colorbar(sc,ax=ax,label='swirling strength λ_ci (relative)')
    rect = plt.Rectangle((roi_m[0]*DISP_L,roi_m[2]*DISP_L),
                         (roi_m[1]-roi_m[0])*DISP_L,(roi_m[3]-roi_m[2])*DISP_L,
                         ec='cyan',fc='none',lw=2,ls='--')
    ax.add_patch(rect)
    ax.add_patch(plt.Circle((xc*DISP_L,yc*DISP_L),a*DISP_L,ec='k',fc='none',lw=2))
    ax.plot([xc*DISP_L],[yc*DISP_L],'wx',ms=8,mew=2)
    ax.set_aspect('equal'); ax.set_title('Center diagnostic (ROI & λ_ci)')
    ax.set_xlabel(f'X ({DISPLAY_POS_UNITS})'); ax.set_ylabel(f'Y ({DISPLAY_POS_UNITS})')
    fig.tight_layout(); fig.savefig('diagnostic_center.png'); plt.close(fig)

# ===================== MAIN =====================
def main():
    global TREE,UVAL,VVAL,XC,YC,A,GAM,RHO,CDG,RTOL,ATOL,XMIN,XMAX,YMIN,YMAX,MAX_STEP_LIMIT,NPTS

    print(f"Reading velocity data ...\nCSV: {CSV_PATH}")
    vf = read_csv_field(CSV_PATH)  # SI (m, m/s)
    points=vf[['x','y']].to_numpy(float); u=vf['u'].to_numpy(float); v=vf['v'].to_numpy(float)
    XMIN,XMAX=points[:,0].min(), points[:,0].max(); YMIN,YMAX=points[:,1].min(), points[:,1].max()
    print(f"Domain (SI): x∈[{XMIN:.6g},{XMAX:.6g}] m, y∈[{YMIN:.6g},{YMAX:.6g}] m")

    # ROI (m) and auto-expand
    roi_m_init = (CENTER_ROI_UM[0]*1e-6, CENTER_ROI_UM[1]*1e-6,
                  CENTER_ROI_UM[2]*1e-6, CENTER_ROI_UM[3]*1e-6)
    roi_m_init = clamp_roi_to_domain(roi_m_init, XMIN, XMAX, YMIN, YMAX)
    roi_m, idx_sel, ok = ensure_roi_points(
        roi_m_init, vf['x'].to_numpy(float), vf['y'].to_numpy(float),
        XMIN, XMAX, YMIN, YMAX,
        min_pts=ROI_MIN_POINTS, step_m=ROI_EXPAND_STEP_UM*1e-6, max_expands=ROI_MAX_EXPANDS
    )
    if not ok:
        print("[ROI] Sparse after expansion → using global center then rebuilding ROI.")
        xc_glob, yc_glob = estimate_center_global(vf, y_exclude_frac=Y_EXCLUDE_BELOW_FRAC)
        d = 60e-6
        roi_m = clamp_roi_to_domain((xc_glob-d, xc_glob+d, yc_glob-d, yc_glob+d),
                                    XMIN, XMAX, YMIN, YMAX)
        roi_m, idx_sel, _ = ensure_roi_points(
            roi_m, vf['x'].to_numpy(float), vf['y'].to_numpy(float),
            XMIN, XMAX, YMIN, YMAX,
            min_pts=ROI_MIN_POINTS, step_m=ROI_EXPAND_STEP_UM*1e-6, max_expands=ROI_MAX_EXPANDS
        )

    # Center from swirling strength inside ROI
    XC, YC = center_from_swirlstrength(vf, idx_sel, k=KNN_K)
    print(f"[Center (λ_ci, ROI)] (xc,yc)=({XC*1e6:.1f},{YC*1e6:.1f}) µm")

    # Calibrate a & Γ in selection
    A, GAM, T_swirl = estimate_rankine_in_selection(
        vf, XC, YC, idx_sel,
        a_edge_frac_max=A_EDGE_FRACTION_MAX,
        a_edge_frac_min=A_EDGE_FRACTION_MIN
    )
    print(f"[Calibrated in ROI] a≈{A*1e6:.1f} µm, Γ≈{GAM:.6g} m^2/s, Tθ≈{T_swirl:.4g} s")

    # ---- TIME HORIZON: a few swirl periods (freeze fix) ----
    t0 = 0.0
    t_final = max(T_swirl * N_SWIRL_PERIODS, T_swirl)  # at least one period
    MAX_STEP_LIMIT = T_swirl / MAX_SWIRL_STEPS_PER_PERIOD
    print(f"[Dynamics] CD={CD}, PRESSURE_GAIN={PRESSURE_GAIN}, "
          f"max_step≤{MAX_STEP_LIMIT:.3g}s, horizon={t_final:.4g}s "
          f"(~{N_SWIRL_PERIODS} periods)")

    # Build KD-tree for workers
    TREE = cKDTree(points); UVAL=u; VVAL=v
    RHO = rho; CDG = CD
    RTOL = rtol; ATOL = atol
    NPTS = int(points.shape[0])

    # Integrate in one process (simpler + avoids pool overhead on Windows)
    # (You can switch to multiprocessing later if needed.)
    Y0 = [MB1_START_ABS_UM[0]*1e-6, MB1_START_ABS_UM[1]*1e-6, 0.0, 0.0,
          MB2_START_ABS_UM[0]*1e-6, MB2_START_ABS_UM[1]*1e-6, 0.0, 0.0]

    def events(t, Y):
        return [out_of_bounds(t, Y), reach_core(t, Y)]
    events.terminal = True  # not used by solve_ivp, but kept for clarity

    sol = solve_ivp(rhs, [t0, t_final], Y0, method="RK45",
                    rtol=RTOL, atol=ATOL, max_step=MAX_STEP_LIMIT,
                    events=[out_of_bounds, reach_core])
    Y = sol.y  # [x1,y1,u1,v1,x2,y2,u2,v2]

    # Save trajectory (SI)
    pd.DataFrame({
        "x1_m": Y[0], "y1_m": Y[1], "uMBx1_mps": Y[2], "uMBy1_mps": Y[3],
        "x2_m": Y[4], "y2_m": Y[5], "uMBx2_mps": Y[6], "uMBy2_mps": Y[7],
    }).to_csv("trajectory_two_mb.csv", index=False)

    # Diagnostics (ROI + center + core)
    save_center_diagnostic(vf, roi_m, XC, YC, A)

    # ---------- Plot (µm + cm/s) ----------
    DISP_L=m_to_L(DISPLAY_POS_UNITS); DISP_V=mps_to(DISPLAY_VEL_UNITS)
    X=vf['x'].to_numpy(float)*DISP_L; Yp=vf['y'].to_numpy(float)*DISP_L
    U=vf['u'].to_numpy(float)*DISP_V; V=vf['v'].to_numpy(float)*DISP_V
    idx=np.arange(X.size)[::max(1,skip_interval)]
    Xs,Ys,Us,Vs=X[idx],Yp[idx],U[idx],V[idx]
    mags=np.hypot(Us,Vs); mags=np.where(mags==0,1e-12,mags)
    Un, Vn = Us/mags, Vs/mags
    norm=plt.Normalize(mags.min(),mags.max()); colors=plt.cm.viridis(norm(mags))

    tx1,ty1 = Y[0]*DISP_L, Y[1]*DISP_L
    tx2,ty2 = Y[4]*DISP_L, Y[5]*DISP_L

    fig,ax=plt.subplots(figsize=FIGSIZE,dpi=DPI_STATIC)
    ax.quiver(Xs,Ys,Un,Vn,color=colors,scale=quiver_scale)
    cbar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='viridis'),ax=ax)
    cbar.set_label(f'Velocity Magnitude ({DISPLAY_VEL_UNITS})',fontsize=LABEL_FONTSIZE)

    ax.plot(tx1,ty1,lw=3.0,color='r',label='MB1 path')
    ax.plot(tx2,ty2,lw=3.0,color='b',label='MB2 path')
    ax.plot(tx1[0],ty1[0],'o',ms=7,color='r',label='start MB1')
    ax.plot(tx2[0],ty2[0],'o',ms=7,color='b',label='start MB2')
    ax.add_patch(plt.Circle((XC*DISP_L,YC*DISP_L),A*DISP_L,ec='k',fc='none',lw=2))

    ax.set_title('Velocity field + Two MB trajectories',fontsize=TITLE_FONTSIZE)
    ax.set_xlabel(f'Position x ({DISPLAY_POS_UNITS})',fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(f'Position y ({DISPLAY_POS_UNITS})',fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='both',labelsize=TICK_FONTSIZE)
    ax.set_aspect('equal','box')
    if SHOW_GRID: ax.grid(True,alpha=0.3)
    ax.set_xlim(X.min(),X.max()); ax.set_ylim(Yp.min(),Yp.max())
    ax.legend(loc='upper right',ncol=2,prop={'size':LEGEND_FONTSIZE})
    fig.tight_layout()
    plt.savefig("velocity_Trajectory_two_mb.png",dpi=DPI_STATIC,bbox_inches="tight",
                pil_kwargs={"compress_level":1})
    plt.close(fig)
    print("Saved: velocity_Trajectory_two_mb.png  +  diagnostic_center.png")

# ---------------- entry ----------------
if __name__ == "__main__":
    main()
