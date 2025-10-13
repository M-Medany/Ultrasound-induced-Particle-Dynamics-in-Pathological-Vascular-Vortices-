# ================== two_mb_ROI_locked.py ==================
# Read COMSOL CSV (alias tolerant), autoscale normalized speeds.
# Find vortex center by vorticity **inside a user ROI in µm**.
# Calibrate Rankine (a, Γ) strictly inside the ROI + edge-aware clipping.
# SI-consistent dynamics; gentle inward bias for spiral-in.
# Plot field in µm and cm/s (quiver like your working reader).
# Saves an extra diagnostic image: diagnostic_center.png

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
from multiprocessing import get_context, cpu_count
from PIL import ImageFile
ImageFile.MAXBLOCK = 1 << 24

# ---------------- USER CONFIG ----------------
CSV_PATH = r"C:\Users\M4\VSCode_Projects\Ultrasound-Swarm-Microbubbles-Navigating-Vortices-to-Target-and-Fill-Aneurysms\Excel_data_velocity_comsol\Normalized_Velocity_60_cm_Full.csv"

# CSV units & normalization
CSV_POS_UNITS = 'm'          # {'m','mm','um'}
CSV_VEL_UNITS = 'auto'       # {'m/s','cm/s','mm/s','unitless','auto'}
AUTOSCALE_TARGET_MAX_CM_S = 60.0
AUTOSCALE_REGION = 'global'  # or 'inlet'

# Display units
DISPLAY_POS_UNITS = 'um'
DISPLAY_VEL_UNITS = 'cm/s'

# ---- Center locking ROI (µm) — this is the key fix ----
CENTER_ROI_UM = (90.0, 150.0, 110.0, 170.0)  # (x_min, x_max, y_min, y_max) in µm

# Rankine a safety clipping (as fraction of nearest edge distance)
A_EDGE_FRACTION_MAX = 0.65
A_EDGE_FRACTION_MIN = 0.05

# Releases near vortex (absolute µm)
MB1_START_ABS_UM = (100.0, 120.0)
MB2_START_ABS_UM = (140.0, 160.0)
MASK_RADIUS_UM   = 10.0

# Physics (SI)
rho = 1000.0
CD  = 4.0            # slightly lower drag -> easier spiral
PRESSURE_GAIN = 1.25 # >1 increases inward -∇p
KNN_K = 12
AUTO_CALIBRATE_RANKINE = True

# Integration & output
t_span = (0.0, 2_000_000.0)
rtol, atol = 1e-6, 1e-9
chunk_steps_target = 400
MAX_SWIRL_STEPS_PER_PERIOD = 60

FIGSIZE = (8,5)
DPI_STATIC = 600
TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 14
TICK_FONTSIZE  = 12
LEGEND_FONTSIZE= 11
skip_interval  = 1
quiver_scale   = 80
SHOW_GRID = True

# ---------------- Internals ----------------
TREE=None; UVAL=None; VVAL=None
XC=None; YC=None; A=None
GAM=None; RHO=None; CDG=None
RTOL=None; ATOL=None
CHUNK_STEPS_TARGET=None
XMIN=None; XMAX=None; YMIN=None; YMAX=None
MAX_STEP_LIMIT=np.inf
NPTS=0

def L_to_m(u): return {'m':1.0,'mm':1e-3,'um':1e-6}[u]
def m_to_L(u): return {'m':1.0,'mm':1e3,'um':1e6}[u]
def to_mps(u): return {'m/s':1.0,'cm/s':1e-2,'mm/s':1e-3}[u]
def mps_to(u): return {'m/s':1.0,'cm/s':100.0,'mm/s':1000.0}[u]

def standardize_columns(df):
    df.columns = df.columns.str.strip()
    lower = {c.lower():c for c in df.columns}
    def pick(*cands):
        for c in cands:
            if c in lower: return lower[c]
        return None
    x = pick('x','pos_x','position x (m)','x (m)','x_mm'); y = pick('y','pos_y','position y (m)','y (m)','y_mm')
    u = pick('u','ux','vx','u2'); v = pick('v','uy','vy','v2')
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
        u = raw['u'].astype(float).to_numpy(); v = raw['v'].astype(float).to_numpy()
        s = np.hypot(u,v)
        looks_unitless = (np.nanmax(s) <= 1.5) or (np.nanmedian(s) < 1e-2)
        if looks_unitless or CSV_VEL_UNITS=='unitless' or CSV_VEL_UNITS=='auto':
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

# ---------------- Center via vorticity inside ROI ----------------
def estimate_center_vorticity_roi(df, roi_m, k=12):
    X = df['x'].to_numpy(float); Y = df['y'].to_numpy(float)
    U = df['u'].to_numpy(float); V = df['v'].to_numpy(float)
    xlo,xhi,ylo,yhi = roi_m
    sel = (X>=xlo)&(X<=xhi)&(Y>=ylo)&(Y<=yhi)
    Xr, Yr, Ur, Vr = X[sel], Y[sel], U[sel], V[sel]
    if Xr.size < 200:
        raise RuntimeError("ROI too small / nearly empty. Enlarge CENTER_ROI_UM.")
    pts = np.column_stack([Xr,Yr]); tree = cKDTree(pts)
    k = min(k, pts.shape[0])
    # estimate local vorticity at ROI points using weighted LSQ
    omega = np.zeros(pts.shape[0])
    for i in range(pts.shape[0]):
        d, idx = tree.query(pts[i], k=k)
        dx = pts[idx,0]-pts[i,0]; dy = pts[idx,1]-pts[i,1]
        du = Ur[idx]-Ur[i];        dv = Vr[idx]-Vr[i]
        A  = np.column_stack([dx,dy])
        w  = 1.0/(d+1e-12); W = np.diag(w)
        try:
            gu = np.linalg.lstsq(W@A, W@du, rcond=None)[0]  # [ux,uy]
            gv = np.linalg.lstsq(W@A, W@dv, rcond=None)[0]  # [vx,vy]
            omega[i] = gv[0] - gu[1]                        # ωz
        except np.linalg.LinAlgError:
            omega[i] = 0.0
    q = np.nanpercentile(np.abs(omega),95.0)
    core = np.abs(omega) >= q
    w = np.abs(omega[core]) + 1e-12
    xc = np.average(Xr[core], weights=w); yc = np.average(Yr[core], weights=w)
    return float(xc), float(yc), (Xr,Yr,omega)

# ---------------- Rankine calibration inside ROI ----------------
def estimate_rankine_in_roi(df, xc, yc, roi_m,
                            a_edge_frac_max=0.65, a_edge_frac_min=0.05):
    X=df['x'].to_numpy(float); Y=df['y'].to_numpy(float)
    U=df['u'].to_numpy(float); V=df['v'].to_numpy(float)
    xlo,xhi,ylo,yhi = roi_m
    sel = (X>=xlo)&(X<=xhi)&(Y>=ylo)&(Y<=yhi)
    Xr, Yr, Ur, Vr = X[sel], Y[sel], U[sel], V[sel]
    if Xr.size < 300:
        sel = np.ones_like(X,dtype=bool)  # fallback to full field
        Xr,Yr,Ur,Vr = X[sel],Y[sel],U[sel],V[sel]

    dx = Xr - xc; dy = Yr - yc; r = np.hypot(dx,dy); rs = np.where(r<1e-12,1e-12,r)
    rx,ry = dx/rs, dy/rs
    vt = -Ur*ry + Vr*rx

    # robust |vt|(r): medians on 48 bins
    nbins=48
    bins = np.linspace(r.min(), r.max(), nbins+1)
    which = np.digitize(r, bins) - 1
    vt_abs_med = np.full(nbins,np.nan); r_med = np.full(nbins,np.nan)
    for i in range(nbins):
        m = which==i
        if m.sum()>=15:
            vt_abs_med[i]=np.median(np.abs(vt[m])); r_med[i]=np.median(r[m])
    i_peak = int(np.nanargmax(vt_abs_med))
    a_raw  = float(r_med[i_peak]) if np.isfinite(r_med[i_peak]) else float(np.median(r))
    vt_a   = float(vt_abs_med[i_peak]) if np.isfinite(vt_abs_med[i_peak]) else float(np.median(np.abs(vt)))

    # clip a by distance to nearest edge
    xmin,xmax = X.min(),X.max(); ymin,ymax = Y.min(),Y.max()
    r_edge = min(xc-xmin, xmax-xc, yc-ymin, ymax-yc)
    a_min = a_edge_frac_min*r_edge; a_max=a_edge_frac_max*r_edge
    a_est = float(np.clip(a_raw, a_min, a_max))

    shell = (r>=0.9*a_est)&(r<=1.1*a_est)
    if shell.sum()>=40:
        vt_a = float(np.median(np.abs(vt[shell])))

    r_outer_hi = min(2.5*a_est, 0.95*r_edge)
    outer = (r > 1.15*a_est) & (r < r_outer_hi)
    if outer.sum() < 80: outer = (r > a_est) & (r < r_outer_hi)
    gamma_vals = 2*np.pi*r[outer]*vt[outer]
    if gamma_vals.size==0: gamma_vals = 2*np.pi*r*vt
    Gamma_est = float(np.median(gamma_vals))
    T_swirl = 2*np.pi*a_est/max(vt_a,1e-12)
    return a_est, Gamma_est, T_swirl

# ---------------- Field & dynamics ----------------
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

def solve_ode_chunk(t0,t1,Y0,max_step):
    sol=solve_ivp(rhs,[t0,t1],Y0,method="RK45",rtol=RTOL,atol=ATOL,max_step=max_step,events=[out_of_bounds,reach_core])
    return sol.t, sol.y

def init_worker(points,u,v,xc,yc,a,gamma,rho,cd,rtol,atol,chunk_steps,xmin,xmax,ymin,ymax,max_step):
    global TREE,UVAL,VVAL,XC,YC,A,GAM,RHO,CDG,RTOL,ATOL,CHUNK_STEPS_TARGET,XMIN,XMAX,YMIN,YMAX,MAX_STEP_LIMIT,NPTS
    TREE=cKDTree(points); UVAL=u; VVAL=v; XC=xc; YC=yc; A=a; GAM=gamma; RHO=rho; CDG=cd
    RTOL=rtol; ATOL=atol; CHUNK_STEPS_TARGET=chunk_steps
    XMIN,XMAX,YMIN,YMAX=xmin,xmax,ymin,ymax; MAX_STEP_LIMIT=max_step; NPTS=int(points.shape[0])

# ---------------- Diagnostics ----------------
def save_center_diagnostic(vf, xc, yc, a, roi_m):
    # scatter ω magnitude in ROI + rectangle and circle
    X=vf['x'].to_numpy(float); Y=vf['y'].to_numpy(float)
    U=vf['u'].to_numpy(float); V=vf['v'].to_numpy(float)
    # coarse local vorticity for coloring
    pts=np.column_stack([X,Y]); tree=cKDTree(pts)
    k=min(10,pts.shape[0])
    omega=np.zeros(pts.shape[0])
    for i in range(pts.shape[0]//10):  # sample 10% for speed
        ii=10*i
        d,idx=tree.query(pts[ii],k=k)
        dx=pts[idx,0]-pts[ii,0]; dy=pts[idx,1]-pts[ii,1]
        du=U[idx]-U[ii]; dv=V[idx]-V[ii]
        A=np.column_stack([dx,dy]); w=1.0/(d+1e-12); W=np.diag(w)
        try:
            gu=np.linalg.lstsq(W@A, W@du, rcond=None)[0]
            gv=np.linalg.lstsq(W@A, W@dv, rcond=None)[0]
            omega[ii]=gv[0]-gu[1]
        except: omega[ii]=0.0
    DISP_L=m_to_L(DISPLAY_POS_UNITS); DISP_V=mps_to(DISPLAY_VEL_UNITS)
    fig,ax=plt.subplots(figsize=(6,5),dpi=220)
    sc=ax.scatter((X[::10]*DISP_L),(Y[::10]*DISP_L),c=np.abs(omega[::10]),cmap='magma',s=6)
    plt.colorbar(sc,ax=ax,label='|local ω| (1/s, relative)')
    rect = plt.Rectangle((roi_m[0]*DISP_L,roi_m[2]*DISP_L),
                         (roi_m[1]-roi_m[0])*DISP_L,(roi_m[3]-roi_m[2])*DISP_L,
                         ec='cyan',fc='none',lw=2,ls='--')
    ax.add_patch(rect)
    ax.add_patch(plt.Circle((xc*DISP_L,yc*DISP_L),a*DISP_L,ec='k',fc='none',lw=2))
    ax.plot([xc*DISP_L],[yc*DISP_L],'wx',ms=8,mew=2)
    ax.set_aspect('equal'); ax.set_title('Center diagnostic (ROI lock)')
    ax.set_xlabel(f'X ({DISPLAY_POS_UNITS})'); ax.set_ylabel(f'Y ({DISPLAY_POS_UNITS})')
    fig.tight_layout(); fig.savefig('diagnostic_center.png'); plt.close(fig)

# ---------------- Main ----------------
def main():
    global GAM, RTOL, ATOL, CHUNK_STEPS_TARGET, MAX_STEP_LIMIT

    print(f"Reading velocity data ...\nCSV: {CSV_PATH}")
    vf = read_csv_field(CSV_PATH)  # SI (m, m/s)
    points=vf[['x','y']].to_numpy(float); u=vf['u'].to_numpy(float); v=vf['v'].to_numpy(float)
    xmin,xmax=points[:,0].min(), points[:,0].max(); ymin,ymax=points[:,1].min(), points[:,1].max()
    print(f"Domain (SI): x∈[{xmin:.6g},{xmax:.6g}] m, y∈[{ymin:.6g},{ymax:.6g}] m")

    # ---- Center in ROI (µm → m) ----
    xlo_um,xhi_um,ylo_um,yhi_um = CENTER_ROI_UM
    roi_m = (xlo_um*1e-6, xhi_um*1e-6, ylo_um*1e-6, yhi_um*1e-6)
    xc,yc,_ = estimate_center_vorticity_roi(vf, roi_m, k=KNN_K)
    print(f"[Center ROI lock] (xc,yc)=({xc*1e6:.1f},{yc*1e6:.1f}) µm")

    # ---- Rankine calibration inside same ROI ----
    a_est,Gamma_est,T_swirl = estimate_rankine_in_roi(
        vf, xc, yc, roi_m,
        a_edge_frac_max=A_EDGE_FRACTION_MAX,
        a_edge_frac_min=A_EDGE_FRACTION_MIN
    )
    print(f"[Calibrated in ROI] a≈{a_est*1e6:.1f} µm, Γ≈{Gamma_est:.6g} m^2/s, Tθ≈{T_swirl:.4g} s")

    # ---- Dynamics setup ----
    A_local = a_est; GAM = float(Gamma_est)
    MAX_STEP_LIMIT = float(T_swirl)/float(MAX_SWIRL_STEPS_PER_PERIOD)
    print(f"[Dynamics] Γ={GAM:.6g}, a={A_local*1e6:.1f} µm, max_step≤{MAX_STEP_LIMIT:.3g}s, CD={CD}, PRESSURE_GAIN={PRESSURE_GAIN}")

    # Starts (µm → m)
    x0_1,y0_1 = MB1_START_ABS_UM[0]*1e-6, MB1_START_ABS_UM[1]*1e-6
    x0_2,y0_2 = MB2_START_ABS_UM[0]*1e-6, MB2_START_ABS_UM[1]*1e-6
    if not (xmin<=x0_1<=xmax and ymin<=y0_1<=ymax): raise ValueError("MB1 start outside domain.")
    if not (xmin<=x0_2<=xmax and ymin<=y0_2<=ymax): raise ValueError("MB2 start outside domain.")

    # Integrate (spawn-safe chain)
    num_chunks=cpu_count(); edges=np.linspace(t_span[0],t_span[1],num_chunks+1)
    intervals=[(edges[i],edges[i+1]) for i in range(num_chunks)]
    from multiprocessing import get_context
    ctx=get_context("spawn"); results=[]
    with ctx.Pool(processes=num_chunks,
                  initializer=init_worker,
                  initargs=(points,u,v,xc,yc,A_local,GAM,rho,CD,rtol,atol,chunk_steps_target,
                            xmin,xmax,ymin,ymax,MAX_STEP_LIMIT)) as pool:
        Yprev=[x0_1,y0_1,0.0,0.0,x0_2,y0_2,0.0,0.0]
        results.append(pool.apply_async(solve_ode_chunk,args=(intervals[0][0],intervals[0][1],Yprev,MAX_STEP_LIMIT)))
        for i in range(1,num_chunks):
            t,y=results[i-1].get()
            Yprev=[y[0,-1],y[1,-1],y[2,-1],y[3,-1],y[4,-1],y[5,-1],y[6,-1],y[7,-1]]
            results.append(pool.apply_async(solve_ode_chunk,args=(intervals[i][0],intervals[i][1],Yprev,MAX_STEP_LIMIT)))
        final_t=[]; final_y=[]
        for r in results:
            t,y=r.get(); final_t.extend(t); final_y.append(y)
    final_y=np.concatenate(final_y,axis=1)

    # Save trajectory (SI)
    pd.DataFrame({
        "x1_m":final_y[0], "y1_m":final_y[1], "uMBx1_mps":final_y[2], "uMBy1_mps":final_y[3],
        "x2_m":final_y[4], "y2_m":final_y[5], "uMBx2_mps":final_y[6], "uMBy2_mps":final_y[7],
    }).to_csv("trajectory_two_mb.csv", index=False)

    # ---- Diagnostics: verify the center really sits in the ROI ----
    save_center_diagnostic(vf, xc, yc, A_local, roi_m)

    # ---- Plot (µm + cm/s) ----
    DISP_L=m_to_L(DISPLAY_POS_UNITS); DISP_V=mps_to(DISPLAY_VEL_UNITS)
    X=vf['x'].to_numpy(float)*DISP_L; Y=vf['y'].to_numpy(float)*DISP_L
    U=vf['u'].to_numpy(float)*DISP_V; V=vf['v'].to_numpy(float)*DISP_V
    idx=np.arange(X.size)[::max(1,skip_interval)]
    Xs,Ys,Us,Vs=X[idx],Y[idx],U[idx],V[idx]
    mags=np.hypot(Us,Vs); mags=np.where(mags==0,1e-12,mags)
    Un, Vn = Us/mags, Vs/mags
    norm=plt.Normalize(mags.min(),mags.max()); colors=plt.cm.viridis(norm(mags))

    tx1,ty1=final_y[0]*DISP_L,final_y[1]*DISP_L
    tx2,ty2=final_y[4]*DISP_L,final_y[5]*DISP_L

    fig,ax=plt.subplots(figsize=FIGSIZE,dpi=DPI_STATIC)
    ax.quiver(Xs,Ys,Un,Vn,color=colors,scale=quiver_scale)
    cbar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='viridis'),ax=ax)
    cbar.set_label(f'Velocity Magnitude ({DISPLAY_VEL_UNITS})',fontsize=LABEL_FONTSIZE)

    ax.plot(tx1,ty1,lw=3.0,color='r',label='MB1 path'); ax.plot(tx2,ty2,lw=3.0,color='b',label='MB2 path')
    ax.plot(tx1[0],ty1[0],'o',ms=7,color='r',label='start MB1'); ax.plot(tx2[0],ty2[0],'o',ms=7,color='b',label='start MB2')
    ax.add_patch(plt.Circle((xc*DISP_L,yc*DISP_L),A_local*DISP_L,ec='k',fc='none',lw=2))

    ax.set_title('Velocity field + Two MB trajectories',fontsize=TITLE_FONTSIZE)
    ax.set_xlabel(f'Position x ({DISPLAY_POS_UNITS})',fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(f'Position y ({DISPLAY_POS_UNITS})',fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='both',labelsize=TICK_FONTSIZE)
    ax.set_aspect('equal','box'); 
    if SHOW_GRID: ax.grid(True,alpha=0.3)
    ax.set_xlim(X.min(),X.max()); ax.set_ylim(Y.min(),Y.max())
    ax.legend(loc='upper right',ncol=2,prop={'size':LEGEND_FONTSIZE})
    fig.tight_layout(); plt.savefig("velocity_Trajectory_two_mb.png",dpi=DPI_STATIC,bbox_inches="tight",pil_kwargs={"compress_level":1})
    plt.close(fig)
    print("Saved: velocity_Trajectory_two_mb.png  +  diagnostic_center.png")

# -------------- entry --------------
if __name__ == "__main__":
    RHO=rho; CDG=CD; RTOL=rtol; ATOL=atol; CHUNK_STEPS_TARGET=chunk_steps_target
    main()
