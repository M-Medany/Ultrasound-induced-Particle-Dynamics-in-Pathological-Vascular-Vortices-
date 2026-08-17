# Cavity cross-sectional area (volume proxy) over time, from the full traced
# PDMS wall boundary -- not a fixed body/neck split, since the wall shape itself
# changes with pressure (validated on vid_2026-08-09_20-15-21: the body/neck
# distinction that's clear at rest nearly disappears at peak pressure).
#
# Boundary definition: for each angle, scan radially OUTWARD starting from
# R0_PX (set below the smallest plausible wall radius, but above the
# debris-heavy interior) and take the FIRST sufficiently deep, sufficiently
# wide dark trough. This is more robust than "deepest trough in a fixed
# window", which was found to jump unpredictably between the near (body) and
# far (neck) edges when both are visible in the same window.
#
# Per-video config: some videos have an extra dark gas-pocket/bubble region
# occluding part of the boundary (in addition to the always-open channel
# junction) -- exclude those angular ranges explicitly per video rather than
# trying to make one rule handle every occlusion type.

import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks

from pathlib import Path as _Path
REPO_ROOT = _Path(__file__).resolve().parents[2]

# ===================== Per-video configs =====================
VIDEOS = {
    "vid1": dict(
        folder="/Volumes/Mahmoud_Black/Aneurysm_Exp/10-08-2026 pulsatile soft PDMS/vid_2026-08-09_20-15-21",
        cx=786.8, cy=558.8,
        r0=190, r1=330,
        exclude_deg=[],  # channel-open side handled by the prominence/width filter itself
    ),
    "vid3": dict(
        folder="/Volumes/Mahmoud_Black/Aneurysm_Exp/10-08-2026 pulsatile soft PDMS/vid_2026-08-10_00-09-45",
        cx=790, cy=530,
        r0=170, r1=330,
        exclude_deg=[(-40, 95)],  # gas-pocket region occludes the outer wall here (inner
                                  # gas-fluid edge would otherwise be mistaken for the wall)
    ),
    "vid5": dict(
        folder="/Volumes/Mahmoud_Black/Aneurysm_Exp/10-08-2026 pulsatile soft PDMS/vid_2026-08-10_01-05-57",
        cx=780, cy=530,
        r0=190, r1=330,
        exclude_deg=[],
        # much lower contrast + heavy interior debris than vid1 -- needed a lower
        # prominence to catch the faint true edge, and a wider min-width to keep
        # narrow debris specks from being mistaken for it (validated visually).
        min_prominence=6, min_width_px=15,
        min_coverage=0.25,  # gas pocket + debris-filtered region legitimately caps
                            # coverage around 30-40% here, well below vid1's 75%+
    ),
}

N_ANGLES = 360
RADIAL_STEP_PX = 0.25
MIN_PROMINENCE = 15
MIN_WIDTH_PX = 6
FRAME_STRIDE = 1
PIXELS_PER_UM = None  # set once known
FPS = None             # set once known

# ===================== Core =====================

def radial_first_trough(gray, cx, cy, angle_deg, r0, r1, step_px, min_prominence, min_width_px):
    theta = np.radians(angle_deg)
    radii = np.arange(r0, r1, step_px)
    xs = cx + radii * np.cos(theta)
    ys = cy + radii * np.sin(theta)
    prof = map_coordinates(gray.astype(float), [ys, xs], order=1, mode='nearest')
    peaks, _ = find_peaks(-prof, prominence=min_prominence, width=min_width_px / step_px)
    if len(peaks) == 0:
        return None
    return radii[peaks[0]]

def excluded(a, ranges):
    for lo, hi in ranges:
        lo_n, hi_n = lo % 360, hi % 360
        if lo_n <= hi_n:
            if lo_n <= a <= hi_n:
                return True
        else:  # wraps through 0
            if a >= lo_n or a <= hi_n:
                return True
    return False

def measure_boundary(gray, cx, cy, r0, r1, exclude_ranges, min_prominence=None, min_width_px=None):
    min_prominence = MIN_PROMINENCE if min_prominence is None else min_prominence
    min_width_px = MIN_WIDTH_PX if min_width_px is None else min_width_px
    angles, radii = [], []
    for a in np.linspace(0, 360, N_ANGLES, endpoint=False):
        if excluded(a, exclude_ranges):
            continue
        r = radial_first_trough(gray, cx, cy, a, r0, r1, RADIAL_STEP_PX, min_prominence, min_width_px)
        if r is not None:
            angles.append(a)
            radii.append(r)
    return np.array(angles), np.array(radii)

def polar_area(angles, radii, n_angles):
    order = np.argsort(angles)
    a = angles[order]; r = radii[order]
    full_angles = np.linspace(0, 360, n_angles, endpoint=False)
    r_full = np.interp(full_angles, a, r, period=360.0)
    dtheta = np.radians(360.0 / n_angles)
    area = 0.5 * np.sum(r_full ** 2) * dtheta
    coverage = len(angles) / n_angles
    return float(area), float(coverage)

def iter_frames(folder):
    for f in sorted(glob.glob(folder + "/*.tiff")):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            yield img

def run(name, cfg):
    print(f"\n=== {name}: {cfg['folder']} ===")
    min_coverage = cfg.get("min_coverage", 0.5)
    rows = []
    for i, gray in enumerate(iter_frames(cfg["folder"])):
        if i % FRAME_STRIDE != 0:
            continue
        angles, radii = measure_boundary(
            gray, cfg["cx"], cfg["cy"], cfg["r0"], cfg["r1"], cfg["exclude_deg"],
            min_prominence=cfg.get("min_prominence"), min_width_px=cfg.get("min_width_px"),
        )
        if len(angles) < N_ANGLES * min_coverage:
            area, coverage, r_eff = np.nan, len(angles) / N_ANGLES, np.nan
        else:
            area, coverage = polar_area(angles, radii, N_ANGLES)
            r_eff = np.sqrt(area / np.pi)
        rows.append({"frame": i, "area_px2": area, "coverage": coverage, "r_eff_px": r_eff})
        if i % 100 == 0:
            print(f"  frame {i}: area={area:.0f}px^2  r_eff={r_eff:.2f}px  coverage={coverage:.0%}")

    df = pd.DataFrame(rows)
    if FPS:
        df["time_s"] = df["frame"] / FPS
    if PIXELS_PER_UM:
        df["area_um2"] = df["area_px2"] / PIXELS_PER_UM ** 2
        df["r_eff_um"] = df["r_eff_px"] / PIXELS_PER_UM

    out_csv = REPO_ROOT / "outputs" / f"pdms_volume_change_{name}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    x = df["time_s"] if "time_s" in df else df["frame"]
    xlabel = "Time (s)" if "time_s" in df else "Frame"
    ycol = "area_um2" if "area_um2" in df else "area_px2"
    ylabel = "Cross-sectional area (um^2)" if "area_um2" in df else "Cross-sectional area (px^2)"

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(x, df[ycol], lw=1.2)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"Cavity cross-sectional area over time ({name}) -- volume-change proxy")
    fig.tight_layout()
    out_png = REPO_ROOT / "plotting" / f"pdms_volume_change_{name}.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_png}")

    valid = df.dropna(subset=[ycol])
    amp = valid[ycol].max() - valid[ycol].min()
    pct = 100 * amp / valid[ycol].mean()
    print(f"{name}: mean area={valid[ycol].mean():.1f}, peak-to-peak amplitude={amp:.1f} ({pct:.2f}% of mean)")
    print(f"{name}: mean angular coverage={df['coverage'].mean():.0%}")
    return df

if __name__ == "__main__":
    for name, cfg in VIDEOS.items():
        run(name, cfg)
