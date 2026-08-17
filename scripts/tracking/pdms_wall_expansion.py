# Sub-pixel tracking of the compliant PDMS cavity wall (and inlet channel width)
# over a pulsation cycle, for the soft (15:1 curing ratio) PDMS device.
#
# Two measurements, both via sub-pixel peak-finding along intensity profiles
# (not simple thresholding, since the wall displacement is expected to be small):
#   1) Cavity rim radius: sample radially around a circle center at many angles,
#      find the bright-rim peak along each radial profile, average over angles.
#   2) Inlet channel width: sample perpendicular to the channel at a fixed
#      cross-section, find the two wall peaks, take their distance.
#
# Run once in DIAGNOSTIC_ONLY mode first (default) to check the detected
# circle, the angular ranges used, and the channel cross-section line against
# your actual footage before trusting the full time series.

import os
import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks

from pathlib import Path as _Path
REPO_ROOT = _Path(__file__).resolve().parents[2]

# ===================== USER CONFIG =====================

# Either a video file, or a folder of already-extracted frame images (png/tif/jpg).
INPUT_PATH = "/Volumes/Mahmoud_Black/Aneurysm_Exp/10-08-2026 pulsatile soft PDMS/vid_2026-08-09_20-15-21"

FPS = None  # frames per second of the source video/sequence; None -> x-axis is frame index
PIXELS_PER_UM = None  # e.g. 1280/900 if you know the field of view width in um; None -> report pixels

# ---- Cavity rim circle ----
# None = auto-detect via Hough on the first frame (verify with the diagnostic image!).
CENTER_INIT = (786.8, 558.8)   # (cx, cy) in pixels, confirmed via diagnostic overlay on frame 0
RADIUS_INIT = 229.9            # pixels
HOUGH_DP = 1.5
HOUGH_MIN_DIST = 500
HOUGH_PARAM1 = 100      # Canny high threshold for Hough's internal edge step
HOUGH_PARAM2 = 60       # accumulator threshold; lower = more (and weaker) circle candidates
HOUGH_RADIUS_RANGE = (200, 700)  # (min_r, max_r) px to search

N_ANGLES = 360                 # angular samples around the rim
RADIAL_SEARCH_RANGE_PX = 65    # WIDE search, used only on the reference frame (frame 0) to find
                                # and classify both the body (~230px) and neck/bulge (~280px)
                                # edges, which are about 45px apart at this device's geometry.
TRACKING_SEARCH_RANGE_PX = 20  # NARROW search used on every subsequent frame, centered on each
                                # angle's OWN frame-0 radius. Must stay well below the ~45px
                                # body/neck gap, or an angle near the boundary between the two
                                # groups can jump to the other cluster's trough frame-to-frame
                                # and corrupt both group means (this happened at +/-65px).
RADIAL_STEP_PX = 0.25          # sampling step along each radial profile (sub-pixel)

# The wall shows as a DARK trough along the radial profile (not a bright ridge) --
# confirmed empirically on real footage. A candidate radius is only accepted if the
# trough is at least this many intensity levels below the brighter of its two
# surrounding shoulders (min over [r-search,r_min] and [r_min,r+search] windows).
# This rejects angles with no real edge (noisy texture, occlusion) automatically,
# instead of relying on a hand-tuned angle-exclusion list.
MIN_TROUGH_PROMINENCE = 18

# Extra fixed exclusion (degrees, 0=+x axis, CCW) for regions you know are always
# occluded (e.g. the inlet channel) -- combined with the automatic prominence filter.
ANGLE_EXCLUDE_DEG = []

# The cavity is not circular: the main body sits at one radius, the neck/bulge
# near the channel connection sits further out. Angles are classified once (on
# the reference frame) by whether their measured radius is below/above this
# threshold, then that assignment is reused for every subsequent frame.
BODY_NECK_SPLIT_RADIUS = 255  # px, between the ~230px body and ~280px neck

# ---- Inlet channel width (two parallel bright wall lines) ----
# Endpoints of a line drawn ACROSS the channel (perpendicular to its walls) at a
# fixed cross-section, in pixels: (x0,y0) -> (x1,y1). None = skip channel-width measurement.
CHANNEL_LINE = None  # e.g. ((400, 60), (520, 180))
CHANNEL_SEARCH_STEP_PX = 0.25
CHANNEL_MIN_PEAK_DISTANCE_PX = 10   # minimum expected wall-to-wall separation
CHANNEL_PEAK_PROMINENCE = 5          # intensity prominence required to count as a wall peak

DIAGNOSTIC_ONLY = False   # True: just process frame 0 and save a diagnostic overlay
FRAME_STRIDE = 1         # process every Nth frame for the full run (1 = every frame)

OUT_CSV = REPO_ROOT / "outputs" / "pdms_wall_expansion.csv"
OUT_DIAGNOSTIC_PNG = REPO_ROOT / "plotting" / "pdms_wall_diagnostic.png"
OUT_PLOT_PNG = REPO_ROOT / "plotting" / "pdms_wall_expansion.png"

# ===================== Frame source =====================

def iter_frames(input_path):
    input_path = str(input_path)
    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "*")))
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                yield img
    else:
        cap = cv2.VideoCapture(input_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        cap.release()

def peek_first_frame(input_path):
    for gray in iter_frames(input_path):
        return gray
    raise RuntimeError(f"No frames found at {input_path}")

# ===================== Sub-pixel peak finding =====================

def subpixel_parabolic_offset(f_left, f_center, f_right):
    denom = (f_left - 2 * f_center + f_right)
    if abs(denom) < 1e-9:
        return 0.0
    return 0.5 * (f_left - f_right) / denom

def sample_line(gray, x0, y0, x1, y1, step_px):
    length = np.hypot(x1 - x0, y1 - y0)
    n = max(2, int(length / step_px))
    t = np.linspace(0.0, 1.0, n)
    xs = x0 + t * (x1 - x0)
    ys = y0 + t * (y1 - y0)
    profile = map_coordinates(gray.astype(float), [ys, xs], order=1, mode='nearest')
    dist = t * length
    return dist, profile, xs, ys

def radial_rim_radius(gray, cx, cy, angle_deg, r_guess, search_px, step_px, min_prominence):
    # The wall is a DARK trough along the radial profile -- find the minimum,
    # sub-pixel refine it, and only accept it if it's a real dip (prominent
    # relative to its brighter shoulders), not just noise/texture.
    theta = np.radians(angle_deg)
    r0, r1 = r_guess - search_px, r_guess + search_px
    n = max(2, int((r1 - r0) / step_px))
    radii = np.linspace(r0, r1, n)
    xs = cx + radii * np.cos(theta)
    ys = cy + radii * np.sin(theta)
    profile = map_coordinates(gray.astype(float), [ys, xs], order=1, mode='nearest')
    i = int(np.argmin(profile))
    shoulder_left = profile[:max(i, 1)].max() if i > 0 else profile[i]
    shoulder_right = profile[min(i + 1, len(profile) - 1):].max() if i < len(profile) - 1 else profile[i]
    prominence = min(shoulder_left, shoulder_right) - profile[i]
    if prominence < min_prominence:
        return None, None, None, prominence
    if 0 < i < len(profile) - 1:
        off = subpixel_parabolic_offset(profile[i - 1], profile[i], profile[i + 1])
    else:
        off = 0.0
    r_peak = radii[i] + off * (radii[1] - radii[0])
    return r_peak, xs[i], ys[i], prominence

def measure_rim(gray, cx, cy, r_guess, n_angles, exclude_ranges, search_px, step_px, min_prominence):
    """Per-angle rim measurement. Returns (used_angles, radii, points) as arrays --
    the cavity isn't circular, so callers should group/average by angle themselves
    rather than collapsing everything to one mean."""
    angles = np.linspace(0, 360, n_angles, endpoint=False)
    def excluded(a):
        return any(lo <= a <= hi for lo, hi in exclude_ranges)
    radii, used_angles, points = [], [], []
    for a in angles:
        if excluded(a):
            continue
        r_peak, xp, yp, prom = radial_rim_radius(gray, cx, cy, a, r_guess, search_px, step_px, min_prominence)
        if r_peak is None:
            continue
        radii.append(r_peak)
        used_angles.append(a)
        points.append((xp, yp))
    used_angles = np.array(used_angles)
    radii = np.array(radii)
    if len(radii) == 0:
        raise RuntimeError(
            "No angle produced a sufficiently prominent trough -- lower MIN_TROUGH_PROMINENCE "
            "or check CENTER_INIT/RADIUS_INIT/RADIAL_SEARCH_RANGE_PX against the diagnostic image."
        )
    return used_angles, radii, points

def track_rim_at_angles(gray, cx, cy, angles, r_guesses, search_px, step_px, min_prominence):
    """Like measure_rim, but each angle is searched around its OWN r_guess with a
    (typically narrow) fixed window -- used for tracking after the reference frame,
    so an angle near the body/neck boundary can't jump ~45px to the other cluster's
    trough just because it's momentarily more prominent that frame."""
    radii = np.full(len(angles), np.nan)
    for k, (a, rg) in enumerate(zip(angles, r_guesses)):
        r_peak, _, _, _ = radial_rim_radius(gray, cx, cy, a, rg, search_px, step_px, min_prominence)
        if r_peak is not None:
            radii[k] = r_peak
    return radii

def classify_body_neck(used_angles, radii, split_radius):
    """One-time classification (on a reference frame) of which angles belong to the
    main circular body vs. the top neck/bulge, based on a radius threshold."""
    body_mask = radii < split_radius
    return used_angles[body_mask], used_angles[~body_mask]

def group_mean_std(used_angles, radii, group_angles, tol_deg=0.5):
    if len(group_angles) == 0:
        return None, None
    mask = np.isin(np.round(used_angles / tol_deg), np.round(group_angles / tol_deg))
    sel = radii[mask]
    if len(sel) == 0:
        return None, None
    return float(np.mean(sel)), float(np.std(sel))

def polar_area(used_angles, radii, n_angles):
    """Cross-sectional area enclosed by the traced boundary (volume proxy), via the
    polar/shoelace integral A = 0.5 * sum(r_i^2 * dtheta). Angles with no valid
    measurement that frame (occluded/rejected) are filled by linear interpolation
    across the gap so a temporary dropout doesn't bias the area -- if the gap is
    large this is a real limitation, so we also report angular coverage.
    """
    order = np.argsort(used_angles)
    a = used_angles[order]; r = radii[order]
    full_angles = np.linspace(0, 360, n_angles, endpoint=False)
    r_full = np.interp(full_angles, a, r, period=360.0)
    dtheta = np.radians(360.0 / n_angles)
    area = 0.5 * np.sum(r_full ** 2) * dtheta
    coverage = len(used_angles) / n_angles
    return float(area), float(coverage)

def measure_channel_width(gray, line, step_px, min_dist_px, prominence):
    (x0, y0), (x1, y1) = line
    dist, profile, xs, ys = sample_line(gray, x0, y0, x1, y1, step_px)
    min_dist_samples = max(1, int(min_dist_px / step_px))
    peaks, _ = find_peaks(profile, distance=min_dist_samples, prominence=prominence)
    if len(peaks) < 2:
        return None, dist, profile, xs, ys, peaks
    # take the two most prominent peaks
    order = np.argsort(profile[peaks])[::-1][:2]
    p = np.sort(peaks[order])
    refined = []
    for i in p:
        if 0 < i < len(profile) - 1:
            off = subpixel_parabolic_offset(profile[i - 1], profile[i], profile[i + 1])
        else:
            off = 0.0
        refined.append(dist[i] + off * (dist[1] - dist[0]))
    width = refined[1] - refined[0]
    return float(width), dist, profile, xs, ys, p

# ===================== Diagnostic overlay =====================

def save_diagnostic(gray, cx, cy, r_guess, body_points, neck_points, channel_line, channel_pts, path):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.circle(vis, (int(cx), int(cy)), int(r_guess), (0, 255, 0), 1)
    cv2.drawMarker(vis, (int(cx), int(cy)), (0, 255, 0), cv2.MARKER_CROSS, 12, 1)
    for xp, yp in body_points:
        cv2.circle(vis, (int(round(xp)), int(round(yp))), 2, (0, 0, 255), -1)      # red = body
    for xp, yp in neck_points:
        cv2.circle(vis, (int(round(xp)), int(round(yp))), 2, (0, 165, 255), -1)    # orange = neck
    if channel_line is not None:
        (x0, y0), (x1, y1) = channel_line
        cv2.line(vis, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 1)
        for xp, yp in channel_pts:
            cv2.circle(vis, (int(round(xp)), int(round(yp))), 3, (0, 255, 255), -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis)
    print(f"Saved diagnostic overlay: {path}")
    print("Red dots = body-group rim peaks. Orange dots = neck/bulge-group rim peaks.")
    print("Yellow dots = detected channel-wall peaks.")
    print("Check these land ON the bright rim/walls, not offset -- tune RADIAL_SEARCH_RANGE_PX,")
    print("ANGLE_EXCLUDE_DEG, CENTER_INIT/RADIUS_INIT, BODY_NECK_SPLIT_RADIUS, or CHANNEL_LINE accordingly.")

# ===================== Main =====================

def main():
    if INPUT_PATH is None:
        raise ValueError("Set INPUT_PATH to a video file or folder of frames before running.")

    first = peek_first_frame(INPUT_PATH)
    print(f"First frame shape: {first.shape}")

    cx, cy, r_guess = CENTER_INIT[0] if CENTER_INIT else None, \
                       CENTER_INIT[1] if CENTER_INIT else None, RADIUS_INIT
    if cx is None or r_guess is None:
        blurred = cv2.GaussianBlur(first, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=HOUGH_DP, minDist=HOUGH_MIN_DIST,
            param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
            minRadius=HOUGH_RADIUS_RANGE[0], maxRadius=HOUGH_RADIUS_RANGE[1],
        )
        if circles is None:
            raise RuntimeError(
                "Hough circle detection found nothing -- set CENTER_INIT and RADIUS_INIT manually."
            )
        cx, cy, r_guess = circles[0, 0]
        print(f"[Hough] detected circle: center=({cx:.1f},{cy:.1f}) r={r_guess:.1f} px "
              f"-- VERIFY this against the diagnostic image before trusting it.")

    used_angles, radii, points = measure_rim(
        first, cx, cy, r_guess, N_ANGLES, ANGLE_EXCLUDE_DEG,
        RADIAL_SEARCH_RANGE_PX, RADIAL_STEP_PX, MIN_TROUGH_PROMINENCE,
    )
    body_angles, neck_angles = classify_body_neck(used_angles, radii, BODY_NECK_SPLIT_RADIUS)
    body_mean, body_std = group_mean_std(used_angles, radii, body_angles)
    neck_mean, neck_std = group_mean_std(used_angles, radii, neck_angles)
    print(f"Frame 0: body radius = {body_mean:.2f} +/- {body_std:.2f} px over {len(body_angles)} angles")
    if neck_mean is not None:
        print(f"Frame 0: neck radius = {neck_mean:.2f} +/- {neck_std:.2f} px over {len(neck_angles)} angles")
    else:
        print("Frame 0: no neck/bulge angles found (BODY_NECK_SPLIT_RADIUS may need adjusting).")

    body_points = [(x, y) for a, (x, y) in zip(used_angles, points) if a in set(body_angles)]
    neck_points = [(x, y) for a, (x, y) in zip(used_angles, points) if a in set(neck_angles)]

    channel_pts = []
    if CHANNEL_LINE is not None:
        width, dist, profile, xs, ys, peak_idx = measure_channel_width(
            first, CHANNEL_LINE, CHANNEL_SEARCH_STEP_PX,
            CHANNEL_MIN_PEAK_DISTANCE_PX, CHANNEL_PEAK_PROMINENCE,
        )
        if width is not None:
            print(f"Frame 0 channel width: {width:.2f} px")
            channel_pts = [(xs[i], ys[i]) for i in peak_idx]
        else:
            print("Channel width: fewer than 2 peaks found -- tune CHANNEL_LINE/prominence.")

    save_diagnostic(first, cx, cy, r_guess, body_points, neck_points, CHANNEL_LINE, channel_pts, OUT_DIAGNOSTIC_PNG)

    if DIAGNOSTIC_ONLY:
        print("\nDIAGNOSTIC_ONLY=True: stopping after frame 0. "
              "Check the overlay, tune config, then set DIAGNOSTIC_ONLY=False to run the full series.")
        return

    # Per-angle reference radii from frame 0 -- each angle is tracked in later frames
    # with a NARROW window centered on its own reference value (see TRACKING_SEARCH_RANGE_PX).
    ref_radius = dict(zip(used_angles.tolist(), radii.tolist()))
    track_angles = np.concatenate([body_angles, neck_angles])
    track_r_guess = np.array([ref_radius[a] for a in track_angles])
    body_mask_track = np.isin(track_angles, body_angles)
    neck_mask_track = np.isin(track_angles, neck_angles)

    rows = []
    for i, gray in enumerate(iter_frames(INPUT_PATH)):
        if i % FRAME_STRIDE != 0:
            continue
        radii_i = track_rim_at_angles(
            gray, cx, cy, track_angles, track_r_guess,
            TRACKING_SEARCH_RANGE_PX, RADIAL_STEP_PX, MIN_TROUGH_PROMINENCE,
        )
        b_vals = radii_i[body_mask_track]; b_vals = b_vals[~np.isnan(b_vals)]
        n_vals = radii_i[neck_mask_track]; n_vals = n_vals[~np.isnan(n_vals)]
        b_mean = float(b_vals.mean()) if len(b_vals) else None
        b_std = float(b_vals.std()) if len(b_vals) else None
        n_mean = float(n_vals.mean()) if len(n_vals) else None
        n_std = float(n_vals.std()) if len(n_vals) else None
        row = {
            "frame": i,
            "body_radius_px": b_mean, "body_radius_std_px": b_std,
            "neck_radius_px": n_mean, "neck_radius_std_px": n_std,
        }
        if CHANNEL_LINE is not None:
            width, *_ = measure_channel_width(
                gray, CHANNEL_LINE, CHANNEL_SEARCH_STEP_PX,
                CHANNEL_MIN_PEAK_DISTANCE_PX, CHANNEL_PEAK_PROMINENCE,
            )
            row["channel_width_px"] = width
        rows.append(row)
        if i % 50 == 0:
            print(f"frame {i}: body_radius={b_mean:.2f}px  neck_radius={n_mean if n_mean else float('nan'):.2f}px")

    df = pd.DataFrame(rows)
    if FPS:
        df["time_s"] = df["frame"] / FPS
    if PIXELS_PER_UM:
        df["body_radius_um"] = df["body_radius_px"] / PIXELS_PER_UM
        df["neck_radius_um"] = df["neck_radius_px"] / PIXELS_PER_UM
        if "channel_width_px" in df:
            df["channel_width_um"] = df["channel_width_px"] / PIXELS_PER_UM

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")

    x = df["time_s"] if "time_s" in df else df["frame"]
    xlabel = "Time (s)" if "time_s" in df else "Frame"
    body_col = "body_radius_um" if "body_radius_um" in df else "body_radius_px"
    neck_col = "neck_radius_um" if "neck_radius_um" in df else "neck_radius_px"
    unit = "um" if "body_radius_um" in df else "px"

    n_panels = 3 if "channel_width_px" in df else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 3 * n_panels), sharex=True)
    axes = np.atleast_1d(axes)
    axes[0].plot(x, df[body_col], lw=1.5, color='tab:red')
    axes[0].set_ylabel(f"Body radius ({unit})")
    axes[0].set_title("Cavity wall expansion/contraction: main body vs. neck/bulge")
    axes[1].plot(x, df[neck_col], lw=1.5, color='tab:orange')
    axes[1].set_ylabel(f"Neck radius ({unit})")
    if n_panels == 3:
        ycol2 = "channel_width_um" if "channel_width_um" in df else "channel_width_px"
        ylabel2 = f"Channel width ({unit})"
        axes[2].plot(x, df[ycol2], lw=1.5, color='tab:blue')
        axes[2].set_ylabel(ylabel2)
    axes[-1].set_xlabel(xlabel)
    fig.tight_layout()
    OUT_PLOT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PLOT_PNG, dpi=200)
    print(f"Saved: {OUT_PLOT_PNG}")

    for label, col in [("Body", body_col), ("Neck", neck_col)]:
        amp = df[col].max() - df[col].min()
        pct = 100 * amp / df[col].mean()
        print(f"{label} radius: mean={df[col].mean():.3f}, peak-to-peak amplitude={amp:.3f} "
              f"({pct:.2f}% of mean)")

if __name__ == "__main__":
    main()
