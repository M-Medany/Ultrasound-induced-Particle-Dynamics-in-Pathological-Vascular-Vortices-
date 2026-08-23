# One-off processing of newly-added, tighter-crop TIFF sequences (added directly
# at repo root as "...realtime"/"...Real_Time" folders). Each has a DIFFERENT
# pixel scale than the originally-analyzed frames (verified per-video from its
# own cavity radius), so each crop is calibrated independently.
#
# For each video, produces:
#   - outputs/pdms_<name>_volume_change.csv / plotting/pdms_<name>_volume_change.png
#   - output_images_pdms_<name>_tracking/ -- per-frame overlay PNGs (for a
#     supplementary video), with time + area burned into each frame.

import sys, glob, time
sys.path.insert(0, "scripts/tracking")
import numpy as np, pandas as pd, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pdms_volume_change as v

FPS = 100.0
# Rolling-median smoothing window, as a REAL-TIME duration rather than a fixed
# frame count -- validated at 31 frames/0.31s on the 100 FPS videos. Frame
# count is derived per-job from each job's own fps, so a 20 FPS video gets a
# proportionally smaller window (~6 frames) instead of smoothing over 5x more
# real time than intended (the bug that made vid2023sep's amplitude look
# artificially small the first time around).
SMOOTH_WINDOW_S = 0.31

JOBS = {
    "vid1crop": dict(
        folder="vid_2026-08-09_20-15-21-2_realtime",
        # Actually 1000 FPS with a ~230-240 BPM pump (confirmed by user) -- NOT
        # 100 FPS/4 RPM as originally logged. Smoothing window is overridden to
        # 0.031s (still 31 frames, same as it implicitly was before this fix)
        # rather than the global 0.31s default, which at the true 1000 FPS
        # would be wider than one whole pump cycle (~0.26s) and wash the
        # oscillation out entirely.
        fps=1000.0,
        smooth_window_s=0.031,
        pattern="*.tif",
        cx=443, cy=322, r0=200, r1=340,
        exclude_deg=[(60, 120)],
        # These crops are auto-contrast-stretched to full 0-255 per frame (unlike
        # the raw originals), so the debris-vs-edge width separation is narrower
        # and frame-dependent -- min_width_px=15 (tuned on frame 0 only) was
        # over-rejecting the true edge on other frames, dropping coverage to
        # ~46% average. Re-tuned across samples spanning the whole sequence:
        # width=8 keeps ~294-299/300 angles with no debris leakage.
        min_prominence=10, min_width_px=8,
        ref_radius_px=268.25,
    ),
    "vid5crop": dict(
        folder="vid_2026-08-10_01-05-57-1_real_time",
        pattern="*.tif",
        cx=500, cy=280, r0=190, r1=340,
        exclude_deg=[(60, 120)],
        min_prominence=6, min_width_px=8,
        ref_radius_px=264.5,
    ),
    "vid2023sep": dict(
        # Different (older, 2023) recording, captured at 20 FPS (confirmed by
        # user) -- NOT 100 FPS like the other two.
        folder="vid_2023-09-06_21-10-03-1_Real_Time",
        fps=20.0,
        pattern="*.tif",
        cx=481, cy=470, r0=400, r1=510,
        exclude_deg=[(45, 130)],  # neck opening into the straight channel
        # Lower contrast than vid1crop and, critically, a gas pocket that GROWS
        # over the sequence and swallows the upper-cavity trough signal (visually
        # confirmed: frame 0 has a small dark corner patch, frame ~1531 has it
        # covering the whole upper half) -- coverage genuinely declines late in
        # the sequence rather than this being a threshold-tuning artifact.
        min_prominence=10, min_width_px=6,
        min_coverage=0.20,
        ref_radius_px=447.25,
    ),
    "vid5edited": dict(
        # Video 5, re-exported with adjusted lighting and a tighter crop than
        # vid5crop (different pixel scale again -- calibrated from its own
        # cavity radius). Same recording as vid5crop, but this crop's growing
        # bubble sits close enough to the true wall that the "first qualifying
        # trough" scan sometimes locks onto the bubble's own (irregular, closer)
        # edge instead of the wall over a big chunk of the right side -- a WRONG
        # detection, not a missing one, so it can't be fixed by better gap-
        # filling. Fix: measure only the 172-193deg arc (left side, verified
        # detected in 100% of sampled frames across the whole video -- always
        # far from both the neck and the bubble) and use it as a single-
        # direction radius proxy instead of full 360deg area integration.
        folder="vid_2026-08-10_01-05-57-1_real_time_edited",
        pattern="*.tif",
        cx=374, cy=292, r0=200, r1=330,
        clean_arc_deg=(172, 193),
        min_prominence=10, min_width_px=8,
        ref_radius_px=263.25,
    ),
    "vid5edited_black": dict(
        # Same crop/geometry/fix as vid5edited, but pixel-inverted (bright wall
        # on dark background) -- Hough-confirmed identical circle position/
        # radius to vid5edited once re-inverted, so same cx/cy/r0/r1.
        folder="vid_2026-08-10_01-05-57-1_real_time_edited_Black",
        pattern="*.tif",
        invert=True,
        cx=374, cy=292, r0=200, r1=330,
        clean_arc_deg=(172, 193),
        min_prominence=10, min_width_px=8,
        ref_radius_px=263.25,
    ),
}

def run_clean_arc(name, cfg):
    # Single-direction radius proxy for videos where a growing bubble sits
    # close enough to the true wall to fool the full-boundary scan into
    # locking onto the bubble's own edge over part of the circle -- a WRONG
    # detection that better gap-filling can't fix, since it isn't "missing"
    # data. Measuring one arc that's verified clear of both the neck and the
    # bubble for the whole video sacrifices sensitivity to non-uniform
    # deformation (this cavity type is known not to expand perfectly
    # uniformly under pressure) in exchange for actually being correct.
    t0 = time.time()
    fps = cfg.get("fps", FPS)
    invert = cfg.get("invert", False)
    px_per_um = 2 * cfg["ref_radius_px"] / 300.0
    lo, hi = cfg["clean_arc_deg"]
    exclude = [(hi, lo)]  # exclude everything OUTSIDE (lo, hi)
    print(f"\n=== {name}: {cfg['folder']}  (calibration: {px_per_um:.4f} px/um, "
          f"clean-arc method: {lo}-{hi} deg) ===")

    files = sorted(glob.glob(f"{cfg['folder']}/{cfg['pattern']}"))
    print(f"{len(files)} frames")

    out_dir = f"output_images_pdms_{name}_tracking"
    import os
    os.makedirs(out_dir, exist_ok=True)

    def load(f):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        return 255 - img if invert else img

    rows = []
    for i, f in enumerate(files):
        img = load(f)
        angles, radii = v.measure_boundary(
            img, cfg["cx"], cfg["cy"], cfg["r0"], cfg["r1"], exclude,
            min_prominence=cfg["min_prominence"], min_width_px=cfg["min_width_px"],
        )
        if len(radii):
            # Report RADIUS directly -- do not square this into an "equivalent
            # circle area" for comparison with the other (true full-boundary)
            # videos' area percentages. That conversion assumes the whole
            # boundary scales proportionally with this one arc, i.e. uniform
            # expansion -- exactly the assumption this cavity type is known
            # NOT to satisfy. A radius %-change and an area %-change are
            # different quantities (area ~ r^2, so a uniform expansion would
            # roughly double the % when squared) and must not be compared
            # directly against the other videos' area-based percentages.
            r_mm = float(np.mean(radii)) / px_per_um / 1000.0
            coverage = len(angles) / (hi - lo + 1)
            label = f"t = {i/fps:5.2f} s   r = {r_mm:.4f} mm (clean-arc proxy)"
        else:
            r_mm, coverage = np.nan, 0.0
            label = f"t = {i/fps:5.2f} s   r = n/a"

        vis = cv2.cvtColor(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)
        for a, r in zip(angles, radii):
            theta = np.radians(a)
            x = cfg["cx"] + r * np.cos(theta); y = cfg["cy"] + r * np.sin(theta)
            cv2.circle(vis, (int(round(x)), int(round(y))), 3, (0, 60, 255), -1)
        cv2.putText(vis, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
        cv2.imwrite(f"{out_dir}/frame_{i:04d}.png", vis)

        rows.append({"frame": i, "time_s": i/fps, "radius_mm": r_mm, "coverage": coverage})
        if i % 400 == 0:
            print(f"  frame {i}/{len(files)}  ({time.time()-t0:.0f}s elapsed)")

    df = pd.DataFrame(rows)
    _finish_clean_arc(name, df, fps, t0, out_dir, lo, hi)

def _finish_clean_arc(name, df, fps, t0, out_dir, lo, hi):
    smooth_frames = max(3, round(SMOOTH_WINDOW_S * fps))
    df["radius_mm_smoothed"] = df["radius_mm"].rolling(
        smooth_frames, center=True, min_periods=max(smooth_frames // 2, 1)).median()
    df.to_csv(f"outputs/pdms_{name}_volume_change.csv", index=False)

    valid = df.dropna(subset=["radius_mm_smoothed"])
    mean_v = valid["radius_mm_smoothed"].mean()
    amp = valid["radius_mm_smoothed"].max() - valid["radius_mm_smoothed"].min()

    raw_valid = df.dropna(subset=["radius_mm"])
    raw_amp = raw_valid["radius_mm"].max() - raw_valid["radius_mm"].min()
    raw_pct = 100 * raw_amp / raw_valid["radius_mm"].mean()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["time_s"], df["radius_mm"], color="0.75", lw=0.5, label="Raw (per-frame)")
    ax.plot(df["time_s"], df["radius_mm_smoothed"], color="#2a78d6", lw=1.6, label="Smoothed (rolling median)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel(f"Radius, {lo}-{hi}° arc only (mm)")
    ax.set_title(f"{name}: single-direction radius proxy (NOT an area -- not directly\n"
                 f"comparable to other videos' area-based %)")
    ax.legend(frameon=False, loc="upper right"); ax.grid(alpha=0.3)
    ax.text(0.99, 0.06, f"smoothed mean {mean_v:.3f} mm  ·  p-p {amp:.4f} mm ({100*amp/mean_v:.1f}% of radius)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="0.35")
    ax.text(0.99, 0.02, f"raw max change (incl. noise): {raw_amp:.4f} mm ({raw_pct:.1f}% of radius)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="0.6")
    fig.tight_layout()
    fig.savefig(f"plotting/pdms_{name}_volume_change.png", dpi=200)
    plt.close(fig)

    print(f"{name}: [clean-arc, RADIUS not area] smoothed mean={mean_v:.4f}mm amp={amp:.4f}mm "
          f"({100*amp/mean_v:.1f}% of radius) raw_amp={raw_amp:.4f}mm ({raw_pct:.1f}% of radius) "
          f"mean_coverage={df['coverage'].mean():.0%}  took {time.time()-t0:.0f}s")
    print(f"Saved: outputs/pdms_{name}_volume_change.csv, plotting/pdms_{name}_volume_change.png, {out_dir}/")

def run(name, cfg):
    if "clean_arc_deg" in cfg:
        run_clean_arc(name, cfg)
        return

    t0 = time.time()
    fps = cfg.get("fps", FPS)
    invert = cfg.get("invert", False)
    px_per_um = 2 * cfg["ref_radius_px"] / 300.0
    print(f"\n=== {name}: {cfg['folder']}  (calibration: {px_per_um:.4f} px/um) ===")

    files = sorted(glob.glob(f"{cfg['folder']}/{cfg['pattern']}"))
    print(f"{len(files)} frames")

    out_dir = f"output_images_pdms_{name}_tracking"
    import os
    os.makedirs(out_dir, exist_ok=True)

    N = v.N_ANGLES
    angle_grid = np.linspace(0, 360, N, endpoint=False)
    step = 360.0 / N

    def load(f):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        return 255 - img if invert else img

    # ---- Pass 1: detect only, build a (frames x angles) radius grid ----
    print("  pass 1/2: detecting boundary...")
    detections = []
    grid = np.full((len(files), N), np.nan)
    for i, f in enumerate(files):
        angles, radii = v.measure_boundary(
            load(f), cfg["cx"], cfg["cy"], cfg["r0"], cfg["r1"], cfg["exclude_deg"],
            min_prominence=cfg["min_prominence"], min_width_px=cfg["min_width_px"],
        )
        detections.append((angles, radii))
        if len(angles):
            idx = np.round(np.asarray(angles) / step).astype(int) % N
            grid[i, idx] = radii
        if i % 400 == 0:
            print(f"    frame {i}/{len(files)}  ({time.time()-t0:.0f}s elapsed)")

    # Fill gaps with the NEAREST real detection in time (bidirectional), not a
    # forward-only hold and not spatial (angular) interpolation. Spatial
    # interpolation redraws a straight line between whatever neighboring
    # angles happen to be valid that frame -- since an occluding bubble's
    # shape shifts frame to frame, those neighbors shift too, injecting
    # frame-to-frame jitter even though the true wall barely moves between
    # frames (100 fps). A forward-only hold fixes the jitter but still leaves
    # a spurious high-then-decaying start wherever an angle's first-ever
    # detection doesn't happen until partway through the video (validated on
    # vid5crop: coverage of angles-ever-seen kept climbing for the first ~3s).
    # Nearest-in-time removes both problems; only angles NEVER detected in the
    # whole sequence fall back to spatial interpolation (per-frame, below).
    filled = grid.copy()
    for a_idx in range(N):
        col = pd.Series(grid[:, a_idx])
        if col.notna().sum() == 0:
            continue
        filled[:, a_idx] = col.interpolate(method="nearest", limit_direction="both").ffill().bfill().to_numpy()

    # ---- Pass 2: compute area/coverage, draw overlays, save frames ----
    print("  pass 2/2: computing area + saving overlay frames...")
    rows = []
    for i, f in enumerate(files):
        angles, radii = detections[i]

        if len(angles) >= N * cfg.get("min_coverage", 0.4):
            row_full = filled[i].copy()
            valid = ~np.isnan(row_full)
            if not valid.all():
                row_full[~valid] = np.interp(angle_grid[~valid], angle_grid[valid], row_full[valid], period=360.0)
            area_px2 = 0.5 * np.sum(row_full ** 2) * np.radians(step)
            coverage = len(angles) / N  # fraction freshly measured this frame (diagnostic)
            area_mm2 = area_px2 / px_per_um**2 / 1e6
            label = f"t = {i/fps:5.2f} s   area = {area_mm2:.4f} mm2"
        else:
            area_px2, coverage, area_mm2 = np.nan, len(angles)/N, np.nan
            label = f"t = {i/fps:5.2f} s   area = n/a"

        vis = cv2.cvtColor(cv2.normalize(load(f), None, 0, 255, cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)
        for a, r in zip(angles, radii):
            theta = np.radians(a)
            x = cfg["cx"] + r * np.cos(theta); y = cfg["cy"] + r * np.sin(theta)
            cv2.circle(vis, (int(round(x)), int(round(y))), 3, (0, 60, 255), -1)
        cv2.putText(vis, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
        cv2.imwrite(f"{out_dir}/frame_{i:04d}.png", vis)

        rows.append({"frame": i, "time_s": i/fps, "area_px2": area_px2,
                      "area_mm2": area_mm2, "coverage": coverage})
        if i % 400 == 0:
            print(f"    frame {i}/{len(files)}  ({time.time()-t0:.0f}s elapsed)")

    df = pd.DataFrame(rows)
    _finish(name, cfg, df, fps, t0, out_dir)

def _finish(name, cfg, df, fps, t0, out_dir, title_suffix=""):
    # Rolling-median smoothed trend, same real-time window as video5's
    # area_mm2_smoothed -- the reliable signal on noisier footage. Per-job
    # override available (see vid1crop) for videos whose pump cycle is too
    # fast for the default 0.31s window to make sense.
    smooth_window_s = cfg.get("smooth_window_s", SMOOTH_WINDOW_S)
    smooth_frames = max(3, round(smooth_window_s * fps))
    df["area_mm2_smoothed"] = df["area_mm2"].rolling(
        smooth_frames, center=True, min_periods=max(smooth_frames // 2, 1)).median()
    df.to_csv(f"outputs/pdms_{name}_volume_change.csv", index=False)

    valid = df.dropna(subset=["area_mm2_smoothed"])
    mean_v = valid["area_mm2_smoothed"].mean()
    amp = valid["area_mm2_smoothed"].max() - valid["area_mm2_smoothed"].min()

    raw_valid = df.dropna(subset=["area_mm2"])
    raw_amp = raw_valid["area_mm2"].max() - raw_valid["area_mm2"].min()
    raw_pct = 100 * raw_amp / raw_valid["area_mm2"].mean()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["time_s"], df["area_mm2"], color="0.75", lw=0.5, label="Raw (per-frame)")
    ax.plot(df["time_s"], df["area_mm2_smoothed"], color="#2a78d6", lw=1.6, label="Smoothed (rolling median)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Cross-sectional area (mm²)")
    ax.set_title(f"{name}: cavity volume-change proxy (new crop){title_suffix}")
    ax.legend(frameon=False); ax.grid(alpha=0.3)
    ax.text(0.99, 0.06, f"smoothed mean {mean_v:.3f} mm²  ·  p-p {amp:.3f} mm² ({100*amp/mean_v:.1f}%)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="0.35")
    ax.text(0.99, 0.02, f"raw max change (incl. noise): {raw_amp:.3f} mm² ({raw_pct:.1f}%)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="0.6")
    fig.tight_layout()
    fig.savefig(f"plotting/pdms_{name}_volume_change.png", dpi=200)
    plt.close(fig)

    print(f"{name}: smoothed mean={mean_v:.4f}mm2 amp={amp:.4f}mm2 ({100*amp/mean_v:.1f}%) "
          f"raw_amp={raw_amp:.4f}mm2 ({raw_pct:.1f}%) mean_coverage={df['coverage'].mean():.0%}  took {time.time()-t0:.0f}s")
    print(f"Saved: outputs/pdms_{name}_volume_change.csv, plotting/pdms_{name}_volume_change.png, {out_dir}/")

if __name__ == "__main__":
    names = sys.argv[1:] or list(JOBS.keys())
    for name in names:
        run(name, JOBS[name])
