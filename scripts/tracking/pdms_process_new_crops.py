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

JOBS = {
    "vid1crop": dict(
        folder="vid_2026-08-09_20-15-21-2_realtime",
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
}

def run(name, cfg):
    t0 = time.time()
    fps = cfg.get("fps", FPS)
    px_per_um = 2 * cfg["ref_radius_px"] / 300.0
    print(f"\n=== {name}: {cfg['folder']}  (calibration: {px_per_um:.4f} px/um) ===")

    files = sorted(glob.glob(f"{cfg['folder']}/{cfg['pattern']}"))
    print(f"{len(files)} frames")

    out_dir = f"output_images_pdms_{name}_tracking"
    import os
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for i, f in enumerate(files):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        angles, radii = v.measure_boundary(
            img, cfg["cx"], cfg["cy"], cfg["r0"], cfg["r1"], cfg["exclude_deg"],
            min_prominence=cfg["min_prominence"], min_width_px=cfg["min_width_px"],
        )

        vis = cv2.cvtColor(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)
        for a, r in zip(angles, radii):
            theta = np.radians(a)
            x = cfg["cx"] + r * np.cos(theta); y = cfg["cy"] + r * np.sin(theta)
            cv2.circle(vis, (int(round(x)), int(round(y))), 3, (0, 60, 255), -1)

        if len(angles) >= v.N_ANGLES * cfg.get("min_coverage", 0.4):
            area_px2, coverage = v.polar_area(angles, radii, v.N_ANGLES)
            area_mm2 = area_px2 / px_per_um**2 / 1e6
            label = f"t = {i/fps:5.2f} s   area = {area_mm2:.4f} mm2"
        else:
            area_px2, coverage, area_mm2 = np.nan, len(angles)/v.N_ANGLES, np.nan
            label = f"t = {i/fps:5.2f} s   area = n/a"

        cv2.putText(vis, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
        cv2.imwrite(f"{out_dir}/frame_{i:04d}.png", vis)

        rows.append({"frame": i, "time_s": i/fps, "area_px2": area_px2,
                      "area_mm2": area_mm2, "coverage": coverage})
        if i % 200 == 0:
            print(f"  frame {i}/{len(files)}  ({time.time()-t0:.0f}s elapsed)")

    df = pd.DataFrame(rows)
    df.to_csv(f"outputs/pdms_{name}_volume_change.csv", index=False)

    valid = df.dropna(subset=["area_mm2"])
    mean_v = valid["area_mm2"].mean(); amp = valid["area_mm2"].max() - valid["area_mm2"].min()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["time_s"], df["area_mm2"], lw=1.0, color="#2a78d6")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Cross-sectional area (mm²)")
    ax.set_title(f"{name}: cavity volume-change proxy (new crop)")
    ax.grid(alpha=0.3)
    ax.text(0.99, 0.02, f"mean {mean_v:.3f} mm²  ·  p-p {amp:.3f} mm² ({100*amp/mean_v:.1f}%)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="0.35")
    fig.tight_layout()
    fig.savefig(f"plotting/pdms_{name}_volume_change.png", dpi=200)
    plt.close(fig)

    print(f"{name}: mean={mean_v:.4f}mm2 amp={amp:.4f}mm2 ({100*amp/mean_v:.1f}%) "
          f"mean_coverage={df['coverage'].mean():.0%}  took {time.time()-t0:.0f}s")
    print(f"Saved: outputs/pdms_{name}_volume_change.csv, plotting/pdms_{name}_volume_change.png, {out_dir}/")

if __name__ == "__main__":
    names = sys.argv[1:] or list(JOBS.keys())
    for name in names:
        run(name, JOBS[name])
