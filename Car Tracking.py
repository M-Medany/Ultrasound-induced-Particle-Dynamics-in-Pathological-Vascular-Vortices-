import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt

try:
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename
except Exception:  # On headless machines Tk can be missing.
    Tk = None
    askopenfilename = None


TRACKER_PREFERENCE_ORDER = ("CSRT", "KCF", "MOSSE", "MIL")


def create_tracker(tracker_name: str) -> "cv2.Tracker":
    """Instantiate the requested tracker if it exists in the current OpenCV build."""
    name = tracker_name.upper()

    if name == "CSRT":
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        raise RuntimeError("CSRT requires opencv-contrib-python.")

    if name == "KCF":
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
            return cv2.legacy.TrackerKCF_create()
        if hasattr(cv2, "TrackerKCF_create"):
            return cv2.TrackerKCF_create()
        raise RuntimeError("KCF tracker is unavailable in this build.")

    if name == "MOSSE":
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMOSSE_create"):
            return cv2.legacy.TrackerMOSSE_create()
        if hasattr(cv2, "TrackerMOSSE_create"):
            return cv2.TrackerMOSSE_create()
        raise RuntimeError("MOSSE tracker is unavailable in this build.")

    if name == "MIL":
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMIL_create"):
            return cv2.legacy.TrackerMIL_create()
        if hasattr(cv2, "TrackerMIL_create"):
            return cv2.TrackerMIL_create()
        raise RuntimeError("MIL tracker is unavailable in this build.")

    raise ValueError(f"Unsupported tracker: {tracker_name}")


def select_tracker(primary: str, fallbacks: Iterable[str]) -> Tuple[str, "cv2.Tracker"]:
    attempted: List[str] = []
    messages: List[str] = []

    for candidate in [primary, *fallbacks]:
        candidate_upper = candidate.upper()
        if candidate_upper in attempted:
            continue
        attempted.append(candidate_upper)
        try:
            tracker = create_tracker(candidate_upper)
            return candidate_upper, tracker
        except (RuntimeError, ValueError) as exc:
            messages.append(f"{candidate_upper}: {exc}")

    raise RuntimeError(
        "Unable to create any tracker. Enable opencv-contrib-python or choose a different tracker.\n"
        + "\n".join(messages)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track an object in a video and plot its trajectory."
    )
    parser.add_argument(
        "video",
        type=Path,
        nargs="?",
        help="Path to the input video file (e.g. MP4). If omitted a file dialog opens.",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path.cwd() / "cideo tracking",
        help="Directory to search when opening the file dialog (defaults to ./cideo tracking).",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="CSRT",
        help="Preferred tracker (default: CSRT). Other options include KCF, MOSSE, MIL.",
    )
    parser.add_argument(
        "--fallback-trackers",
        type=str,
        nargs="*",
        default=[name for name in TRACKER_PREFERENCE_ORDER if name != "CSRT"],
        metavar="NAME",
        help="Tracker names to try if the preferred tracker is unavailable.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Skip the real-time tracking preview window (useful for headless runs).",
    )
    parser.add_argument(
        "--display-skip",
        type=int,
        default=1,
        metavar="N",
        help="Render only every Nth frame in the preview window to speed up playback (default: 1).",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=None,
        help="Optional path to save the trajectory plot instead of (or in addition to) showing it.",
    )
    return parser.parse_args()


def resolve_video_path(video_arg: Optional[Path], initial_dir: Path) -> Path:
    if video_arg is not None:
        return video_arg

    if askopenfilename is None:
        raise RuntimeError(
            "No video path provided and Tkinter is unavailable to open a file dialog."
        )

    root = Tk()
    root.withdraw()

    initial_directory = initial_dir if initial_dir.exists() else Path.cwd()
    file_path = askopenfilename(
        title="Select video for tracking",
        initialdir=str(initial_directory),
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()

    if not file_path:
        raise RuntimeError("No video selected.")
    return Path(file_path)


def bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def main() -> None:
    args = parse_args()
    if args.display_skip < 1:
        raise ValueError("--display-skip must be >= 1")
    video_path = resolve_video_path(args.video, args.video_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    tracker_name, tracker = select_tracker(args.tracker, args.fallback_trackers)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Could not read the first frame from the video.")

    print("Drag to select the object, press ENTER/SPACE to confirm, or press C to cancel.")
    selection_window = "Select ROI"
    cv2.namedWindow(selection_window, cv2.WINDOW_NORMAL)
    bbox = cv2.selectROI(selection_window, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(selection_window)

    if bbox == (0, 0, 0, 0):
        cap.release()
        print("ROI selection cancelled. Exiting without tracking.")
        return

    tracker.init(frame, bbox)

    trajectory: List[Tuple[float, float]] = [bbox_center(bbox)]

    display_window = f"{tracker_name} Tracking"
    show_preview = not args.no_display
    if show_preview:
        cv2.namedWindow(display_window, cv2.WINDOW_AUTOSIZE)

    frame_idx = 0
    max_trail_segments = 200

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        tracked, tracked_bbox = tracker.update(frame)
        if tracked:
            cx, cy = bbox_center(tracked_bbox)
            trajectory.append((cx, cy))

        if not show_preview:
            continue

        if args.display_skip > 1 and frame_idx % args.display_skip != 0:
            continue

        display_frame = frame
        if tracked:
            x, y, w, h = [int(v) for v in tracked_bbox]
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(display_frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)

            for i in range(1, min(len(trajectory), max_trail_segments)):
                pt1 = (int(trajectory[-i][0]), int(trajectory[-i][1]))
                pt2 = (int(trajectory[-i - 1][0]), int(trajectory[-i - 1][1]))
                cv2.line(display_frame, pt1, pt2, (255, 0, 0), 1)
        else:
            cv2.putText(
                display_frame,
                "Tracking failure detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        cv2.imshow(display_window, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    if show_preview:
        cv2.destroyWindow(display_window)
    cv2.destroyAllWindows()

    if len(trajectory) < 2:
        print("Not enough points tracked to plot a trajectory.")
        return

    xs, ys = zip(*trajectory)

    plt.figure(figsize=(8, 6))
    plt.plot(xs, ys, marker="o", markersize=3, linewidth=1, label="Trajectory")
    plt.scatter(xs[0], ys[0], c="green", label="Start")
    plt.scatter(xs[-1], ys[-1], c="red", label="End")
    plt.gca().invert_yaxis()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(f"Tracked Trajectory ({tracker_name})")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.legend()
    plt.tight_layout()

    if args.output_figure is not None:
        plt.savefig(args.output_figure, dpi=300)
        print(f"Saved trajectory plot to {args.output_figure}")

    plt.show()


if __name__ == "__main__":
    main()
