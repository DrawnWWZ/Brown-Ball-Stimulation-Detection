"""Measure finite-frame bead displacements in bright-field microscopy recordings."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from scipy.spatial import cKDTree

from detection import detect


IMAGES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
COLORS = {"moving": (80, 230, 40), "stationary": (250, 170, 50),
          "aggregate": (40, 80, 255), "short": (130, 130, 130),
          "candidate": (50, 225, 245), "edge": (195, 100, 200)}


def image_frame(path):
    if path.suffix.lower() in {".tif", ".tiff"} and cv2.imcount(str(path)) > 1:
        raise ValueError("Export the TIFF stack as a numbered image sequence.")
    frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise ValueError(f"Cannot read image {path}")
    return frame


def frames_from(path):
    """Retain the course tracker convention: one sequence, consecutive suffix numbers."""
    if path.is_dir():
        members = [p for p in path.iterdir() if p.suffix.lower() in IMAGES]
        numbered = {}
        for p in members:
            match = re.search(r"(\d+)$", p.stem)
            if not match or int(match[1]) in numbered:
                raise ValueError("Use unique trailing frame numbers in a sequence directory.")
            numbered[int(match[1])] = p
        numbers = sorted(numbered)
        if not numbers or numbers[-1] - numbers[0] + 1 != len(numbers):
            raise ValueError("Empty sequence or missing frame. Supply a contiguous sequence.")
        for n in numbers:
            yield image_frame(numbered[n])
    elif path.suffix.lower() in IMAGES:
        yield image_frame(path)
    else:
        movie = cv2.VideoCapture(str(path))
        try:
            if not movie.isOpened():
                raise ValueError(f"Cannot open {path}")
            expected = int(movie.get(cv2.CAP_PROP_FRAME_COUNT))
            read = 0
            while True:
                ok, frame = movie.read()
                if not ok:
                    break
                read += 1
                yield frame
            if not read or (expected > 0 and read != expected):
                raise ValueError(f"Decoded {read} video frames, header reports {expected}.")
        finally:
            movie.release()


def channel(frame):
    if frame.dtype != np.uint8:
        raise ValueError("Supply 8-bit images. Convert higher depths with one fixed intensity range.")
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] in (3, 4):
        return frame[:, :, 1]
    raise ValueError("Unsupported image channels.")


def header_fps(path):
    if path.is_dir() or path.suffix.lower() in IMAGES:
        return None
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        return float(fps) if np.isfinite(fps) and fps > 0 else None
    finally:
        cap.release()


def link(previous, current, radius):
    """Only extend mutually unique neighbors in consecutive frames."""
    neighbors = cKDTree(previous).query_ball_tree(cKDTree(current), radius)
    hits = np.zeros(len(current), dtype=int)
    for candidates in neighbors:
        hits[candidates] += 1
    return [(i, c[0]) for i, c in enumerate(neighbors) if len(c) == 1 and hits[c[0]] == 1]


def registration_image(gray):
    values = gray.astype(np.float32)
    return cv2.GaussianBlur(values, (0, 0), 1) - cv2.GaussianBlur(values, (0, 0), 12)


def track(args):
    rows, shifts = [], []
    previous = np.empty((0, 2))
    previous_ids = []
    next_id = 1
    reference = window = shape = None
    for n, frame in enumerate(frames_from(args.input)):
        gray = channel(frame)
        if n == 0:
            shape = gray.shape
        if gray.shape != shape:
            raise ValueError("Frame dimensions changed.")
        shift = np.zeros(2)
        response = None
        if args.stabilize:
            current = registration_image(gray)
            if n == 0:
                reference = current
                window = cv2.createHanningWindow(shape[::-1], cv2.CV_32F)
            else:
                # OpenCV applies a supplied window in-place on some builds.
                xy, response = cv2.phaseCorrelate(reference * window, current * window)
                shift = np.asarray(xy)
                if not np.isfinite(shift).all() or not np.isfinite(response) or response < 0.15:
                    raise ValueError(f"Registration to frame 0 unreliable at frame {n}. "
                                     "Use a shorter stable clip or omit --stabilize.")
        shifts.append((n, *shift, response))
        observations = detect(gray, args.diameter_px, args.polarity, args.snr)
        candidates = [i for i, r in enumerate(observations) if r["status"] == "candidate"]
        xy = np.array([(observations[i]["x"], observations[i]["y"]) for i in candidates],
                      dtype=float).reshape(-1, 2) - shift
        ids = [None] * len(candidates)
        for i, j in link(previous, xy, args.link_px):
            ids[j] = previous_ids[i]
        for j, i in enumerate(candidates):
            if ids[j] is None:
                ids[j] = next_id
                next_id += 1
            observations[i]["track_id"] = ids[j]
        for r in observations:
            rows.append(dict(frame=n, track_id=r.get("track_id"), x_px=r["x"], y_px=r["y"],
                             x_corrected_px=r["x"] - shift[0], y_corrected_px=r["y"] - shift[1],
                             diameter_px=r["diameter_px"], status=r["status"]))
        previous, previous_ids = xy, ids
    return rows, shifts


def classify(rows, nframes, args):
    tracks = defaultdict(list)
    for r in rows:
        if r["track_id"] is not None:
            tracks[r["track_id"]].append(r)
    steps = []
    for tid, group in tracks.items():
        if nframes == 1:
            continue
        xy = np.array([(r["x_corrected_px"], r["y_corrected_px"]) for r in group])
        # Unlike endpoint distance, this retains paths that return to the origin.
        low, high = np.percentile(xy, [10, 90], axis=0)
        if len(group) < args.min_frames:
            status = "short"
        elif np.linalg.norm(high - low) <= args.motion_px:
            status = "stationary"
        else:
            status = "moving"
        for r in group:
            r["status"] = status
        if status == "moving":
            for a, b, (dx, dy) in zip(group, group[1:], np.diff(xy, axis=0)):
                length = np.hypot(dx, dy)
                steps.append(dict(track_id=tid, frame_from=a["frame"], frame_to=b["frame"],
                                  dx_px=dx, dy_px=dy, step_px=length,
                                  speed_px_s=length * args.fps if args.fps else None,
                                  speed_um_s=length * args.fps * args.um_per_px
                                  if args.fps and args.um_per_px else None))
    return steps


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def preview(gray, rows, frame_no):
    # Use the same display stretch under every label. The detector uses the original intensities.
    lo, hi = np.percentile(gray, [1, 99])
    shown = np.clip((gray.astype(float) - lo) * 255 / max(hi - lo, 1), 0, 255).astype("uint8")
    panel = cv2.cvtColor(shown, cv2.COLOR_GRAY2BGR)
    for r in rows:
        center = (round(r["x_px"]), round(r["y_px"]))
        radius = max(3, round(r["diameter_px"] / 2) + 2)
        cv2.circle(panel, center, radius, COLORS[r["status"]], 1, cv2.LINE_AA)
    original = cv2.cvtColor(shown, cv2.COLOR_GRAY2BGR)
    pair = np.hstack([original, panel])
    canvas = np.full((pair.shape[0] + 62, max(540, pair.shape[1]), 3), 28, np.uint8)
    canvas[30:30 + pair.shape[0], :pair.shape[1]] = pair
    for x, title in ((4, f"Raw | {frame_no}"), (gray.shape[1] + 4, "Detections")):
        cv2.putText(canvas, title, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
    x = 6
    for label, color in COLORS.items():
        y = canvas.shape[0] - 12
        cv2.circle(canvas, (x + 4, y - 3), 4, color, -1)
        cv2.putText(canvas, label, (x + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)
        x += 21 + cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)[0][0]
    return canvas


def run(args):
    rows, shifts = track(args)
    nframes = len(shifts)
    steps = classify(rows, nframes, args)
    track_status = {r["track_id"]: r["status"] for r in rows if r["track_id"] is not None}
    summary = dict(input=args.input.name, frames=nframes, fps=args.fps, um_per_px=args.um_per_px,
                   diameter_px=args.diameter_px, snr=args.snr, link_px=args.link_px,
                   motion_px=args.motion_px, min_frames=args.min_frames, stabilized=args.stabilize,
                   polarity=args.polarity, detections_by_status=dict(Counter(r["status"] for r in rows)),
                   segments_by_status=dict(Counter(track_status.values())), accepted_steps=len(steps),
                   preview_frame=nframes // 2)
    for key in ("step_px", "speed_px_s", "speed_um_s"):
        vals = [s[key] for s in steps if s[key] is not None]
        summary["median_" + key] = float(np.median(vals)) if vals else None
    by_frame = defaultdict(list)
    for r in rows:
        by_frame[r["frame"]].append(r)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise ValueError("Choose a new output directory to keep earlier results intact.")
    with TemporaryDirectory(dir=args.out.parent) as tmp:
        out = Path(tmp) / "output"
        out.mkdir()
        write_csv(out / "tracks.csv", ["frame", "track_id", "x_px", "y_px", "x_corrected_px",
                                       "y_corrected_px", "diameter_px", "status"], rows)
        write_csv(out / "steps.csv", ["track_id", "frame_from", "frame_to", "dx_px", "dy_px",
                                      "step_px", "speed_px_s", "speed_um_s"], steps)
        if args.stabilize:
            write_csv(out / "registration.csv", ["frame", "shift_x_px", "shift_y_px", "response"],
                      [dict(frame=n, shift_x_px=x, shift_y_px=y, response=q) for n, x, y, q in shifts])
        writer = None
        try:
            for n, frame in enumerate(frames_from(args.input)):
                middle = n == summary["preview_frame"]
                if not middle and not args.overlay_video:
                    continue
                marked = preview(channel(frame), by_frame[n], n)
                if middle:
                    if not cv2.imwrite(str(out / "preview.png"), marked):
                        raise OSError("Cannot save preview.")
                    if not args.overlay_video:
                        break
                if args.overlay_video:
                    if writer is None:
                        h, w = marked.shape[:2]
                        writer = cv2.VideoWriter(str(out / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 args.fps or 10.0, (w + w % 2, h + h % 2))
                        if not writer.isOpened():
                            raise OSError("Video writer unavailable. Run without --overlay-video.")
                    h, w = marked.shape[:2]
                    writer.write(cv2.copyMakeBorder(marked, 0, h % 2, 0, w % 2, cv2.BORDER_CONSTANT))
        finally:
            if writer is not None:
                writer.release()
        (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        out.rename(args.out)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--out", required=True, type=Path, help="new result directory")
    p.add_argument("--diameter-px", required=True, type=float, help="single-bead FWHM in pixels")
    p.add_argument("--polarity", choices=["dark", "bright"], default="dark")
    p.add_argument("--snr", type=float, default=5.0)
    p.add_argument("--fps", type=float, help="uniform acquisition rate; overrides video metadata")
    p.add_argument("--um-per-px", type=float, help="stage micrometer calibration")
    p.add_argument("--link-px", type=float, help="maximum step, default one bead diameter")
    p.add_argument("--min-frames", type=int, default=8)
    p.add_argument("--motion-px", type=float, help="minimum robust track span, default 0.25 diameters")
    p.add_argument("--overlay-video", action="store_true", help="save annotated MP4 for visual review")
    p.add_argument("--stabilize", action="store_true", help="align using fixed field texture")
    args = p.parse_args()
    if not args.input.exists():
        p.error("Input does not exist.")
    if args.out.exists():
        p.error("Choose a new output directory.")
    if args.fps is None:
        args.fps = header_fps(args.input)
    if args.link_px is None:
        args.link_px = args.diameter_px
    if args.motion_px is None:
        args.motion_px = 0.25 * args.diameter_px
    params = [args.diameter_px, args.snr, args.link_px, args.motion_px, args.fps, args.um_per_px]
    if any(not np.isfinite(v) or v <= 0 for v in params if v is not None) or args.min_frames < 2:
        p.error("Use finite positive parameters and at least two frames per track.")
    try:
        run(args)
    except (OSError, ValueError, cv2.error) as err:
        p.exit(1, f"{err}\n")


if __name__ == "__main__":
    main()
