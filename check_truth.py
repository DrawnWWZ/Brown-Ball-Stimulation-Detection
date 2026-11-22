"""Exercise the tracking CLI on noisy synthetic images with known particle identities."""
import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

FRAMES = 80
FPS = 20
UM_PER_PX = 0.2
DIAMETER = 5.0  # FWHM in pixels
N_MOVING = 4
N_STATIC = 2
TRACKER = Path(__file__).with_name("track.py")


def render(folder, seed):
    """Return camera-plane centers for 4 mobile singles, 2 fixed singles and 2 clumps."""
    folder.mkdir()
    rng = np.random.default_rng(seed)
    starts = np.array([(35, 38), (89, 37), (141, 39), (193, 43),
                       (51, 97), (118, 98), (63, 152), (166, 147)], dtype=float)
    increments = rng.normal(0, 0.9, (FRAMES, len(starts), 2))
    increments[0] = 0
    increments[:, N_MOVING:N_MOVING + N_STATIC] = 0
    centers = starts + np.cumsum(increments, axis=0)
    # Small common stage wobble challenges classification of the fixed spots.
    centers += rng.normal(0, 0.12, (FRAMES, 1, 2))
    yy, xx = np.mgrid[:192, :232]
    background = 150 + 14 * np.sin(xx / 55) + 11 * yy / 192
    for i, positions in enumerate(centers):
        img = background + 2 * np.sin(i / 15) + rng.normal(0, 5, background.shape)
        for j, (x, y) in enumerate(positions):
            # Equal-width touching lobes form one doublet. The other clump is round.
            offsets = (-2.6, 2.6) if j == 6 else (0,)
            fwhm = 9.5 if j == 7 else DIAMETER
            for dx in offsets:
                dist2 = (xx - x - dx) ** 2 + (yy - y) ** 2
                img -= (62 + 6 * np.sin(i / 12 + j)) * np.exp(-4 * np.log(2) * dist2 / fwhm ** 2)
        impulse = rng.random(background.shape) < 0.0004
        img[impulse] += rng.choice([-75, 75], impulse.sum())
        path = folder / f"frame_{i:04d}.png"
        if not cv2.imwrite(str(path), np.rint(img).clip(0, 255).astype(np.uint8)):
            raise OSError(f"Could not write {path}")
    return centers


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def check(folder, seed):
    frames = folder / "frames"
    centers = render(frames, seed)
    out = folder / "results"
    subprocess.run([sys.executable, str(TRACKER), str(frames), "--diameter-px", str(DIAMETER),
                    "--fps", str(FPS), "--um-per-px", str(UM_PER_PX), "--out", str(out)], check=True)
    moving_ids = {}
    located = set()
    rejected_static = set()
    rejected_clumps = set()
    radial_errors = []
    false_positives = stationary_leaks = clump_leaks = 0
    for row in read_csv(out / "tracks.csv"):
        frame = int(row["frame"])
        location = np.array([float(row["x_px"]), float(row["y_px"])])
        distances = np.linalg.norm(location - centers[frame], axis=1)
        bead = int(distances.argmin())
        radius = distances[bead]
        key = (frame, bead)
        if row["status"] == "aggregate" and bead >= 6 and radius <= 8:
            rejected_clumps.add(key)
        if row["status"] == "stationary" and 4 <= bead < 6 and radius <= 2:
            rejected_static.add(key)
        if row["status"] != "moving":
            continue
        if bead < N_MOVING and radius <= 2 and key not in located:
            located.add(key)
            moving_ids[frame, row["track_id"]] = bead
            radial_errors.append(radius)
        else:
            false_positives += 1
            stationary_leaks += int(4 <= bead < 6 and radius <= 2)
            clump_leaks += int(bead >= 6 and radius <= 8)

    speed_errors = []
    bad_steps = 0
    for row in read_csv(out / "steps.csv"):
        a, b = int(row["frame_from"]), int(row["frame_to"])
        if not 0 <= a < b < FRAMES:
            raise ValueError(f"Invalid exported frame interval [{a}, {b}]")
        ka = moving_ids.get((a, row["track_id"]))
        kb = moving_ids.get((b, row["track_id"]))
        if ka is None or kb != ka:
            bad_steps += 1
            continue
        speed_px = float(row["speed_px_s"])
        speed_um = float(row["speed_um_s"])
        if not np.isclose(speed_um, speed_px * UM_PER_PX, atol=1e-5):
            raise ValueError("Speed calibration disagrees with the supplied micrometers per pixel")
        truth = np.linalg.norm(centers[b, ka] - centers[a, ka]) * FPS / (b - a)
        speed_errors.append(abs(speed_px - truth))
    scores = {
        "seed": seed,
        "moving_position_recall": len(located) / (N_MOVING * FRAMES),
        "false_moving_positions": false_positives,
        "stationary_positions_in_moving": stationary_leaks,
        "clump_positions_in_moving": clump_leaks,
        "explicit_stationary_fraction": len(rejected_static) / (N_STATIC * FRAMES),
        "explicit_clump_fraction": len(rejected_clumps) / (2 * FRAMES),
        "radial_rmse_px": float(np.sqrt(np.mean(np.square(radial_errors)))) if radial_errors else None,
        "matched_speed_steps": len(speed_errors),
        "false_or_mislinked_speed_steps": bad_steps,
        "mean_absolute_speed_error_px_s": float(np.mean(speed_errors)) if speed_errors else None,
    }
    print(json.dumps(scores, indent=2), flush=True)
    good = (scores["moving_position_recall"] >= 0.90 and false_positives == 0 and bad_steps == 0
            and scores["explicit_stationary_fraction"] >= 0.95 and scores["explicit_clump_fraction"] >= 0.95
            and len(speed_errors) >= 0.85 * N_MOVING * (FRAMES - 1)
            and scores["radial_rmse_px"] <= 0.7 and scores["mean_absolute_speed_error_px_s"] <= 6.0)
    if not good:
        raise SystemExit("Synthetic tracking check failed. Inspect the result counts above.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--keep", type=Path, help="create a new directory with the frames and tracking output")
    args = parser.parse_args()
    if args.keep:
        args.keep.mkdir(parents=True, exist_ok=False)
        check(args.keep, args.seed)
    else:
        with tempfile.TemporaryDirectory() as temp:
            check(Path(temp), args.seed)


if __name__ == "__main__":
    main()
