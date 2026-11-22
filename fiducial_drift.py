"""Measure image translation from fixed features and check it on held-out features."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from track import channel, frames_from


def xy(text):
    parts = text.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Use integer x,y coordinates.")
    try:
        return tuple(map(int, parts))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use integer x,y coordinates.") from exc


def patch(frame, center, radius):
    x, y = center
    return frame[y - radius:y + radius + 1, x - radius:x + radius + 1]


def quadratic_offset(left, peak, right):
    curvature = left + right - 2 * peak
    if curvature >= -1e-8:
        return 0.
    return np.clip(0.5 * (left - right) / curvature, -.5, .5)


def locate(frame, center, template, search, radius):
    region = patch(frame, center, radius + search)
    score = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
    _, strength, _, (x, y) = cv2.minMaxLoc(score)
    offset = np.full(2, np.nan)
    if 0 < x < 2 * search and 0 < y < 2 * search:
        dx = quadratic_offset(*score[y, x - 1:x + 2])
        dy = quadratic_offset(*score[y - 1:y + 2, x])
        offset = np.array([x - search + dx, y - search + dy])
    # Similar single dark spots may give several strong peaks. The second-best
    # peak outside the central lobe measures whether the match is distinctive.
    competitor = score.copy()
    competitor[max(0, y - 3):y + 4, max(0, x - 3):x + 4] = -1.
    return offset, strength, strength - competitor.max()


def read_reference(video, ref, centers, margin):
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, ref)
        ok, bgr = cap.read()
        if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != ref + 1:
            raise ValueError("Reference frame unavailable.")
    finally:
        cap.release()
    frame = channel(bgr)
    h, w = frame.shape
    if any(not (margin <= x < w - margin and margin <= y < h - margin) for x, y in centers):
        raise ValueError("A reference patch or its search window extends beyond the image.")
    return frame


def consensus(displacements, valid, tolerance):
    if valid.sum() < 4:
        return np.full(2, np.nan), np.zeros_like(valid)
    coarse = np.median(displacements[valid], axis=0)
    inlier = valid & (np.linalg.norm(displacements - coarse, axis=1) <= tolerance)
    estimate = np.median(displacements[inlier], axis=0) if inlier.sum() >= 4 else np.full(2, np.nan)
    return estimate, inlier


def estimate(args):
    centers = args.mark + args.holdout
    if len(set(centers)) != len(centers) or len(args.mark) < 4:
        raise ValueError("Use at least four distinct fitting marks and separate holdout marks.")
    n = len(args.mark)
    reference = read_reference(args.video, args.reference_frame, centers, args.radius + args.search)
    templates = [patch(reference, c, args.radius).copy() for c in centers]
    if any(p.std() < 0.5 for p in templates):
        raise ValueError("A reference patch contains insufficient contrast.")
    if any(args.radius * 2 >= np.linalg.norm(np.subtract(a, b), ord=np.inf)
           for i, a in enumerate(centers) for b in centers[:i]):
        raise ValueError("Use nonoverlapping fit and holdout patches.")
    # The parabolic peak for an identical patch can have a small asymmetry. Subtract
    # each mark's own reference-frame fit so a repeated frame measures zero.
    origin = np.array([locate(reference, c, t, args.search, args.radius)[0]
                       for c, t in zip(centers, templates)])
    rows = []
    detail = []
    for frame_id, frame in enumerate(frames_from(args.video)):
        gray = channel(frame)
        if gray.shape != reference.shape:
            raise ValueError("Frame size changed.")
        matched = [locate(gray, c, t, args.search, args.radius) for c, t in zip(centers, templates)]
        disp = np.array([pair[0] for pair in matched]) - origin
        scores = np.array([pair[1] for pair in matched])
        gaps = np.array([pair[2] for pair in matched])
        valid = (scores >= args.min_ncc) & (gaps >= args.peak_margin) & np.isfinite(disp).all(axis=1)
        # If a neighboring particle outscores a fixed mark, the reverse search
        # frequently maps the observed patch to a different feature at reference time.
        roundtrip = np.full(len(centers), np.nan)
        for i in np.flatnonzero(valid):
            destination = tuple(np.rint(np.add(centers[i], matched[i][0])).astype(int))
            observed = patch(gray, destination, args.radius)
            back, _, _ = locate(reference, centers[i], observed, args.search, args.radius)
            roundtrip[i] = np.linalg.norm(back)
        valid &= roundtrip < args.reverse_px
        shift, inlier = consensus(disp[:n], valid[:n], args.agreement_px)
        # Holdouts never contribute to the shift. Report every qualifying match,
        # including any outlier, so residuals can reveal position-dependent motion.
        held = valid[n:] & np.isfinite(shift).all()
        err = np.linalg.norm(disp[n:][held] - shift, axis=1)
        med_err = np.median(err) if len(err) else np.nan
        quality = "ok" if inlier.sum() >= 4 else "insufficient_fit"
        if quality == "ok" and (len(err) < 2 or med_err > args.agreement_px):
            quality = "unconfirmed"
        accepted = quality == "ok"
        rows.append(dict(frame=frame_id, shift_x_px=shift[0] if accepted else None,
                         shift_y_px=shift[1] if accepted else None, quality=quality,
                         inlier_marks=int(inlier.sum()), valid_holdouts=len(err),
                         holdout_median_error_px=med_err if len(err) else None))
        for i, (center, delta, score) in enumerate(zip(centers, disp, scores)):
            detail.append(dict(frame=frame_id, ref_x=center[0], ref_y=center[1],
                               role="fit" if i < n else "holdout", ncc=score, peak_margin=gaps[i],
                               roundtrip_px=roundtrip[i] if np.isfinite(roundtrip[i]) else None,
                               dx_px=delta[0] if np.isfinite(delta[0]) else None,
                               dy_px=delta[1] if np.isfinite(delta[1]) else None,
                               high_quality=int(valid[i]), used_for_fit=int(accepted and i < n and inlier[i])))
    return rows, detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--reference-frame", type=int, required=True)
    parser.add_argument("--mark", type=xy, nargs="+", required=True, help="training mark centers x,y")
    parser.add_argument("--holdout", type=xy, nargs="+", required=True, help="validation mark centers x,y")
    parser.add_argument("--radius", type=int, default=8, help="half width in pixels of the fixed template")
    parser.add_argument("--search", type=int, default=15, help="maximum search displacement in pixels")
    parser.add_argument("--agreement-px", type=float, default=1.5)
    parser.add_argument("--peak-margin", type=float, default=0.05)
    parser.add_argument("--reverse-px", type=float, default=1.5)
    parser.add_argument("--min-ncc", type=float, default=0.7)
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    if (args.reference_frame < 0 or args.radius < 2 or args.search < 2 or not (0 < args.min_ncc <= 1)
            or not np.isfinite(args.agreement_px) or args.agreement_px <= 0 or len(args.holdout) < 2
            or not 0 <= args.peak_margin < 2 or not np.isfinite(args.reverse_px) or args.reverse_px <= 0):
        parser.error("Check bounds and supply at least two held-out marks.")
    if args.out.exists():
        parser.error("Choose a new output directory.")
    try:
        rows, detail = estimate(args)
    except (ValueError, OSError, cv2.error) as exc:
        parser.exit(1, f"{exc}\n")
    args.out.mkdir(parents=True)
    for name, records in (("shift", rows), ("marks", detail)):
        with (args.out / f"{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    settings = vars(args).copy()
    settings.update(video=args.video.name, out=args.out.name,
                    valid_frames=sum(r["quality"] == "ok" for r in rows), total_frames=len(rows))
    (args.out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"{settings['valid_frames']}/{len(rows)} frames passed independent fixed-feature checks.")


if __name__ == "__main__":
    main()
