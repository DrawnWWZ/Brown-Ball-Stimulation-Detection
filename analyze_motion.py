"""Compare camera-frame steps, fixed-feature translation correction, and collective drift removal."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


def corrected_steps(results, reference_file):
    cfg = json.loads((results / "summary.json").read_text())
    fps = cfg["fps"]
    if cfg["stabilized"] or not fps or fps <= 0:
        raise ValueError("Supply tracks in camera coordinates with a known frame rate.")
    steps = pd.read_csv(results / "steps.csv").sort_values(["track_id", "frame_from"]).reset_index(drop=True)
    if steps.empty or not steps.frame_to.eq(steps.frame_from + 1).all():
        raise ValueError("Moving single-bead steps must link adjacent frames.")
    refs = pd.read_csv(reference_file).set_index("frame")
    frames = np.arange(cfg["first_frame"], cfg["last_frame"] + 1)
    if refs.index.has_duplicates or not pd.Index(frames).isin(refs.index).all():
        raise ValueError("Provide a reference CSV covering each source frame, using NaN for missing shifts.")
    keys = ["dx_px", "dy_px"]
    drift = pd.DataFrame(index=pd.Index(frames[1:], name="frame_to"))
    # Stable split by track ID. A particle's own step cannot reduce its residual.
    ids = np.sort(steps.track_id.unique())
    fold = pd.Series(np.arange(len(ids)) % 2, index=np.random.default_rng(1976).permutation(ids))
    steps["fold"] = steps.track_id.map(fold)
    by_frame = steps.groupby("frame_to")
    drift["n_moving"] = by_frame.size().reindex(drift.index, fill_value=0)
    centers = by_frame[keys].median()
    supports = {}
    for k in (0, 1):
        group = steps.loc[steps.fold != k].groupby("frame_to")
        count = group.size()
        supports[k] = group[keys].median().where(count >= 8, axis="index")
        drift[f"n_ref_for_fold{k}"] = count.reindex(drift.index, fill_value=0)
    for axis in "xy":
        key = f"d{axis}_px"
        shifts = refs[f"shift_{axis}_px"].reindex(frames).diff()
        drift[f"ref_{key}"] = shifts
        drift[f"common_{key}"] = centers[key].where(drift.n_moving >= 16)
        steps[f"{key[:2]}_ref_px"] = steps[key] - steps.frame_to.map(shifts)
        applied = np.full(len(steps), np.nan)
        for k in (0, 1):
            mask = steps.fold == k
            applied[mask] = steps.frame_to[mask].map(supports[k][key])
        # Subtracting a common reference displacement from all vectors, then their
        # median, equals subtracting the camera-frame ensemble median directly.
        steps[f"{key[:2]}_residual_px"] = steps[key] - applied
    for kind, suffix in (("raw", ""), ("ref", "_ref"), ("residual", "_residual")):
        steps[f"speed_{kind}_px_s"] = fps * np.hypot(steps[f"dx{suffix}_px"], steps[f"dy{suffix}_px"])
    return cfg, steps, drift.reset_index()


def msd_table(steps, fps, max_lag):
    cols = ["dx_px", "dy_px", "dx_ref_px", "dy_ref_px", "dx_residual_px", "dy_residual_px"]
    valid = steps[cols].notna().all(axis=1)
    # Never span missing intervals, track gaps, or an unsupported correction.
    s = steps.loc[valid].sort_values(["track_id", "frame_from"])
    boundary = s.track_id.ne(s.track_id.shift()) | s.frame_from.ne(s.frame_to.shift())
    arr = s[cols].to_numpy()
    sums = np.zeros((max_lag, 6), dtype=float)
    counts = np.zeros(max_lag, dtype=int)
    for g in np.split(arr, np.flatnonzero(boundary.to_numpy())[1:]):
        if not len(g):
            continue
        pos = np.vstack([np.zeros(6), g.cumsum(axis=0)])
        for lag in range(1, min(max_lag, len(g)) + 1):
            disp = pos[lag:] - pos[:-lag]
            sums[lag-1] += np.einsum("ij,ij->j", disp, disp)
            counts[lag-1] += len(disp)
    keep = counts > 0
    vals = sums[keep] / counts[keep, None]
    lag = np.flatnonzero(keep) + 1
    return pd.DataFrame(dict(lag_frames=lag, lag_s=lag/fps, pairs=counts[keep],
                             raw_px2=vals[:, :2].sum(axis=1), ref_px2=vals[:, 2:4].sum(axis=1),
                             residual_px2=vals[:, 4:].sum(axis=1), residual_x_px2=vals[:, 4],
                             residual_y_px2=vals[:, 5]))


def fit_msd(table, first=1, last=7):
    sub = table.query("@first <= lag_frames <= @last and residual_px2 > 0")
    if len(sub) != last - first + 1 or len(sub) < 3:
        return None
    tau, msd = sub.lag_s.to_numpy(), sub.residual_px2.to_numpy()
    slope, intercept = np.polyfit(tau, msd, 1)
    alpha = np.polyfit(np.log(tau), np.log(msd), 1)[0]
    return dict(slope_px2_s=slope, intercept_px2=intercept, d_px2_s=slope/4, alpha=alpha,
                lag_min_s=tau[0], lag_max_s=tau[-1], first_lag=first, last_lag=last)


def run(results, ref, out):
    cfg, steps, drift = corrected_steps(results, ref)
    fps = cfg["fps"]
    msd = msd_table(steps, fps, 21)
    kinds = ("raw", "ref", "residual")
    speed_cols = [f"speed_{k}_px_s" for k in kinds]
    matched = steps.dropna(subset=speed_cols)
    if matched.empty:
        raise ValueError("No jointly supported moving-bead steps for the comparison.")
    moving = matched.groupby("track_id")[speed_cols].mean().add_prefix("mean_")
    moving["n_matched_steps"] = matched.groupby("track_id").size()
    quantiles = {k: dict(zip(["p25", "median", "p75"], matched[col].quantile([.25,.5,.75])))
                 for k, col in zip(kinds, speed_cols)}
    summary = dict(fps=fps, frames=cfg["frames"], segments=int(steps.track_id.nunique()),
                   moving_steps=len(steps), matched_steps=len(matched), speed_px_s=quantiles,
                   residual_steps=int(steps.speed_residual_px_s.notna().sum()),
                   matched_segments=int(matched.track_id.nunique()), matched_intervals=int(matched.frame_to.nunique()),
                   msd_fit=fit_msd(msd), msd_fit_late=fit_msd(msd,8,14))
    if out.exists():
        raise ValueError("Choose a new output directory.")
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as tmp:
        dest = Path(tmp) / "analysis"
        dest.mkdir()
        steps.to_csv(dest / "steps.csv.gz", index=False, float_format="%.8g")
        drift.to_csv(dest / "drift.csv", index=False, float_format="%.8g")
        moving.to_csv(dest / "moving_segments.csv", float_format="%.8g")
        msd.to_csv(dest / "msd.csv", index=False, float_format="%.8g")
        (dest / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        dest.rename(out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("reference_csv", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    run(args.results, args.reference_csv, args.out or args.results / "correction")
