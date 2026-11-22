"""Plot matched finite-frame speeds and 2D mean squared displacements."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
import numpy as np


COLORS = ("#405C79", "#B66014", "#007968")
INK = "#182939"
MUTED = "#546675"


def configure_fonts():
    candidates = (
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    )
    for path in map(Path, candidates):
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            family = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.family"] = [family, "DejaVu Sans"]
            return True
    return False


def numeric_columns(path, names):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = set(names) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} is missing {sorted(missing)}")
        data = [[float(row[name]) if row[name] else np.nan for name in names] for row in reader]
    return np.asarray(data, dtype=float).reshape(-1, len(names))


def chart_style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DCE4E8", linewidth=0.65)
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#AEBCC3")
    ax.tick_params(color="#AEBCC3", labelcolor=MUTED, labelsize=10)


def heading(fig, title, subtitle):
    fig.text(0.075, 0.955, title, va="top", color=INK, fontsize=21)
    fig.text(0.075, 0.86, subtitle, va="top", color=MUTED, fontsize=11)


def speeds_chart(out, zh):
    values = numeric_columns(out / "steps.csv.gz", ("speed_raw_px_s", "speed_ref_px_s", "speed_residual_px_s"))
    total = len(values)
    values = values[np.isfinite(values).all(axis=1)]
    if not len(values) or np.any(values < 0):
        raise ValueError("No matched speed data or negative speed.")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.2), sharey=True, sharex=True)
    fig.subplots_adjust(left=.075, right=.98, bottom=.17, top=.66, wspace=.13)
    names = (["画面原始", "扣除固定参照位移", "再扣除单球共同位移"] if zh else
             ["As recorded", "Fixed-reference corrected", "Common motion removed"])
    title = "运动单球的逐帧速率分布" if zh else "Frame-to-frame speeds of moving beads"
    subtitle = (f"三组使用相同的 {len(values):,} 个有效步长；各按本组步数归一化。竖线为中位数。"
                if zh else f"Matched sample of {len(values):,} steps. Each panel sums to 100%. Vertical lines mark medians.")
    heading(fig, title, subtitle)
    # Use common bins over the full range. Rounding the bin width to 0.1 px/s
    # improves readability and keeps the rightmost observation inside the plot.
    bin_width = max(.1, np.ceil(values.max() / 55 * 10) / 10)
    n_bins = max(1, int(np.ceil(values.max() / bin_width)))
    bins = np.arange(n_bins + 1) * bin_width
    for i, ax in enumerate(axes):
        chart_style(ax)
        sample = values[:, i]
        median = float(np.median(sample))
        counts = np.histogram(sample, bins=bins)[0]
        ax.stairs(counts * 100 / len(sample), bins, fill=True, color=COLORS[i], alpha=.21)
        ax.stairs(counts * 100 / len(sample), bins, linewidth=1.6, color=COLORS[i])
        ax.axvline(median, color=COLORS[i], linestyle=(0, (3, 2)), lw=1.6)
        ax.set_title(names[i], loc="left", color=COLORS[i], fontsize=13, pad=20)
        tag = "中位数" if zh else "median"
        ax.text(.97, .94, f"{tag}  {median:.2f} px/s", va="top", ha="right", color=COLORS[i],
                transform=ax.transAxes, fontsize=11)
        ax.set_xlim(0, bins[-1])
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_xlabel("表观速率 (px/s)" if zh else "Apparent speed (px/s)", labelpad=7)
    axes[0].set_ylabel("步数占比 (%)" if zh else "Steps per bin (%)", labelpad=8)
    note = (f"统一组距 {bin_width:g} px/s。原始记录共 {total:,} 个步长。"
            if zh else f"Common bin width {bin_width:g} px/s. Input table contains {total:,} steps.")
    fig.text(.075, .04, note, color=MUTED, fontsize=9)
    fig.savefig(out / "speed_distribution.png", dpi=155, facecolor="white")
    plt.close(fig)


def msd_chart(out, zh):
    table = numeric_columns(out / "msd.csv", ("lag_s", "pairs", "raw_px2", "ref_px2", "residual_px2"))
    table = table[np.isfinite(table).all(axis=1) & (table[:, 1] > 0)]
    if not len(table):
        raise ValueError("No matched MSD values.")
    lag, count, raw, ref, residual = table.T
    with (out / "summary.json").open(encoding="utf-8") as f:
        fit = json.load(f).get("msd_fit") or {}
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.4))
    fig.subplots_adjust(left=.09, right=.975, bottom=.19, top=.67, wspace=.28)
    title = "运动单球的二维均方位移" if zh else "Two-dimensional mean squared displacement"
    subtitle = ("同一滞后下三条曲线采用相同的位置对。两图的纵轴分别标度。" if zh else
                "Curves use matching position pairs at each lag. The panels use separate vertical scales.")
    heading(fig, title, subtitle)
    for ax in axes:
        chart_style(ax)
        ax.set_xlabel("时间间隔 (s)" if zh else "Lag time (s)", labelpad=7)
        ax.set_ylabel("均方位移 (px²)" if zh else "MSD (px²)", labelpad=6)
        ax.set_xlim(left=0, right=lag.max() * 1.03)
    left, right = axes
    left.set_title("画面位移与固定参照校正" if zh else "Image and reference-corrected motion",
                   loc="left", fontsize=13, color=INK, pad=9)
    left.plot(lag, raw, color=COLORS[0], linewidth=2.3, label="画面原始" if zh else "As recorded")
    left.plot(lag, ref, lw=0, marker="o", markerfacecolor="white", markeredgecolor=COLORS[1], ms=4.2,
              label="固定参照校正" if zh else "Fixed-reference corrected")
    left.legend(frameon=False, loc="upper left", fontsize=10)
    right.set_title("扣除共同位移后的剩余运动" if zh else "Motion after removing the common displacement",
                    loc="left", fontsize=13, color=INK, pad=9)
    right.plot(lag, residual, color=COLORS[2], linewidth=1.6, marker="o", ms=3,
               label="剩余运动" if zh else "Residual motion")
    if all(fit.get(key) is not None for key in ("lag_min_s", "lag_max_s", "slope_px2_s", "intercept_px2")):
        times = np.array([fit["lag_min_s"], fit["lag_max_s"]])
        right.plot(times, fit["slope_px2_s"] * times + fit["intercept_px2"], "--", color="#714E90",
                   lw=2, label="短时线性拟合" if zh else "Short-lag linear fit")
    right.legend(loc="upper left", frameon=False, fontsize=10)
    for ax in axes:
        ax.set_ylim(bottom=0)
    note = (f"每个时间间隔的有效位置对为 {int(count.min()):,}–{int(count.max()):,} 对。"
            if zh else f"Matching position pairs per lag range from {int(count.min()):,} to {int(count.max()):,}.")
    fig.text(.075, .04, note, color=MUTED, fontsize=9)
    fig.savefig(out / "msd.png", dpi=155, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="correction directory with steps.csv.gz, msd.csv and summary.json")
    output = parser.parse_args().directory
    chinese = configure_fonts()
    plt.rcParams.update({"text.color": INK, "axes.labelcolor": INK, "axes.unicode_minus": False})
    speeds_chart(output, chinese)
    msd_chart(output, chinese)
