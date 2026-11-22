"""Render the labeled field beside matched observed and common-motion-corrected paths."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from review_frames import CJK, RGB


INK = (30, 46, 57)
MUTED = (73, 91, 101)
BG = (246, 248, 247)
ORIGINAL = (101, 75, 165)
RESIDUAL = (8, 125, 127)
LEFT = 948
RIGHT = LEFT + 40
WIDTH = RIGHT + 430
TRACK_LETTERS = "ABC"


def read_points(path):
    """Read only the three review classes. Track IDs match the speed tables."""
    points = defaultdict(list)
    last_frame = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            state = row["status"]
            if state not in RGB:
                continue
            n = int(row["frame"])
            tid = int(row["track_id"]) if state == "moving" else None
            points[n].append((tid, state, float(row["x_px"]), float(row["y_px"]), float(row["diameter_px"])))
            if tid is not None:
                last_frame[tid] = n
    return points, last_frame


def load_steps(path):
    with gzip.open(path, "rt", newline="") as f:
        steps = defaultdict(dict)
        for row in csv.DictReader(f):
            tid, frame = int(row["track_id"]), int(row["frame_to"])
            if int(row["frame_from"]) + 1 != frame:
                raise ValueError("Speed steps must connect adjacent frames")
            steps[tid][frame] = [float(row[k] or "nan") for k in
                                 ("dx_px", "dy_px", "dx_residual_px", "dy_residual_px")]
    return steps


def update_examples(previous, points, ends, n, w, h):
    # Keep an example until it leaves. When a slot is empty, prefer a long
    # track near its region of the field. This sample is for illustration.
    locations = {tid: (x, y) for tid, _, x, y, _ in points if tid is not None}
    chosen = [p if p in locations else None for p in previous]
    for i, target in enumerate(((.28, .28), (.64, .48), (.40, .76))):
        if chosen[i] is not None:
            continue
        candidates = (tid for tid in locations if tid not in chosen)
        def rank(tid):
            x, y = locations[tid]
            d = np.hypot(x / w - target[0], y / h - target[1])
            return min(40, ends[tid] - n) - 35 * d
        chosen[i] = max(candidates, key=rank, default=None)
    return chosen, locations


def display_field(bgr, points, chosen, locations, font):
    gray = bgr[:, :, 1].astype(np.int16)
    highpass = gray - cv2.blur(gray, (51, 51))
    shown = np.clip(highpass * 12 + 186, 0, 255).astype(np.uint8)
    h, w = gray.shape
    scale = LEFT / w
    image = cv2.resize(shown, (LEFT, round(h * scale)), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for _, state, x, y, diameter in points:
        center = (round(scale * x), round(scale * y))
        radius = max(3, round((diameter / 2 + 2) * scale))
        cx, cy = center
        color = RGB[state]
        if state == "aggregate":
            cv2.rectangle(image, (cx-radius, cy-radius), (cx+radius, cy+radius), color, 1)
        elif state == "stationary":
            corners = np.array([(cx, cy-radius), (cx+radius, cy), (cx, cy+radius), (cx-radius, cy)])
            cv2.polylines(image, [corners.astype(np.int32)], True, color, 1, cv2.LINE_AA)
        else:
            cv2.circle(image, center, radius, color, 1, cv2.LINE_AA)
    panel = Image.fromarray(image)
    draw = ImageDraw.Draw(panel)
    for i, tid in enumerate(chosen):
        if tid is None:
            continue
        xy = np.asarray(locations[tid]) * scale
        x = int(np.clip(xy[0] + 7, 0, panel.width - 25))
        y = int(np.clip(xy[1] - 25, 0, panel.height - 24))
        cx, cy = (int(v) for v in xy)
        draw.line([(cx+4, cy-5), (x, y+21)], fill=INK, width=2)
        draw.ellipse((cx-7, cy-7, cx+7, cy+7), outline=INK, width=1)
        draw.rectangle((x, y, x+22, y+23), fill=INK)
        draw.text((x+4, y-1), TRACK_LETTERS[i], fill="white", font=font)
    return panel


def example_paths(steps, n, window):
    frame = n
    sequence = []
    while len(sequence) < window and frame in steps:
        sequence.append(steps[frame])
        frame -= 1
    if not sequence:
        return np.zeros((1, 2)), None
    pos = np.vstack([np.zeros(4), np.cumsum(sequence[::-1], axis=0)])
    return pos[:, :2], pos[:, 2:] if np.isfinite(pos[:, 2:]).all() else None


def path_panel(draw, box, path, grid_px, color):
    x, y, w, h = box
    cx, cy = x + w // 2, y + h // 2
    pitch = 24
    draw.rectangle((x, y, x+w, y+h), fill=(255, 255, 255))
    for a in range(cx % pitch + (x // pitch) * pitch - pitch, x+w+1, pitch):
        if x <= a <= x+w:
            draw.line((a, y, a, y+h), fill=(225, 231, 230))
    for b in range(cy % pitch + (y // pitch) * pitch - pitch, y+h+1, pitch):
        if y <= b <= y+h:
            draw.line((x, b, x+w, b), fill=(225, 231, 230))
    if path is None:
        return
    xy = (path * (pitch / grid_px)) + [cx, cy]
    if len(xy) > 1:
        draw.line([tuple(p) for p in xy], fill=color, width=3)
    draw.ellipse((cx-4, cy-4, cx+4, cy+4), fill="white", outline=color, width=2)
    if len(xy) > 1:
        ex, ey = xy[-1]
        draw.ellipse((ex-3, ey-3, ex+3, ey+3), fill=color)


def render(args):
    summary = json.loads((args.results / "summary.json").read_text())
    fps = summary["fps"]
    if not fps or fps <= 0:
        raise ValueError("A known frame rate is required.")
    points, ends = read_points(args.results / "tracks.csv")
    steps = load_steps(args.steps or args.results / "correction/steps.csv.gz")
    if sum(map(len, steps.values())) != summary["accepted_steps"]:
        raise ValueError("Corrected steps differ from the count in the tracking run")
    path = next((p for p in CJK if Path(p).is_file()), None)
    fonts = {s: ImageFont.truetype(path or "DejaVuSans.ttf", s) for s in (16, 18, 21, 26, 30)}
    def tr(zh, en):
        return zh if path else en
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise ValueError("Cannot open the original video")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fh = round(h * LEFT / w)
    body = max(fh, 524)
    height = (102 + body + 86 + 1) // 2 * 2
    window = max(1, round(fps))
    chosen = [None] * 3
    output = args.out or args.results / "correction/annotated.mp4"
    output.parent.mkdir(exist_ok=True, parents=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, summary["first_frame"])
    try:
        with TemporaryDirectory(dir=output.parent) as temp:
            tmp = Path(temp) / "movie.mp4"
            cmd = ["ffmpeg", "-v", "error", "-nostdin", "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", f"{WIDTH}x{height}", "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                   "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "fast", "-movflags", "+faststart", str(tmp)]
            with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
                try:
                    for n in range(summary["first_frame"], summary["last_frame"] + 1):
                        ok, bgr = cap.read()
                        if not ok:
                            raise ValueError(f"Missing source frame {n}")
                        obs = points[n]
                        chosen, locations = update_examples(chosen, obs, ends, n, w, h)
                        field = display_field(bgr, obs, chosen, locations, fonts[16])
                        canvas = Image.new("RGB", (WIDTH, height), BG)
                        canvas.paste(field, (16, 91))
                        d = ImageDraw.Draw(canvas)
                        def label(x, y, text, size=18, color=INK):
                            d.text((x,y), text, font=fonts[size], fill=color)
                        label(16, 8, tr("微球运动与漂移校正", "Bead motion and drift correction"), 30)
                        label(560, 15, f"{n / fps:05.2f} s     #{n} / {summary['last_frame']}     {fps:g} fps", 21)
                        stats = Counter(row[1] for row in obs)
                        for j, (s, cn, en) in enumerate([("moving", "运动单球", "Moving"),
                                                          ("aggregate", "疑似团聚", "Clumps"),
                                                          ("stationary", "静止斑点", "Stationary")]):
                            x = 18 + j*235
                            d.rectangle((x, 64, x+10, 74), fill=RGB[s])
                            label(x+19, 54, f"{tr(cn,en)}   {stats[s]}")
                        label(RIGHT, 8, tr("单球轨迹 · 最近一秒", "Single tracks | last second"), 26)
                        label(RIGHT, 55, tr("原始轨迹", "Observed"), 21, ORIGINAL)
                        label(RIGHT+208, 55, tr("扣除共同运动", "Common motion removed"), 21, RESIDUAL)
                        card_h = body // 3
                        for i, tid in enumerate(chosen):
                            y = 92 + i*card_h
                            label(RIGHT+1, y, TRACK_LETTERS[i] + (f"     ID {tid}" if tid is not None else ""), 18)
                            observed, residual = example_paths(steps.get(tid, {}), n, window)
                            pw, ph = 196, card_h-51
                            traces = observed if residual is None else np.vstack([observed, residual])
                            needed = (np.abs(traces).max(axis=0) * 24 / np.array([pw/2-5, ph/2-5])).max()
                            grid_px = 2 if needed <= 2 else max(5, int(np.ceil(needed / 5)) * 5)
                            label(RIGHT+208, y, tr(f"每格 {grid_px} px", f"{grid_px} px/grid"), 16, MUTED)
                            for j, (values, color) in enumerate([(observed if tid is not None else None, ORIGINAL),
                                                                 (residual, RESIDUAL)]):
                                box = (RIGHT+208*j, y+31, pw, ph)
                                path_panel(d, box, values, grid_px, color)
                                if values is None:
                                    label(box[0]+12,box[1]+ph//2-15,tr("无有效步长", "No valid step"),16,MUTED)
                        label(18, body+102, tr("左图位置为相机坐标。A、B、C 对应右侧三个示例，编号变化表示已换球。",
                                             "Labels match the example tracks. A change of ID indicates a new segment."), 18)
                        label(18, body+141, tr("右图以各段起点为零，空心圆为起点，实心点为当前点。两列同步，方向与相机一致。",
                                             "Open circles mark segment starts. Solid dots mark the current position. Axes match the field."), 18)
                        proc.stdin.write(canvas.tobytes())
                finally:
                    proc.stdin.close()
                if proc.wait():
                    raise RuntimeError("ffmpeg failed to encode the annotation")
            tmp.replace(output)
    finally:
        cap.release()
    print(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", type=Path)
    p.add_argument("results", type=Path)
    p.add_argument("--steps", type=Path, help="gzip CSV with corrected displacements")
    p.add_argument("--out", type=Path)
    render(p.parse_args())
