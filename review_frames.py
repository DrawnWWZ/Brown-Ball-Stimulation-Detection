"""Export raw frames, full-field labels and matched close-ups from tracking results.

Usage: python review_frames.py VIDEO RESULTS --frames 100 107 114
This exporter uses Pillow. Chinese labels require a supported CJK font. The displayed green
channel uses a common local background subtraction and contrast scale. Raw PNGs
contain the original decoded 8-bit color frames.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


STATES = ("moving", "aggregate", "stationary")
RGB = dict(moving=(0, 219, 127), aggregate=(255, 115, 69), stationary=(71, 200, 255))
INK = (34, 49, 62)
PAPER = (248, 249, 247)
MUTED = (79, 95, 107)
CJK = [
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/msyh.ttc",
]


def label_fonts():
    font_path = next((p for p in CJK if Path(p).is_file()), None)
    if font_path:
        return True, {s: ImageFont.truetype(font_path, s) for s in (15, 17, 20, 25, 31)}
    def english_font(size):
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()
    return False, {s: english_font(s) for s in (15, 17, 20, 25, 31)}


def read_tracks(path, wanted):
    frames = defaultdict(list)
    for_frame = set(wanted)
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            i = int(r["frame"])
            if i in for_frame and r["status"] in STATES:
                frames[i].append(dict(state=r["status"], xy=np.array([float(r["x_px"]), float(r["y_px"])]),
                                      diameter=float(r["diameter_px"]), tid=r["track_id"]))
    return [frames[n] for n in wanted]


def fetch_frames(path, source_indices):
    cap = cv2.VideoCapture(str(path))
    try:
        frames = []
        for n in source_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, n)
            ok, bgr = cap.read()
            if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != n + 1:
                raise ValueError(f"Cannot decode source frame {n}")
            frames.append(bgr)
        return frames, cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()


def enhance(frames, crop):
    # Keep the scale fixed across all times and both panels.
    values = []
    for bgr in frames:
        gray = bgr[:, :, 1].astype(np.float32)
        values.append(gray - cv2.GaussianBlur(gray, (0, 0), max(10., crop / 4.)))
    sample = np.concatenate([v[::3, ::3].ravel() for v in values])
    low, high = np.percentile(sample, [0.2, 99.8])
    return [Image.fromarray(((v - low) * 255 / max(high - low, 1)).clip(0, 255).astype(np.uint8)).convert("RGB")
            for v in values]


def symbol(draw, state, center, radius, width):
    x, y = center
    box = (x - radius, y - radius, x + radius, y + radius)
    color = RGB[state]
    if state == "moving":
        draw.ellipse(box, outline=color, width=width)
    elif state == "aggregate":
        draw.rectangle(box, outline=color, width=width)
    else:
        draw.line([(x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y), (x, y - radius)],
                  fill=color, width=width)


def pick_center(frames, state, specified, crop, size):
    if specified is not None:
        return np.array(specified, dtype=float)
    # Compare the same field at each time. Prefer a feature contained throughout;
    # for trackable objects reward observing the same track ID across the panel.
    middle = len(frames) // 2
    exemplar_frame = min((i for i, rows in enumerate(frames) if any(r["state"] == state for r in rows)),
                         key=lambda i: abs(i - middle), default=None)
    if exemplar_frame is None:
        return None
    def score(ref):
        xy = ref["xy"]
        matching = neighbors = 0
        for rows in frames:
            local = [r for r in rows if r["state"] == state and
                     np.max(np.abs(r["xy"] - xy)) <= crop * .26]
            matching += bool(local)
            neighbors += int(bool(ref["tid"]) and any(r["tid"] == ref["tid"] for r in local))
        margin = np.minimum(xy, np.array(size) - 1 - xy).min()
        isolation = min((np.linalg.norm(r["xy"] - xy) for r in frames[exemplar_frame] if r is not ref),
                        default=crop)
        return matching, neighbors, margin >= crop / 2, min(isolation, crop), ref["diameter"]
    references = [r for r in frames[exemplar_frame] if r["state"] == state]
    return max(references, key=score)["xy"]


def crop_box(xy, size, crop):
    center = np.rint(xy).astype(int)
    left, top = np.clip(center - crop // 2, 0, np.array(size) - crop)
    return int(left), int(top), int(left + crop), int(top + crop)


def tile(im, rows, box, side, trajectories):
    scale = side / (box[2] - box[0])
    raw = im.crop(box).resize((side, side), Image.Resampling.BILINEAR)
    marked = raw.copy()
    d = ImageDraw.Draw(marked)
    def position(xy):
        return (xy - np.array(box[:2]) + 0.5) * scale - 0.5
    for r in rows:
        if r["state"] == "moving" and len(trajectories.get(r["tid"], [])) > 1:
            path = [tuple(position(p)) for p in trajectories[r["tid"]]]
            d.line(path, fill=RGB["moving"], width=2)
    for r in rows:
        xy = position(r["xy"])
        radius = max(6., (r["diameter"] / 2 + 2.) * scale)
        if all(xy > -radius) and all(xy < side + radius):
            symbol(d, r["state"], xy, radius, 2)
    return raw, marked


def draw_header(draw, fonts, title, subtitle):
    draw.text((22, 11), title, fill=INK, font=fonts[31])
    draw.text((23, 57), subtitle, fill=MUTED, font=fonts[17])


def whole(im, rows, source, fps, out, zh, fonts):
    n = Counter(r["state"] for r in rows)
    w, h = im.size
    canvas = Image.new("RGB", (max(w, 690), h + 91), PAPER)
    canvas.paste(im, (0, 91))
    d = ImageDraw.Draw(canvas)
    d.text((16, 7), ("全视野" if zh else "Full field") + f"   ·   #{source}   {source / fps:.2f} s",
           fill=INK, font=fonts[25])
    names = (["运动单球", "疑似团聚", "静止斑点"] if zh else ["Moving", "Possible clump", "Stationary"])
    for index, (state, name) in enumerate(zip(STATES, names)):
        x = 16 + 207 * index
        symbol(d, state, (x + 8, 64), 6, 3)
        d.text((x + 23, 50), f"{name}  {n[state]}", fill=INK, font=fonts[20])
    for r in rows:
        xy = r["xy"]
        symbol(d, r["state"], (xy[0], xy[1] + 91), max(5, r["diameter"] / 2 + 2), 2)
    canvas.save(out / f"frame_{source:04d}_marked.png")


def make_review(args):
    if len(set(args.frames)) != len(args.frames) or args.frames != sorted(args.frames):
        raise ValueError("Pass distinct frame indices in increasing order.")
    source = [n + args.offset for n in args.frames]
    if min(source) < 0 or min(args.frames) < 0:
        raise ValueError("Source frames and result frames must be nonnegative.")
    rows = read_tracks(args.results / "tracks.csv", args.frames)
    frames, header_fps = fetch_frames(args.video, source)
    with (args.results / "summary.json").open(encoding="utf-8") as f:
        summary = json.load(f)
    first = summary.get("first_frame", 0)
    last = summary.get("last_frame", first + summary["frames"] - 1)
    if args.frames[0] < first or args.frames[-1] > last:
        raise ValueError(f"Requested frames must lie within the tracked interval {first} to {last}.")
    first = summary.get("first_frame", 0)
    last = summary.get("last_frame", first + summary["frames"] - 1)
    if min(args.frames) < first or max(args.frames) > last:
        raise ValueError(f"Select frames within the analyzed interval [{first}, {last}].")
    fps = summary.get("fps") or header_fps
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("An acquisition or video frame rate is required for time labels.")
    size = frames[0].shape[1::-1]
    if args.crop < 12 or args.crop > min(size):
        raise ValueError("Crop width must be between 12 and the smaller frame dimension.")
    images = enhance(frames, args.crop)
    out = args.out or args.results / "review"
    out.mkdir(parents=True, exist_ok=True)
    zh, fonts = label_fonts()
    for bgr, im, obs, src in zip(frames, images, rows, source):
        if not cv2.imwrite(str(out / f"frame_{src:04d}_raw.png"), bgr):
            raise OSError("Cannot write raw frame")
        whole(im, obs, src, fps, out, zh, fonts)

    side, gap, left, top = 144, 5, 164, 136
    width = side * 2 + gap
    height = side + 26
    overview = Image.new("RGB", (left + len(source) * (width + 18) + 4, top + 3 * height + 47), PAPER)
    d = ImageDraw.Draw(overview)
    title = "微球识别 · 连续帧局部复核" if zh else "Microsphere tracking | detail review"
    sub = ("每个格子左为未标注图，右为同图标注。各时间点使用同一视野范围和增强尺度。"
           if zh else "Unmarked and marked views share fixed crops and a common contrast scale.")
    draw_header(d, fonts, title, sub)
    for j, src in enumerate(source):
        x = left + j * (width + 18)
        d.text((x, 92), f"#{src}     {src / fps:.2f} s", font=fonts[25], fill=INK)
    # Draw the measured path between selected times for the same track when it
    # remains visible. The first point in each row lies on the preceding frame.
    by_frame = {}
    with (args.results / "tracks.csv").open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            i = int(r["frame"])
            if r["status"] == "moving" and args.frames[0] <= i <= args.frames[-1]:
                by_frame.setdefault(i, {})[r["track_id"]] = (float(r["x_px"]), float(r["y_px"]))
    names = ["运动单球", "疑似团聚", "静止斑点"] if zh else ["Moving", "Possible clump", "Stationary"]
    for i, (state, name) in enumerate(zip(STATES, names)):
        y = top + i * height
        center = pick_center(rows, state, getattr(args, state), args.crop, size)
        symbol(d, state, (27, y + 20), 8, 3)
        d.text((43, y + 4), name, font=fonts[20], fill=INK)
        if center is None:
            d.text((left, y + 40), "选取的帧中未见该类标记" if zh else "No label in the selected frames",
                   font=fonts[17], fill=MUTED)
            continue
        box = crop_box(center, size, args.crop)
        d.text((21, y + 54), f"x {center[0]:.0f}, y {center[1]:.0f}", font=fonts[15], fill=MUTED)
        d.text((21, y + 80), f"{args.crop} × {args.crop} px", font=fonts[15], fill=MUTED)
        for j, (im, obs, frame) in enumerate(zip(images, rows, args.frames)):
            local = [r for r in obs if box[0] - 6 <= r["xy"][0] <= box[2] + 6 and
                     box[1] - 6 <= r["xy"][1] <= box[3] + 6]
            start = args.frames[j - 1] if j else frame
            trails = {r["tid"]: [by_frame[k][r["tid"]] for k in range(start, frame + 1)
                                  if r["tid"] in by_frame.get(k, {})]
                      for r in local if r["state"] == "moving"}
            a, b = tile(im, local, box, side, trails)
            x = left + j * (width + 18)
            overview.paste(a, (x, y))
            overview.paste(b, (x + side + gap, y))
    footer = ("绿线为同一轨迹在相邻展示时刻之间的实测路径。坐标和裁剪尺寸均为原视频像素，帧号从 0 起。"
              if zh else "Green paths link successive measured positions. All crop coordinates use original pixels. Frame IDs start at zero.")
    d.text((22, overview.height - 38), footer, fill=MUTED, font=fonts[15])
    overview.save(out / "review.png")
    print(out / "review.png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", type=Path)
    p.add_argument("results", type=Path)
    p.add_argument("--frames", nargs="+", type=int, required=True)
    p.add_argument("--offset", type=int, default=0, help="source frame minus the frame ID in tracks.csv")
    p.add_argument("--crop", type=int, default=64, help="width of each fixed crop in original pixels")
    for state in STATES:
        p.add_argument("--" + state, nargs=2, type=float, metavar=("X", "Y"), help="fixed crop center in raw pixels")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    try:
        make_review(args)
    except (ValueError, OSError, cv2.error) as err:
        p.exit(1, f"{err}\n")


if __name__ == "__main__":
    main()
