#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

import cv2
import numpy as np

# -------------------------------------------------
# GLOBAL CACHES FOR SPEED
# -------------------------------------------------
FPS_CACHE = {}
GAZE_CACHE = {}


# -------------------------------------------------
# JSON / TIME HELPERS
# -------------------------------------------------
def load_questions(json_path: Path):
    with json_path.open("r") as f:
        return json.load(f)


def _hms_to_timedelta(hms: str) -> timedelta:
    """Convert 'HH:MM:SS(.mmm)' string to timedelta."""
    fmt = "%H:%M:%S.%f" if "." in hms else "%H:%M:%S"
    t = datetime.strptime(hms, fmt)
    return timedelta(
        hours=t.hour,
        minutes=t.minute,
        seconds=t.second,
        microseconds=t.microsecond,
    )


def _timedelta_to_seconds(td: timedelta) -> float:
    return td.total_seconds()


# -------------------------------------------------
# VIDEO / METADATA HELPERS
# -------------------------------------------------
def get_video_path_from_qdata(hd_epic_root: Path, q_entry: dict) -> Path:
    """
    hd_epic_root / <participant> / <video_id>.mp4
    where video_id is something like "P01-20240202-110250".
    """
    inputs = q_entry["q_data"]["inputs"]
    vid_meta = inputs["video 1"]
    video_id = vid_meta["id"]  # e.g., "P01-20240202-110250"
    participant = video_id.split("-")[0]
    video_path = hd_epic_root / participant / f"{video_id}.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found at {video_path}")
    return video_path


def get_clip_window_from_inputs(q_entry: dict):
    """Return (start_time, end_time) as 'HH:MM:SS(.mmm)' strings."""
    inputs = q_entry["q_data"]["inputs"]
    vid_meta = inputs["video 1"]
    start_time = vid_meta["start_time"]
    end_time = vid_meta["end_time"]
    if not start_time or not end_time:
        raise ValueError(f"Missing start_time or end_time in inputs: {vid_meta}")
    return start_time, end_time


def get_video_fps(video_path: Path, video_id: str) -> float:
    """
    Fast FPS getter with caching.
    Uses OpenCV VideoCapture instead of ffprobe.
    """
    if video_id in FPS_CACHE:
        return FPS_CACHE[video_id]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for FPS: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps <= 0:
        raise RuntimeError(f"Invalid FPS ({fps}) for video: {video_path}")

    FPS_CACHE[video_id] = fps
    return fps


# -------------------------------------------------
# GAZE HELPERS (CSV: frame_idx, vrs_device_time_ns, gaze_x, gaze_y)
# -------------------------------------------------
def load_gaze_for_video(gaze_root: Path, video_id: str) -> np.ndarray:
    """
    Load gaze from CSV:
      <video_id>_gaze_xy_in_mp4.csv

    Expected columns:
      frame_idx, vrs_device_time_ns, gaze_x, gaze_y

    Returns numpy array of shape (N, 3):
      [frame_idx, gaze_x, gaze_y]
    cached per video for speed.
    """
    if video_id in GAZE_CACHE:
        return GAZE_CACHE[video_id]

    gaze_path = gaze_root / f"{video_id}_gaze_xy_in_mp4.csv"
    if not gaze_path.is_file():
        raise FileNotFoundError(f"Gaze file not found: {gaze_path}")

    # Load columns: frame_idx (0), gaze_x (2), gaze_y (3)
    data = np.loadtxt(
        str(gaze_path),
        delimiter=",",
        skiprows=1,   # skip header
        usecols=(0, 2, 3),
    )

    # If single line, ensure shape is (1, 3)
    if data.ndim == 1:
        data = data[None, :]

    # Sort by frame_idx just in case
    order = np.argsort(data[:, 0])
    data = data[order]

    GAZE_CACHE[video_id] = data
    return data


def compute_original_frame_indices(
    num_frames: int,
    start_time: str,
    end_time: str,
    fps: float,
):
    """
    For each extracted frame i in [0, num_frames-1], compute the
    corresponding original-frame index.

    We assume extracted frames are uniformly spaced across [start, end],
    and frame i is at:
        t_i = start + (i + 0.5) * (duration / num_frames)
    Then original index = round(t_i * fps).
    """
    start_td = _hms_to_timedelta(start_time)
    end_td = _hms_to_timedelta(end_time)
    start_sec = _timedelta_to_seconds(start_td)
    end_sec = _timedelta_to_seconds(end_td)

    duration = end_sec - start_sec
    if duration <= 0:
        raise ValueError(f"Non-positive clip duration: {start_time} -> {end_time}")

    indices = []
    for i in range(num_frames):
        t_rel = (i + 0.5) * (duration / num_frames)
        t_abs = start_sec + t_rel
        frame_idx = int(round(t_abs * fps))
        indices.append(frame_idx)
    return indices


def get_gaze_for_frame_index(gaze_arr: np.ndarray, frame_idx: int):
    """
    Return (gaze_x, gaze_y) in *pixel coordinates* for the given
    original frame index.

    gaze_arr: shape (N, 3) = [frame_idx, gaze_x, gaze_y]
    We find the row whose frame_idx is closest to the requested frame_idx.
    """
    if gaze_arr.size == 0:
        raise ValueError("Gaze array is empty.")

    frame_indices = gaze_arr[:, 0].astype(int)

    # Find insertion position
    pos = np.searchsorted(frame_indices, frame_idx)

    if pos == 0:
        chosen_idx = 0
    elif pos >= len(frame_indices):
        chosen_idx = len(frame_indices) - 1
    else:
        # Check neighbors pos-1 and pos, pick whichever is closer
        before = frame_indices[pos - 1]
        after = frame_indices[pos]
        if abs(before - frame_idx) <= abs(after - frame_idx):
            chosen_idx = pos - 1
        else:
            chosen_idx = pos

    _, gaze_x, gaze_y = gaze_arr[chosen_idx]
    return float(gaze_x), float(gaze_y)


# -------------------------------------------------
# IMAGE HELPERS (PIXEL-SPACE GAZE)
# -------------------------------------------------
def crop_around_gaze_pixels(
    img: np.ndarray,
    gaze_x: float,
    gaze_y: float,
    crop_size: int = 256,
) -> np.ndarray:
    """
    Crop a square of size crop_size x crop_size around gaze location
    specified in pixel coordinates (gaze_x, gaze_y).

    The gaze coords are assumed to be in the same pixel space as the
    original mp4, and we assume frames were extracted at that same size.
    """
    h, w = img.shape[:2]
    cx = int(round(gaze_x))
    cy = int(round(gaze_y))
    half = crop_size // 2

    x1 = cx - half
    y1 = cy - half
    x2 = cx + half
    y2 = cy + half

    # Clamp window to image boundaries, shifting if necessary
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > w:
        shift = x2 - w
        x1 = max(0, x1 - shift)
        x2 = w
    if y2 > h:
        shift = y2 - h
        y1 = max(0, y1 - shift)
        y2 = h

    crop = img[y1:y2, x1:x2]

    # NOTE: we no longer force this to be exactly crop_size here.
    # It will be resized later to the desired output size.
    return crop


def resize_to_square(img: np.ndarray, size: int = 256) -> np.ndarray:
    """Resize image to size x size."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


# -------------------------------------------------
# WRITING VIDEO WITH FFMPEG (MATCHES YOUR ORIGINAL STYLE)
# -------------------------------------------------
def save_interleaved_frames_with_ffmpeg(
    frames: list[np.ndarray],
    base_name: str,
    out_dir: Path,
    fps_out: float = 2.0,
) -> Path:
    """
    Save a list of frames to disk as JPEGs and repack into an mp4 using ffmpeg.

    - frames: list of HxWx3 uint8 BGR images (already interleaved orig/crop)
    - base_name: base for filenames, e.g. safe_q_key + "_gaze"
    - out_dir: directory where the final mp4 will live

    Returns: path to the output mp4.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # A subfolder to hold temporary JPEGs
    frames_dir = out_dir / f"{base_name}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    pattern_name = f"{base_name}_%02d.jpg"
    pattern_path = frames_dir / pattern_name

    # Write frames as JPEGs: base_name_00.jpg, base_name_01.jpg, ...
    for idx, frame in enumerate(frames):
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        img_path = frames_dir / f"{base_name}_{idx:02d}.jpg"
        cv2.imwrite(str(img_path), frame)

    # Final mp4 path
    out_video_path = out_dir / f"{base_name}.mp4"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps_out),
        "-i",
        str(pattern_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-y",
        str(out_video_path),
    ]
    subprocess.run(cmd, check=True)

    return out_video_path


# -------------------------------------------------
# PER-QUESTION PROCESSING
# -------------------------------------------------
def process_question(
    q_entry: dict,
    hd_epic_root: Path,
    frames_root: Path,
    gaze_root: Path,
    out_dir: Path,
    frame_size: int = 256,
    fps_out: float = 2.0,
    num_frames: int = 4,
    crop_window_size: int = 256,
) -> bool:
    """
    For a single question entry:
      - Map q_entry to its video and clip window.
      - Compute original frame indices for the extracted frames.
      - Load gaze CSV and frames.
      - Generate interleaved frames [orig_i, crop_i] in memory.
      - Save as mp4 with 256x256 frames via ffmpeg.

    Returns True on success, False on failure/skip.
    """
    src_file = q_entry.get("src_file", "")
    q_key = q_entry.get("q_key", "")
    safe_q_key = q_key.replace("/", "_")

    try:
        video_path = get_video_path_from_qdata(hd_epic_root, q_entry)
        video_id = q_entry["q_data"]["inputs"]["video 1"]["id"]
        start_time, end_time = get_clip_window_from_inputs(q_entry)

        # Where the original frames live (from your first script)
        frame_dir = frames_root / safe_q_key
        if not frame_dir.is_dir():
            print(f"  [SKIP] Frame dir not found for q_key={q_key}: {frame_dir}")
            return False

        # Collect frames: safe_q_key_XX.jpg
        frame_paths = sorted(frame_dir.glob(f"{safe_q_key}_*.jpg"))
        if len(frame_paths) == 0:
            print(f"  [SKIP] No frames found in {frame_dir}")
            return False

        if len(frame_paths) < num_frames:
            print(
                f"  [WARN] Expected at least {num_frames} frames, found {len(frame_paths)}. "
                f"Using {len(frame_paths)}."
            )
            num_frames_effective = len(frame_paths)
        else:
            num_frames_effective = num_frames
            frame_paths = frame_paths[:num_frames_effective]

        # FPS and gaze (cached)
        fps = get_video_fps(video_path, video_id)
        gaze_arr = load_gaze_for_video(gaze_root, video_id)

        # Map extracted frames -> original frame indices
        frame_indices = compute_original_frame_indices(
            num_frames=num_frames_effective,
            start_time=start_time,
            end_time=end_time,
            fps=fps,
        )

        print(f"  Video: {video_path}")
        print(f"  Frames dir: {frame_dir}")
        print(f"  num_frames_effective: {num_frames_effective}")

        interleaved_frames: list[np.ndarray] = []

        for i, frame_path in enumerate(frame_paths):
            img = cv2.imread(str(frame_path))
            if img is None:
                print(f"    [WARN] Could not read frame: {frame_path}, skipping.")
                continue

            # Original resized to frame_size x frame_size
            orig_resized = resize_to_square(img, frame_size)

            # Gaze lookup (pixel coords)
            frame_idx = frame_indices[i]
            gaze_x, gaze_y = get_gaze_for_frame_index(gaze_arr, frame_idx)

            # (1) Crop a 512x512 (or whatever) window around gaze in original pixel space
            gaze_crop_large = crop_around_gaze_pixels(
                img,
                gaze_x,
                gaze_y,
                crop_size=crop_window_size,
            )

            # (2) Downsample that crop to frame_size x frame_size (256x256)
            gaze_crop = resize_to_square(gaze_crop_large, frame_size)

            # Interleave: original 256x256, then gaze crop 256x256
            interleaved_frames.append(orig_resized)
            interleaved_frames.append(gaze_crop)

        if not interleaved_frames:
            print(f"  [SKIP] No valid frames for q_key={q_key}")
            return False

        # Pack into mp4 via ffmpeg (same style as your existing script)
        base_name = f"{safe_q_key}_gaze"
        out_video_path = save_interleaved_frames_with_ffmpeg(
            frames=interleaved_frames,
            base_name=base_name,
            out_dir=out_dir,
            fps_out=fps_out,
        )

        print(f"  [OK] Saved gaze-augmented video: {out_video_path}")
        return True

    except Exception as e:
        print(f"  [FAIL] src_file={src_file}, q_key={q_key}: {e}")
        return False


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Create gaze-augmented videos from extracted frames and gaze CSV data."
    )
    parser.add_argument(
        "--agg_json",
        type=str,
        required=True,
        help="Path to aggregated questions JSON.",
    )
    parser.add_argument(
        "--hd_epic_root",
        type=str,
        required=True,
        help="Path to HD-EPIC root (for locating original full videos).",
    )
    parser.add_argument(
        "--frames_root",
        type=str,
        required=True,
        help="Root dir where extracted frames are stored "
             "(same as --out_dir from your trimming script).",
    )
    parser.add_argument(
        "--gaze_root",
        type=str,
        required=True,
        help="Dir containing per-video gaze CSVs (<video_id>_gaze_xy_in_mp4.csv).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Dir to save gaze-augmented videos.",
    )
    parser.add_argument(
        "--frame_size",
        type=int,
        default=256,
        help="Output frame size (frame_size x frame_size). Default: 256.",
    )
    parser.add_argument(
        "--fps_out",
        type=float,
        default=2.0,
        help="FPS for the output videos. Default: 2.0.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=4,
        help="Number of frames originally extracted per clip. Default: 4.",
    )
    parser.add_argument(
        "--crop_window_size",
        type=int,
        default=512,
        help="Size (in pixels) of the square crop around the gaze in the original frame. "
             "This crop is then downsampled to --frame_size. Default: 512.",
    )

    args = parser.parse_args()

    agg_json_path = Path(args.agg_json)
    hd_epic_root = Path(args.hd_epic_root)
    frames_root = Path(args.frames_root)
    gaze_root = Path(args.gaze_root)
    out_dir = Path(args.out_dir)

    data = load_questions(agg_json_path)
    questions = data.get("questions", [])
    print(f"Found {len(questions)} questions in {agg_json_path}")

    num_ok = 0
    num_fail = 0

    for q_entry in questions:
        print("\nProcessing q_key:", q_entry.get("q_key", ""))
        ok = process_question(
            q_entry=q_entry,
            hd_epic_root=hd_epic_root,
            frames_root=frames_root,
            gaze_root=gaze_root,
            out_dir=out_dir,
            frame_size=args.frame_size,
            fps_out=args.fps_out,
            num_frames=args.num_frames,
            crop_window_size=args.crop_window_size,
        )
        if ok:
            num_ok += 1
        else:
            num_fail += 1

    print(f"\nDone. Succeeded: {num_ok}, Failed: {num_fail}")

"""
python /home/aryan/ami/381V-final/381V-final-project/HD-EPIC/add_gaze_crops.py \
  --agg_json "/home/aryan/ami/381V-final/381V-final-project/HD-EPIC/test_vqa.json" \
  --hd_epic_root "/home/aryan/ami/381V-final/data" \
  --frames_root "/home/aryan/ami/381V-final/data/trimmed_clips_2" \
  --gaze_root "/home/aryan/ami/381V-final/data/xy_gaze" \
  --out_dir "/home/aryan/ami/381V-final/data/both_512" \
  --frame_size 512
"""

if __name__ == "__main__":
    main()