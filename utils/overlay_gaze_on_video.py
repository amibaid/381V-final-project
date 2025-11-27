#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

"""
python overlay_gaze_on_video.py \
  --video "/content/drive/MyDrive/381V final project/eval data/P01/P01-20240202-110250.mp4" \
  --gaze_csv "/content/drive/MyDrive/381V final project/eval data/P01/P01-20240202-110250_gaze_xy_in_mp4_novrs.csv" \
  --output "/content/drive/MyDrive/381V final project/eval data/P01/P01-20240202-110250_with_gaze.mp4" \
  --max_frames 500

"""


def parse_args():
    p = argparse.ArgumentParser(
        description="Overlay gaze points on a video and save a preview MP4."
    )
    p.add_argument(
        "--video",
        required=True,
        help="Path to the input MP4 video (e.g., P01-20240202-110250.mp4).",
    )
    p.add_argument(
        "--gaze_csv",
        required=True,
        help="CSV with columns [frame_idx, ..., gaze_x, gaze_y].",
    )
    p.add_argument(
        "--output",
        default="video_with_gaze.mp4",
        help="Output video path (default: video_with_gaze.mp4).",
    )
    p.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional: only process first N frames (for quick inspection).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.video)
    gaze_path = Path(args.gaze_csv)
    out_path = Path(args.output)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not gaze_path.exists():
        raise FileNotFoundError(f"Gaze CSV not found: {gaze_path}")

    # -------------------------------------------------
    # 1) Load gaze CSV
    # -------------------------------------------------
    df = pd.read_csv(gaze_path)

    # Try to figure out column names
    if "frame_idx" in df.columns:
        frame_col = "frame_idx"
    else:
        # Fallback: assume first column is frame_idx
        frame_col = df.columns[0]
        print(f"[WARN] Using '{frame_col}' as frame index column.")

    if "gaze_x" not in df.columns or "gaze_y" not in df.columns:
        raise RuntimeError(
            f"CSV {gaze_path} must contain 'gaze_x' and 'gaze_y' columns."
        )

    # For fast lookup: group by frame_idx
    gaze_by_frame = df.groupby(frame_col)

    # Quick sanity stats
    x = df["gaze_x"].values
    y = df["gaze_y"].values
    print(f"Loaded {len(df)} gaze rows.")
    print(f"gaze_x: min={np.nanmin(x):.2f}, max={np.nanmax(x):.2f}")
    print(f"gaze_y: min={np.nanmin(y):.2f}, max={np.nanmax(y):.2f}")

    # -------------------------------------------------
    # 2) Open video and create writer
    # -------------------------------------------------
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"Video size: {width} x {height}, FPS={fps:.2f}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    # -------------------------------------------------
    # 3) Loop over frames and overlay gaze
    # -------------------------------------------------
    frame_idx = 0
    max_frames = args.max_frames

    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        # Check if we have gaze for this frame
        if frame_idx in gaze_by_frame.groups:
            rows = gaze_by_frame.get_group(frame_idx)

            for _, row in rows.iterrows():
                gx = row["gaze_x"]
                gy = row["gaze_y"]

                if not (np.isfinite(gx) and np.isfinite(gy)):
                    continue

                # Only draw if inside bounds
                if 0 <= gx < width and 0 <= gy < height:
                    # Red circle
                    center = (int(round(gx)), int(round(gy)))
                    cv2.circle(frame, center, 10, (0, 0, 255), 2)
                    # Small crosshair
                    cv2.line(
                        frame,
                        (center[0] - 5, center[1]),
                        (center[0] + 5, center[1]),
                        (0, 0, 255),
                        1,
                    )
                    cv2.line(
                        frame,
                        (center[0], center[1] - 5),
                        (center[0], center[1] + 5),
                        (0, 0, 255),
                        1,
                    )
                else:
                    # Optional: visualize out-of-bounds by drawing a marker on the border
                    # For now we just skip them.
                    pass

        # Write frame with overlay
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"Done. Wrote video with gaze overlay to: {out_path}")


if __name__ == "__main__":
    main()
