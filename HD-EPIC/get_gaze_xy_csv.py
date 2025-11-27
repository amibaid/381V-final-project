#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import sys
import subprocess

from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection

"""
Usage ex:
python3 get_gaze_xy_csv.py --root /Users/amibaid/Downloads/381V_final/hd-epic-downloader/data/HD-EPIC --participant P02 --recording all
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Path to HD-EPIC root")
    p.add_argument("--participant", default="P01")
    p.add_argument("--recording", required=True)
    p.add_argument("--use_personalized_gaze", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    root = Path(args.root)
    participant = args.participant
    rec = args.recording

    if rec == "all":
        videos_dir = root / "Videos" / participant
        pattern = f"{participant}-*_mp4_to_vrs_time_ns.csv"

        for csv_path in sorted(videos_dir.glob(pattern)):
            stem = csv_path.stem  # e.g. P01-20240203-184045_mp4_to_vrs_time_ns
            prefix = f"{participant}-"
            suffix = "_mp4_to_vrs_time_ns"
            if not (stem.startswith(prefix) and stem.endswith(suffix)):
                continue

            rec_id = stem[len(prefix):-len(suffix)]
            print(f"\n=== Running gaze export for {participant}-{rec_id} ===")

            cmd = [
                sys.executable,
                __file__,
                "--root", str(root),
                "--participant", participant,
                "--recording", rec_id,
            ]
            if args.use_personalized_gaze:
                cmd.append("--use_personalized_gaze")

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[WARN] Failed on {participant}-{rec_id}: {e}")

        return  # don't run the single-recording code below

    
    # single recording 

    # Paths
    videos_dir = root / "Videos" / participant
    # mp4_path = videos_dir / f"{participant}-{rec}.mp4"
    mapping_csv = videos_dir / f"{participant}-{rec}_mp4_to_vrs_time_ns.csv"

    vrs_path = (
        root / "VRS" / participant / f"{participant}-{rec}_anonymized.vrs"
    )

    completed_log = Path("/Users/amibaid/Downloads/381V_final/hd-epic-downloader/data/completed_downloads.txt")
    if completed_log.exists():
        completed = {
            line.strip()
            for line in completed_log.read_text().splitlines()
            if line.strip()
        }
        vrs_filename = vrs_path.name
        if vrs_filename not in completed:
            raise RuntimeError(
                f"VRS file {vrs_filename} is not listed in {completed_log}. "
                "Assuming incomplete download; aborting."
            )
    else:
        print(
            "[WARN] completed_downloads.txt not found; "
            "skipping VRS completeness check."
        )

    mps_root = (
        root
        / "SLAM-and-Gaze"
        / participant
        / "GAZE_HAND"
        / f"mps_{participant}-{rec}_vrs"
    )

    # if not mp4_path.exists():
    #     raise FileNotFoundError(f"MP4 not found: {mp4_path}")
    if not mapping_csv.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {mapping_csv}")
    if not vrs_path.exists():
        raise FileNotFoundError(f"VRS not found: {vrs_path}")
    if not mps_root.exists():
        raise FileNotFoundError(f"MPS root not found: {mps_root}")

    # -------------------------------------------------
    # 1) Load MP4 → VRS device time mapping
    # -------------------------------------------------
    print("Loading MP4→VRS timestamp mapping...")
    df_map = pd.read_csv(mapping_csv)

    # In your file, the relevant column is 'vrs_device_time_ns'
    # but we support a few possible names just in case.
    candidate_cols = [
        "vrs_device_time_ns",
        "vrs_timestamp_ns",
        "vrs_time_ns",
        "aria_vrs_timestamp_ns",
    ]

    ts_col = None
    for c in candidate_cols:
        if c in df_map.columns:
            ts_col = c
            break

    if ts_col is None:
        raise RuntimeError(
            f"Could not find a VRS timestamp column in {mapping_csv}.\n"
            f"Available columns: {list(df_map.columns)}"
        )

    print(f"Using VRS timestamp column: {ts_col}")
    ts_vrs = df_map[ts_col].astype(np.int64).values
    num_frames = len(ts_vrs)
    print(f"Found {num_frames} MP4 frames in mapping file.")

    # -------------------------------------------------
    # 2) Open VRS and get RGB camera calibration
    # -------------------------------------------------
    print("Opening VRS...")
    vrs_dp = data_provider.create_vrs_data_provider(str(vrs_path))
    if vrs_dp is None:
        raise RuntimeError("Failed to create VRS data provider")

    device_calib = vrs_dp.get_device_calibration()

    # Aria RGB is typically stream "214-1"
    rgb_stream_id = StreamId("214-1")
    rgb_label = vrs_dp.get_label_from_stream_id(rgb_stream_id)
    if rgb_label is None:
        raise RuntimeError(
            "Could not find RGB stream label for StreamId('214-1'). "
            "Inspect your VRS streams if this fails."
        )

    rgb_calib = device_calib.get_camera_calib(rgb_label)
    img_cfg = vrs_dp.get_image_configuration(rgb_stream_id)
    width = img_cfg.image_width
    height = img_cfg.image_height
    print(f"RGB image size (VRS orientation): {width} x {height}")

    # -------------------------------------------------
    # 3) Open MPS gaze data
    # -------------------------------------------------
    print("Opening MPS gaze data...")
    mps_paths = mps.MpsDataPathsProvider(str(mps_root)).get_data_paths()
    mps_dp = mps.MpsDataProvider(mps_paths)

    use_general = not args.use_personalized_gaze
    if use_general:
        if not mps_dp.has_general_eyegaze():
            raise RuntimeError("No general_eye_gaze available in MPS data.")
        print("Using general eye gaze.")
    else:
        if not mps_dp.has_personalized_eyegaze():
            raise RuntimeError("No personalized_eye_gaze available in MPS data.")
        print("Using personalized eye gaze.")

    # -------------------------------------------------
    # 4) For each MP4 frame / VRS timestamp, get gaze and project to RGB
    # -------------------------------------------------
    gaze_x = np.full(num_frames, np.nan, dtype=np.float64)
    gaze_y = np.full(num_frames, np.nan, dtype=np.float64)

    print("Projecting gaze for each MP4 frame...")
    for i, ts in enumerate(ts_vrs):
        # ts is already in *device* time nanoseconds
        if use_general:
            eg = mps_dp.get_general_eyegaze(int(ts))
        else:
            eg = mps_dp.get_personalized_eyegaze(int(ts))

        if eg is None:
            # No valid gaze at this timestamp
            continue

        depth = eg.depth or 1.0

        proj = get_gaze_vector_reprojection(
            eg,
            rgb_label,
            device_calib,
            rgb_calib,
            depth,
        )

        if proj is None or np.any(np.isnan(proj)):
            continue

        x_raw, y_raw = float(proj[0]), float(proj[1])

        # -------------------------------------------------
        # 5) Rotate to match upright MP4 orientation
        #
        # VRS 'camera-rgb' frames are rotated when exporting to MP4.
        # Using the same convention as Aria example code:
        #   new_x = W - 1 - y_raw
        #   new_y = x_raw
        # -------------------------------------------------
        x_rot = (width - 1) - y_raw
        y_rot = x_raw

        gaze_x[i] = x_rot
        gaze_y[i] = y_rot

    # -------------------------------------------------
    # 6) Save results: one gaze point per MP4 frame
    # -------------------------------------------------
    out_csv = root / "xy_gaze" / f"{participant}-{rec}_gaze_xy_in_mp4.csv"
    out_df = pd.DataFrame(
        {
            "frame_idx": np.arange(num_frames, dtype=int),
            ts_col: ts_vrs,
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
        }
    )
    out_df.to_csv(out_csv, index=False)

    print(f"Saved per-frame gaze (x, y) to:\n  {out_csv}")


if __name__ == "__main__":
    main()
