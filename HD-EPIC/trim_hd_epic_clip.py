import argparse
import json
import re
import subprocess
from pathlib import Path

"""
python trim_hd_epic_clip.py \
  --agg_json "/content/drive/MyDrive/381V-final-project/HD-EPIC/test_vqa.json" \
  --hd_epic_root "/content/eval_data" \
  --out_dir "/content/drive/MyDrive/381V final project/eval data/trimmed_clips"
"""


def load_questions(json_path: Path) -> dict:
    with json_path.open("r") as f:
        return json.load(f)


# def find_qa_entry(data: dict, src_file: str, q_key: str) -> dict:
#     """
#     Find the question entry in data["questions"] matching given src_file and q_key.
#     Returns the whole entry (with keys: src_file, q_key, q_data).
#     """
#     for entry in data["questions"]:
#         if entry.get("src_file") == src_file and entry.get("q_key") == q_key:
#             return entry
#     raise ValueError(f"No entry found for src_file={src_file}, q_key={q_key}")


def get_video_path_from_qdata(
    hd_epic_root: Path, q_entry: dict
) -> Path:
    """
    Given the question entry, derive the path to the corresponding .mp4 video.

    Assumes layout:
        hd_epic_root / <participant> / <video_id>.mp4
    where video_id looks like "P01-20240202-110250".
    """
    inputs = q_entry["q_data"]["inputs"]
    # assuming only "video 1" is present
    vid_meta = inputs["video 1"]
    video_id = vid_meta["id"]          # e.g., "P01-20240202-110250"
    participant = video_id.split("-")[0]  # e.g., "P01"

    video_path = hd_epic_root / participant / f"{video_id}.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found at {video_path}")
    return video_path


def get_clip_window_from_inputs(q_entry: dict) -> tuple[str, str]:
    """
    Use the clip window specified in the inputs['video 1'] metadata.
    Returns (start_time, end_time) as strings like "HH:MM:SS.mmm".
    """
    inputs = q_entry["q_data"]["inputs"]
    vid_meta = inputs["video 1"]
    start_time = vid_meta["start_time"]
    end_time = vid_meta["end_time"]
    if not start_time or not end_time:
        raise ValueError(f"Missing start_time or end_time in inputs: {vid_meta}")
    return start_time, end_time


def trim_video_ffmpeg(
    input_video: Path,
    start_time: str,
    end_time: str,
    output_video: Path,
    reencode: bool = False,
) -> None:
    """
    Trim the segment [start_time, end_time] from input_video and write to output_video.

    - If reencode=False (default), uses `-c copy` for fast, keyframe-based trimming.
    - If you need frame-exact trims, set reencode=True (slower).
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        start_time,
        "-to",
        end_time,
        "-i",
        str(input_video),
    ]

    if reencode:
        # Let ffmpeg choose defaults; you can customize codecs if you want.
        cmd += ["-y", str(output_video)]
    else:
        # Fast, no re-encode
        cmd += ["-c", "copy", "-y", str(output_video)]

    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Trim HD-EPIC video segment for a given src_file and q_key."
    )
    parser.add_argument(
        "--agg_json",
        type=str,
        required=True,
        help="Path to aggregated questions JSON (the one with 'questions' list).",
    )
    parser.add_argument(
        "--hd_epic_root",
        type=str,
        required=True,
        help="Path to HD-EPIC root (e.g. /content/data/HD-EPIC).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to save the trimmed clip.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="clip",
        help="Prefix for output filename (default: 'clip').",
    )
    parser.add_argument(
        "--reencode",
        action="store_true",
        help="If set, re-encode instead of stream-copy (slower but frame-accurate).",
    )

    args = parser.parse_args()

    agg_json_path = Path(args.agg_json)
    hd_epic_root = Path(args.hd_epic_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_questions(agg_json_path)
    questions = data.get("questions", [])

    print(f"Found {len(questions)} questions in {agg_json_path}")

    num_processed = 0
    num_failed = 0

    for q_entry in questions:
        src_file = q_entry.get("src_file", "")
        q_key = q_entry.get("q_key", "")

        try:
            video_path = get_video_path_from_qdata(hd_epic_root, q_entry)
            start_time, end_time = get_clip_window_from_inputs(q_entry)

            # Build output filename: prefix_src_qkey_start_end.mp4 (sanitized)
            safe_q_key = q_key.replace("/", "_")
            output_name = f"{safe_q_key}.mp4"
            output_path = out_dir / output_name

            print(f"\nTrimming {video_path}")
            print(f"  src_file: {src_file}")
            print(f"  q_key:    {q_key}")
            print(f"  Segment:  {start_time} -> {end_time}")
            print(f"  Saving to: {output_path}")

            trim_video_ffmpeg(
                input_video=video_path,
                start_time=start_time,
                end_time=end_time,
                output_video=output_path,
                reencode=args.reencode,
            )
            num_processed += 1

        except Exception as e:
            print(f"!!! Failed for src_file={src_file}, q_key={q_key}: {e}")
            num_failed += 1
            continue

    print(f"\nDone. Processed clips: {num_processed}, failed: {num_failed}")


if __name__ == "__main__":
    main()

