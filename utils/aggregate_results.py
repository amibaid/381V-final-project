import pandas as pd
from pathlib import Path
import argparse


def load_and_aggregate(csv_dir: str):
    csv_dir = Path(csv_dir)
    csv_files = sorted(csv_dir.glob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in: {csv_dir}")

    # Load and concatenate
    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)

    # Ensure correctness field is int
    df["is_correct"] = df["is_correct"].astype(int)

    # Extract question type (prefix before the last underscore)
    df["question_type"] = df["question_id"].apply(
        lambda x: "_".join(x.split("_")[:-1])
    )

    # ---- Aggregate metrics ----
    total_questions = len(df)
    total_correct = df["is_correct"].sum()
    overall_accuracy = total_correct / total_questions if total_questions else 0

    # ---- Per question-type accuracy ----
    per_type = (
        df.groupby("question_type")["is_correct"]
        .mean()
        .reset_index()
        .rename(columns={"is_correct": "accuracy"})
        .sort_values("accuracy", ascending=False)
    )

    return df, overall_accuracy, per_type


def main():
    parser = argparse.ArgumentParser(description="Aggregate accuracy metrics from CSVs.")
    parser.add_argument(
        "--csv_dir",
        type=str,
        required=True,
        help="Directory containing CSV files to aggregate.",
    )
    args = parser.parse_args()

    df, overall_acc, per_type_acc = load_and_aggregate(args.csv_dir)

    print("=== Overall Accuracy ===")
    print(f"{overall_acc:.4f}")

    print("\n=== Accuracy by Question Type ===")
    print(per_type_acc.to_string(index=False))


if __name__ == "__main__":
    main()

