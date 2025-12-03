import json
import argparse
from pathlib import Path

def build_mcq_prompt(question: str, choices: list[str]) -> str:
    """
    Build a multiple-choice prompt. The model is asked to output ONLY
    the index of the correct choice (0, 1, 2, ...).
    """
    choices_text = "\n".join(f"{i}. {c}" for i, c in enumerate(choices))
    prompt = f"""You are given a video segment and a multiple-choice question about it.

    Question:
    {question}

    Choices:
    {choices_text}

    Respond with the index of the correct choice (0-{len(choices) - 1}) ONLY.
    Do not output any words or explanation, just a single integer."""
    return prompt

def convert_vqa_json(input_path: str, output_path: str):
    with open(input_path, "r") as f:
        data = json.load(f)

    converted = []

    for q in data["questions"]:
        q_data = q["q_data"]

        # --- Extract question text ---
        question_text = q_data["question"]

        # --- Extract choices text ---
        choices = q_data["choices"]

        prompt = build_mcq_prompt(question_text, choices)

        # Build human question block
        human_value = "<video>\n" + prompt

        # --- Determine correct answer ---
        gpt_value = str(q_data["correct_idx"])

        # --- Video path ---
        video_path = q["q_key"] + ".mp4" 

        # --- Append one converted entry ---
        converted.append(
            {
                "video": video_path,
                "conversations": [
                    {"from": "human", "value": human_value},
                    {"from": "gpt", "value": gpt_value}
                ]
            }
        )

    # --- Save output ---
    with open(output_path, "w") as f:
        json.dump(converted, f, indent=2)

    print(f"Saved {len(converted)} converted questions to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert HD-EPIC VQA JSON to ChatGPT conversation format.")
    parser.add_argument("--input", required=True, help="Path to input VQA JSON file")
    parser.add_argument("--output", required=True, help="Path to save converted JSON file")

    args = parser.parse_args()

    convert_vqa_json(args.input, args.output)

if __name__ == "__main__":
    main()

