#!/usr/bin/env python

"""
Make two QA figures:

  - qa_successes.png  : (a) Successes (correct QA)
  - qa_failures.png   : (b) Failures (incorrect QA)

You hard-code examples in success_examples and failure_examples.
Each example:
  - panel: "success" or "failure" (not really needed now but kept)
  - folder: path with frames
  - category, question, gold_answer, predicted_answer

One example -> two grid rows:
  Row 0: frames
  Row 1: text spanning full width

If `question` contains '\\n', the line breaks are preserved.
"""

import os
import glob
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# --------------- config you will edit ---------------------------------------

@dataclass
class QAExample:
    panel: str           # "success" or "failure" (kept for clarity)
    folder: str          # folder with frames for this example
    category: str
    question: str        # you can include '\n' for manual line breaks
    gold_answer: str
    predicted_answer: str


# fill these with your own paths/text
success_examples: List[QAExample] = [
    QAExample(
        panel="success",
        folder="/home/aryan/ami/381V-final/data/gaze_crops_256/fine_grained_action_recognition_280_gaze_frames/",
        category="Fine-grained action recognition",
        question=(
         "Question: Which of these sentences best describe the action(s) in the video?\n\n"
         "Choices:\n"
         "0. Pick up the plates from the shelf under the cutlery drawer.\n"
         "1. Using my right hand to pick up plates while stacking them in my left hand.\n"
         "2. Pick up both plates that were used for eating from the tabletop.\n"
         "3. Pick up plates with salad bowl and bring to center of kitchen counter.\n"
         "4. Pick up plates with fork and move to side to clean up."
        ),
        gold_answer="2",
        predicted_answer="2 ✅",
    ),
]

failure_examples: List[QAExample] = [
    QAExample(
        panel="failure",
        folder="/home/aryan/ami/381V-final/data/gaze_crops_256/gaze_gaze_estimation_262_gaze_frames/",
        category="Gaze estimation",
        question=(
            "Question: What is the person looking at in this video segment?\n\n"
            "Choices:\n"
            "0. At the sink.\n"
            "1. At the microwave.\n"
            "2. At the counter to the right of the microwave.\n"
            "3. At the washing machine.\n"
            "4. At the counter to the right of the sink"

        ),
        gold_answer="4",
        predicted_answer="2 ❌",
    ),
]

# max frames to show per example (None = all)
MAX_FRAMES: Optional[int] = 6

# ---------------------------------------------------------------------------

def load_frames(folder: str, max_frames: Optional[int]) -> List[Image.Image]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(folder, p)))
    files.sort()
    if not files:
        raise FileNotFoundError(f"No images found in {folder}")
    if max_frames is not None:
        files = files[:max_frames]
    return [Image.open(f) for f in files]


def make_panel_figure(
    examples: List[QAExample],
    title: str,
    output_path: str,
    max_frames: Optional[int] = MAX_FRAMES,
):
    if not examples:
        print(f"No examples for {title}, skipping.")
        return

    # work out max number of frames over all examples (for consistent columns)
    max_cols = 1
    for ex in examples:
        n = len(load_frames(ex.folder, max_frames))
        max_cols = max(max_cols, n)

    n_examples = len(examples)
    n_rows = 2 * n_examples  # images row + text row for each example

    # figure size: width fixed, height grows with number of examples
    fig_height = 2.2 + 1.6 * (n_examples - 1)
    fig = plt.figure(figsize=(8, fig_height*2), dpi=300)

    gs = GridSpec(
        n_rows,
        max_cols,
        figure=fig,
        wspace=0.02,
        hspace=0.15,
    )

    for i, ex in enumerate(examples):
        img_row = 2 * i
        txt_row = 2 * i + 1

        # ---- image row ----
        imgs = load_frames(ex.folder, max_frames)
        for col in range(max_cols):
            ax = fig.add_subplot(gs[img_row, col])
            if col < len(imgs):
                ax.imshow(imgs[col])
            ax.axis("off")

        # ---- text row ----
        text_ax = fig.add_subplot(gs[txt_row, :])
        text_ax.axis("off")
        text = (
            f"Category: {ex.category}\n\n"
            f"Q: {ex.question}\n\n"
            f"Ground truth: {ex.gold_answer}\n"
            f"Predicted: {ex.predicted_answer}"
        )
        text_ax.text(
            0.0,
            1.0,
            text,
            fontsize=8,
            va="top",
            ha="left",
            transform=text_ax.transAxes,
        )

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    make_panel_figure(
        success_examples,
        title="(a) Successes (correct QA)",
        output_path="qa_successes.png",
    )
    make_panel_figure(
        failure_examples,
        title="(b) Failures (incorrect QA)",
        output_path="qa_failures.png",
    )
