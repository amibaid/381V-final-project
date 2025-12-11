import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# Edit this dictionary to add/remove models and accuracies
# ============================================================

model_results = {
    "Orig-Gaze-256 (Ours)": [
        ("fine_grained_why_recognition", 0.416667),
        ("gaze_gaze_estimation", 0.294118),
        ("fine_grained_action_recognition", 0.285714),
        ("fine_grained_how_recognition", 0.250000),
        ("fine_grained_action_localization", 0.176471),
        ("gaze_interaction_anticipation", 0.150943),
    ],
    "Orig-512": [
        ("fine_grained_action_recognition", 0.428571),
        ("fine_grained_why_recognition", 0.416667),
        ("gaze_gaze_estimation", 0.333333),
        ("fine_grained_how_recognition", 0.333333),
        ("gaze_interaction_anticipation", 0.113208),
        ("fine_grained_action_localization", 0.058824),
    ],
    "Gaze-512": [
        ("fine_grained_why_recognition", 0.625000),
        ("gaze_gaze_estimation", 0.509804),
        ("fine_grained_action_recognition", 0.357143),
        ("fine_grained_how_recognition", 0.333333),
        ("fine_grained_action_localization", 0.088235),
        ("gaze_interaction_anticipation", 0.075472),
    ]
}

# model_results = {
#     "Orig-Gaze-256 (Ours)": [
#         ("fine_grained_action_recognition", 0.500000),
#         ("gaze_gaze_estimation", 0.392157),
#         ("fine_grained_why_recognition", 0.333333),
#         ("fine_grained_action_localization", 0.235294),
#         ("gaze_interaction_anticipation", 0.169811),
#         ("fine_grained_how_recognition", 0.166667),
#     ],
#     "Orig-512": [
#         ("gaze_gaze_estimation", 0.450980),
#         ("fine_grained_action_recognition", 0.428571),
#         ("fine_grained_how_recognition", 0.416667),
#         ("fine_grained_why_recognition", 0.291667),
#         ("fine_grained_action_localization", 0.205882),
#         ("gaze_interaction_anticipation", 0.094340),
#     ],
# }

# ============================================================
# Convert to a single DataFrame for grouped bar plotting
# ============================================================

df = pd.DataFrame()

for model_name, rows in model_results.items():
    temp = pd.DataFrame(rows, columns=["question_type", model_name])
    if df.empty:
        df = temp
    else:
        df = pd.merge(df, temp, on="question_type", how="outer")

df = df.set_index("question_type")
df = df.sort_index()

# ============================================================
# Global font sizes (bigger everything)
# ============================================================
plt.rcParams.update({
    "font.size": 18,          # base font size
    "axes.titlesize": 28,
    "axes.labelsize": 24,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 20,
    "legend.title_fontsize": 22,
})

# ============================================================
# Plot grouped bar chart with big fonts
# ============================================================
fig, ax = plt.subplots(figsize=(18, 8))

df.plot(kind="bar", ax=ax)

ax.set_title("Accuracy by Question Type")
ax.set_ylabel("Accuracy")
ax.set_xlabel("Question Type")

ax.set_ylim(0, 1)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

ax.legend(title="Input")

fig.tight_layout()

output_filename = "accuracy_by_question_type_big.png"
fig.savefig(output_filename, dpi=300)

print(f"Saved bar chart to: {output_filename}")

plt.show()
