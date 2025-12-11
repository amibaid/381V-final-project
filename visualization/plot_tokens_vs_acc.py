import matplotlib.pyplot as plt

def main():
    methods = [
        "Original 256x256",
        "Gaze crop 256x256",
        "Original 512x512",
        "Original + Gaze Crop (ours)",
        "Gaze crop 512x512",
    ]
    tokens = [512, 512, 2048, 1024, 2048]
    accuracies = [23.04, 20.81, 23.04, 24.47, 30.32]

    # Define colors for each method to highlight our method in orange
    colors = [
        "tab:blue",   # Orig 256x256
        "tab:blue",   # Gaze crop 256x256
        "tab:blue",   # Orig 512x512
        "orange",     # Orig + Gaze (ours) ← highlighted in orange
        "tab:blue",   # Gaze crop 512x512
    ]
    random_guess = 20.0

    plt.figure(figsize=(10, 6))

    # Bigger dots
    plt.scatter(tokens, accuracies, s=150, c=colors)   # ← increased dot size & custom colors

    # Bigger text labels
    for x, y, label in zip(tokens, accuracies, methods):
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(0, 8),                   # slightly above point
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=15,                     # ← larger label text
            fontweight="medium"
        )

    plt.axhline(random_guess, linestyle="--", linewidth=1.2, label="Random guessing (20%)")

    plt.xlabel("Vision tokens", fontsize=15)
    plt.ylabel("HD-EPIC % accuracy", fontsize=15)
    plt.title("Effect of Vision Tokens and Preprocessing Method on Accuracy", fontsize=16)

    plt.xlim(250, 2250)
    plt.ylim(15, 35)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=15)

    plt.tight_layout()
    plt.savefig("/home/aryan/ami/381V-final/381V-final-project/viz/tokens_vs_accuracy_scatter.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    main()
