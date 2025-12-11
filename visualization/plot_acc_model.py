import matplotlib.pyplot as plt

def main():
    # ================== EDIT THESE NUMBERS ==================
    # Training steps
    steps = [0, 1000, 2500]

    # Accuracies (%) for each schedule
    # Example values – replace with your real numbers
    acc_ego = [23.04, 25.77, 28.19]      # finetuned on ego data
    acc_our = [24.47, 26.55, 28.72]      # finetuned on our data
    # =======================================================

    plt.figure(figsize=(8, 5))

    # Plot both time series
    plt.plot(steps, acc_ego, marker="o", markersize=8, linewidth=2,
             label="Original data 512x512", zorder=5, clip_on=False)
    plt.plot(steps, acc_our, marker="o", markersize=8, linewidth=2,
             label="Original + Gaze Crop 256x256 (Ours)", zorder=5, clip_on=False)

    # Labels and title
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("HD-EPIC Test Accuracy (%)", fontsize=12)
    plt.title("Accuracy over Finetuning Steps", fontsize=14)

    # X-axis from 0 to 2500 as requested
    plt.xlim(0, 2500)

    # Optional: make x-ticks exactly at your steps
    plt.xticks(steps, [str(s) for s in steps])

    # Light grid + legend
    plt.grid(True, linestyle="--", alpha=0.4, zorder=0)  # keep grid behind points
    plt.legend(fontsize=11)

    # Tight layout and save
    plt.tight_layout()
    plt.savefig("/home/aryan/ami/381V-final/381V-final-project/viz/finetune_time_series.png", dpi=300, bbox_inches="tight")
    # plt.show()  # uncomment if running interactively

if __name__ == "__main__":
    main()
