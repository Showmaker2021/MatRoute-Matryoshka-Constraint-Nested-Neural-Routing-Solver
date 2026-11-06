'''
Plots training curves (RL loss, Distillation loss, Total loss, Metric) from train_log.csv.
'''

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_curves(csv_path: str, output_image_path: str):
    if not os.path.exists(csv_path):
        print(f"Log CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print("Empty log file.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. RL Loss
    axes[0, 0].plot(df["step"], df["loss_rl"], color="tab:blue", label="RL Loss")
    axes[0, 0].set_title("REINFORCE Loss (with POMO baseline)")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)

    # 2. Distillation Loss
    axes[0, 1].plot(df["step"], df["loss_dist"], color="tab:orange", label="Distill Loss")
    axes[0, 1].set_title("On-Policy Distillation Loss (KL)")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("KL Divergence")
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)

    # 3. Total Loss
    axes[1, 0].plot(df["step"], df["total_loss"], color="tab:purple", label="Total Loss")
    axes[1, 0].set_title("Total Combined Loss")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)

    # 4. Cost Metric by variant
    for variant in df["variant"].unique():
        sub_df = df[df["variant"] == variant]
        axes[1, 1].scatter(sub_df["step"], sub_df["cost_metric"], alpha=0.7, s=15, label=variant)
    axes[1, 1].set_title("Solution Cost Metric by Variant")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Cost")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_image_path)), exist_ok=True)
    plt.savefig(output_image_path, dpi=150)
    plt.close()
    print(f"Saved training curves plot to {output_image_path}")

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "results/local/smoke/train_log.csv"
    img_file = sys.argv[2] if len(sys.argv) > 2 else "results/local/micro_curves.png"
    plot_curves(csv_file, img_file)
