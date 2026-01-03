from __future__ import annotations

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="e.g., runs/comparison/mnist/rsbs")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    metrics_path = os.path.join(args.run_dir, "metrics_round.csv")
    df = pd.read_csv(metrics_path)

    plt.figure()
    plt.plot(df["round"], df["fairness_var"])
    plt.xlabel("Round")
    plt.ylabel("Var(Reward Proxy)")
    plt.title("Fairness (Variance Proxy) Across Rounds")

    out = args.out or os.path.join(args.run_dir, "plot_fairness_var.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
