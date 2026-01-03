from __future__ import annotations

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="e.g., runs/dynamics/mnist/rsbs_track")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    clients_path = os.path.join(args.run_dir, "clients_round.csv")
    df = pd.read_csv(clients_path)

    tracked = df[df["tracked"] == 1].copy()
    if tracked.empty:
        raise SystemExit("No tracked clients found. Run dynamics script first.")

    # Plot epsilon_total trajectories
    plt.figure()
    for cid, g in tracked.groupby("cid"):
        plt.plot(g["round"], g["epsilon_total"], label=f"cid={cid}")
    plt.xlabel("Round")
    plt.ylabel("Epsilon_k(t)")
    plt.title("Budget Trajectories (Tracked Clients)")
    plt.legend()

    out = args.out or os.path.join(args.run_dir, "plot_budget_traj.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")

    # Plot eta trajectories
    plt.figure()
    for cid, g in tracked.groupby("cid"):
        plt.plot(g["round"], g["eta"], label=f"cid={cid}")
    plt.xlabel("Round")
    plt.ylabel("Eta_k(t)")
    plt.title("Split Coefficient Trajectories (Tracked Clients)")
    plt.legend()

    out2 = os.path.join(args.run_dir, "plot_eta_traj.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
