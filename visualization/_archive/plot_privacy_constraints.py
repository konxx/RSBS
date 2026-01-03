from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _read_meta_budget(run_dir: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Try to read B_total / epsilon_min / epsilon_max from meta.json.
    Returns (B_total, eps_min, eps_max) possibly None if missing.
    """
    meta_path = os.path.join(run_dir, "meta.json")
    if not os.path.exists(meta_path):
        return None, None, None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        priv = meta.get("privacy_config", {})
        budget = priv.get("budget", {})
        B_total = budget.get("B_total", None)
        eps_min = budget.get("epsilon_min", None)
        eps_max = budget.get("epsilon_max", None)

        return (
            float(B_total) if B_total is not None else None,
            float(eps_min) if eps_min is not None else None,
            float(eps_max) if eps_max is not None else None,
        )
    except Exception:
        return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="e.g., runs/mnist_dyn/mnist/rsbs_track")
    ap.add_argument("--out", type=str, default=None, help="output png path; default in run_dir")

    # Optional overrides (if meta.json is missing)
    ap.add_argument("--B_total", type=float, default=None)
    ap.add_argument("--eps_min", type=float, default=None)
    ap.add_argument("--eps_max", type=float, default=None)

    # Numerical tolerance for boundary-touch 판단
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    run_dir = args.run_dir

    metrics_path = os.path.join(run_dir, "metrics_round.csv")
    clients_path = os.path.join(run_dir, "clients_round.csv")

    if not os.path.exists(metrics_path):
        raise SystemExit(f"Missing file: {metrics_path}")
    if not os.path.exists(clients_path):
        raise SystemExit(f"Missing file: {clients_path}")

    metrics = pd.read_csv(metrics_path)
    clients = pd.read_csv(clients_path)

    # meta.json -> budget params
    meta_B, meta_min, meta_max = _read_meta_budget(run_dir)
    B_total = args.B_total if args.B_total is not None else meta_B
    eps_min = args.eps_min if args.eps_min is not None else meta_min
    eps_max = args.eps_max if args.eps_max is not None else meta_max

    if B_total is None or eps_min is None or eps_max is None:
        raise SystemExit(
            "Cannot infer B_total/eps_min/eps_max. "
            "Please provide --B_total --eps_min --eps_max, or ensure meta.json exists."
        )

    # Round axis (metrics includes round=0 initial eval)
    # For constraints, clients_round starts from round>=1.
    metrics = metrics.sort_values("round").reset_index(drop=True)
    rounds = metrics["round"].to_numpy()

    # Budget utilization
    budget_used = metrics["budget_used"].to_numpy(dtype=float)
    utilization = budget_used / float(B_total)

    # Boundary touches per round (selected clients only)
    tol = float(args.tol)

    # Ensure epsilon_total exists
    if "epsilon_total" not in clients.columns:
        raise SystemExit("clients_round.csv missing column: epsilon_total")

    clients = clients.sort_values(["round", "cid"]).reset_index(drop=True)
    touch_low = []
    touch_high = []
    touch_rounds = []

    for r, grp in clients.groupby("round"):
        eps = grp["epsilon_total"].to_numpy(dtype=float)
        low_cnt = int(np.sum(eps <= float(eps_min) + tol))
        high_cnt = int(np.sum(eps >= float(eps_max) - tol))
        touch_rounds.append(int(r))
        touch_low.append(low_cnt)
        touch_high.append(high_cnt)

    touch_rounds = np.array(touch_rounds, dtype=int)
    touch_low = np.array(touch_low, dtype=int)
    touch_high = np.array(touch_high, dtype=int)

    # Plot: utilization (left axis), boundary touches (right axis)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.plot(rounds, utilization)
    ax1.axhline(1.0, linestyle="--")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Budget Pool Utilization (budget_used / B_total)")
    ax1.set_title("Privacy/Constraint Curves: Utilization + Boundary Touches")

    ax2 = ax1.twinx()
    ax2.plot(touch_rounds, touch_low, linestyle="-")
    ax2.plot(touch_rounds, touch_high, linestyle="-")
    ax2.set_ylabel("Boundary Touch Count (selected clients)")

    # Legend (no fixed colors; let matplotlib default)
    lines = ax1.get_lines() + ax2.get_lines()
    labels = ["utilization", "utilization=1.0", "touch eps_min", "touch eps_max"]
    ax1.legend(lines, labels, loc="best")

    out = args.out or os.path.join(run_dir, "plot_privacy_constraints.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
