from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _read_meta(run_dir: str) -> Tuple[float, float, float]:
    """
    Returns (B_total, eps_min, eps_max). Falls back to NaN if missing.
    """
    meta_path = os.path.join(run_dir, "meta.json")
    if not os.path.exists(meta_path):
        return float("nan"), float("nan"), float("nan")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    priv = meta.get("privacy_config", {})
    budget = priv.get("budget", {})
    B_total = float(budget.get("B_total", float("nan")))
    eps_min = float(budget.get("epsilon_min", float("nan")))
    eps_max = float(budget.get("epsilon_max", float("nan")))
    return B_total, eps_min, eps_max


def _read_tracked(run_dir: str) -> List[int]:
    p = os.path.join(run_dir, "tracked_clients.json")
    if not os.path.exists(p):
        return []
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    track = obj.get("track_list", [])
    return [int(x) for x in track]


def _load_csvs(run_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    m = pd.read_csv(os.path.join(run_dir, "metrics_round.csv"))
    c = pd.read_csv(os.path.join(run_dir, "clients_round.csv"))
    return m, c


def _ensure_cols(df: pd.DataFrame, defaults: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for k, v in defaults.items():
        if k not in out.columns:
            out[k] = v
    return out


def _plot_tracked_curves(run_dir: str, clients_df: pd.DataFrame, tracked: List[int], title: str):
    # epsilon_total
    plt.figure()
    for cid in tracked:
        d = clients_df[clients_df["cid"] == cid].sort_values("round")
        plt.plot(d["round"], d["epsilon_total"], label=f"cid={cid}")
    plt.xlabel("Round")
    plt.ylabel(r"$\epsilon_k^t$ (total)")
    plt.title((title + " Tracked Clients: Total Budget $\\epsilon_k^t$").strip())
    plt.legend()
    plt.savefig(os.path.join(run_dir, "dyn_tracked_epsilon_total.png"), dpi=160, bbox_inches="tight")

    # eta
    plt.figure()
    for cid in tracked:
        d = clients_df[clients_df["cid"] == cid].sort_values("round")
        plt.plot(d["round"], d["eta"], label=f"cid={cid}")
    plt.xlabel("Round")
    plt.ylabel(r"$\eta_k^t$")
    plt.title((title + " Tracked Clients: Split Coefficient $\\eta_k^t$").strip())
    plt.legend()
    plt.savefig(os.path.join(run_dir, "dyn_tracked_eta.png"), dpi=160, bbox_inches="tight")

    # split budgets
    plt.figure()
    for cid in tracked:
        d = clients_df[clients_df["cid"] == cid].sort_values("round")
        plt.plot(d["round"], d["epsilon_up"], label=f"cid={cid} (up)")
        plt.plot(d["round"], d["epsilon_ft"], label=f"cid={cid} (ft)")
    plt.xlabel("Round")
    plt.ylabel(r"Budget ($\epsilon$)")
    plt.title((title + " Tracked Clients: Two-Stage Budgets ($\\epsilon^{up}$ / $\\epsilon^{ft}$)").strip())
    plt.legend()
    plt.savefig(os.path.join(run_dir, "dyn_tracked_split_budget.png"), dpi=160, bbox_inches="tight")


def _plot_boundary_and_projection(run_dir: str, metrics_df: pd.DataFrame, title: str):
    # boundary hits
    plt.figure()
    d = metrics_df.sort_values("round")
    plt.plot(d["round"], d["num_eps_min"], label="# at eps_min")
    plt.plot(d["round"], d["num_eps_max"], label="# at eps_max")
    plt.xlabel("Round")
    plt.ylabel("#Clients on Boundary")
    plt.title((title + " Boundary Hits (eps_min/eps_max)").strip())
    plt.legend()
    plt.savefig(os.path.join(run_dir, "dyn_boundary_hits.png"), dpi=160, bbox_inches="tight")

    # projection triggered (bar)
    plt.figure()
    trig = pd.to_numeric(d["projection_triggered"], errors="coerce").fillna(0.0)
    plt.bar(d["round"], trig)
    plt.xlabel("Round")
    plt.ylabel("Projection Triggered (0/1)")
    plt.title((title + " Budget Pool Projection Trigger").strip())
    plt.savefig(os.path.join(run_dir, "dyn_projection_trigger.png"), dpi=160, bbox_inches="tight")


def _plot_oscillation(run_dir: str, clients_df: pd.DataFrame, tracked: List[int], title: str):
    labels = []
    mean_vals = []
    var_vals = []

    for cid in tracked:
        d = clients_df[clients_df["cid"] == cid].sort_values("round")
        eps = pd.to_numeric(d["epsilon_total"], errors="coerce").to_numpy()
        if len(eps) < 2:
            continue
        delta = np.abs(eps[1:] - eps[:-1])
        labels.append(f"cid={cid}")
        mean_vals.append(float(np.mean(delta)))
        var_vals.append(float(np.var(delta)))

    # mean bar
    plt.figure()
    plt.bar(labels, mean_vals)
    plt.xlabel("Tracked Client")
    plt.ylabel(r"Mean $|\Delta \epsilon|$")
    plt.title((title + " Oscillation: Mean |Δε|").strip())
    plt.savefig(os.path.join(run_dir, "dyn_delta_eps_mean.png"), dpi=160, bbox_inches="tight")

    # var bar
    plt.figure()
    plt.bar(labels, var_vals)
    plt.xlabel("Tracked Client")
    plt.ylabel(r"Var $|\Delta \epsilon|$")
    plt.title((title + " Oscillation: Var |Δε|").strip())
    plt.savefig(os.path.join(run_dir, "dyn_delta_eps_var.png"), dpi=160, bbox_inches="tight")


def _compute_violations(
    metrics_df: pd.DataFrame,
    clients_df: pd.DataFrame,
    B_total: float,
    eps_min: float,
    eps_max: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - per-round violation rates dataframe
      - summary dataframe
    """
    tol = 1e-10

    # interval violations (from clients_round)
    rounds = sorted(set(int(r) for r in clients_df["round"].unique()))
    rows = []
    for r in rounds:
        sub = clients_df[clients_df["round"] == r]
        n = len(sub)
        if n == 0:
            rows.append({"round": r, "interval_violation_rate": 0.0})
            continue
        eps = pd.to_numeric(sub["epsilon_total"], errors="coerce").to_numpy()
        bad = np.sum((eps < eps_min - tol) | (eps > eps_max + tol)) if np.isfinite(eps_min) and np.isfinite(eps_max) else 0
        rows.append({"round": r, "interval_violation_rate": float(bad / max(n, 1))})

    interval_df = pd.DataFrame(rows).sort_values("round")

    # pool violations (from metrics_round)
    m = metrics_df.copy().sort_values("round")
    if np.isfinite(B_total):
        budget_used = pd.to_numeric(m["budget_used"], errors="coerce").fillna(0.0).to_numpy()
        pool_bad = (budget_used > (B_total + tol)).astype(float)
    else:
        pool_bad = np.zeros(len(m), dtype=float)

    pool_df = pd.DataFrame({"round": m["round"].astype(int), "pool_violation_rate": pool_bad})

    # merge
    merged = pd.merge(interval_df, pool_df, on="round", how="outer").fillna(0.0).sort_values("round")

    # summary (should be 0)
    summary = pd.DataFrame(
        [{
            "interval_violations_total": float(np.sum(merged["interval_violation_rate"] > 0)),
            "pool_violations_total": float(np.sum(merged["pool_violation_rate"] > 0)),
            "note": "理论上应为0；若非0，说明实现/日志存在问题",
        }]
    )
    return merged, summary


def _plot_violation_rates(run_dir: str, merged: pd.DataFrame, title: str):
    plt.figure()
    d = merged.sort_values("round")
    plt.plot(d["round"], d["interval_violation_rate"], label="Interval Violation Rate")
    plt.plot(d["round"], d["pool_violation_rate"], label="Pool Violation Rate")
    plt.xlabel("Round")
    plt.ylabel("Violation Rate")
    plt.title((title + " Constraint Violations (Should be 0)").strip())
    plt.legend()
    plt.savefig(os.path.join(run_dir, "dyn_violation_rates.png"), dpi=160, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="e.g., runs/mnist_dyn/mnist/rsbs_track")
    ap.add_argument("--title", type=str, default="MNIST", help="Plot title prefix")
    args = ap.parse_args()

    run_dir = args.run_dir
    metrics_df, clients_df = _load_csvs(run_dir)

    # Ensure needed columns exist (older logs won't crash)
    metrics_df = _ensure_cols(
        metrics_df,
        {
            "num_eps_min": 0.0,
            "num_eps_max": 0.0,
            "projection_triggered": 0.0,
            "budget_used": 0.0,
        },
    )
    clients_df = _ensure_cols(
        clients_df,
        {
            "epsilon_total": float("nan"),
            "epsilon_up": float("nan"),
            "epsilon_ft": float("nan"),
            "eta": float("nan"),
        },
    )

    B_total, eps_min, eps_max = _read_meta(run_dir)
    tracked = _read_tracked(run_dir)

    if not tracked:
        raise SystemExit(
            f"tracked_clients.json not found or empty in {run_dir}. "
            f"Please rerun dynamics experiment (experiments.run_dynamics) to generate it."
        )

    # 1) tracked evolution curves
    _plot_tracked_curves(run_dir, clients_df, tracked, args.title)

    # 2) boundary/projection + oscillation
    _plot_boundary_and_projection(run_dir, metrics_df, args.title)
    _plot_oscillation(run_dir, clients_df, tracked, args.title)

    # 3) constraint satisfaction (violations should be 0)
    merged, summary = _compute_violations(metrics_df, clients_df, B_total, eps_min, eps_max)
    _plot_violation_rates(run_dir, merged, args.title)
    merged.to_csv(os.path.join(run_dir, "dyn_violation_rates.csv"), index=False)
    summary.to_csv(os.path.join(run_dir, "dyn_violation_summary.csv"), index=False)


if __name__ == "__main__":
    main()
