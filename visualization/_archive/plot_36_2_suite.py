from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


STRATEGY_ORDER = ["fixed", "linear_decay", "heuristic", "rsbs"]
STRATEGY_DISPLAY = {
    "fixed": "Fixed",
    "linear_decay": "Linear Decay",
    "heuristic": "Heuristic",
    "rsbs": "RS-BS",
}


def _find_datasets(root_dir: str) -> List[str]:
    if not os.path.isdir(root_dir):
        return []
    ds = []
    for name in os.listdir(root_dir):
        p = os.path.join(root_dir, name)
        if os.path.isdir(p):
            # dataset folder should contain strategy folders
            if any(os.path.isdir(os.path.join(p, st)) for st in STRATEGY_ORDER):
                ds.append(name)
    return sorted(ds)


def _read_metrics(strategy_dir: str) -> pd.DataFrame:
    path = os.path.join(strategy_dir, "metrics_round.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"metrics_round.csv not found: {path}")
    df = pd.read_csv(path)
    return df


def _ensure_cols(df: pd.DataFrame, cols_defaults: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for c, d in cols_defaults.items():
        if c not in out.columns:
            out[c] = d
    return out


def _compute_summary(df: pd.DataFrame, ignore_round0: bool = True) -> Dict[str, float]:
    d = df.copy()
    if ignore_round0 and (d["round"] == 0).any():
        d = d[d["round"] > 0]

    def safe_mean(x):
        x = pd.to_numeric(x, errors="coerce")
        return float(x.mean()) if len(x) else float("nan")

    def safe_last(x):
        if len(d) == 0:
            return float("nan")
        return float(pd.to_numeric(d.sort_values("round")[x].iloc[-1], errors="coerce"))

    return {
        "mean_acc": safe_mean(d["global_acc"]),
        "mean_loss": safe_mean(d["global_loss"]),
        "last_acc": safe_last("global_acc"),
        "last_loss": safe_last("global_loss"),
        "mean_pool_usage": safe_mean(d["pool_usage"]),
        "proj_rate": safe_mean(d["projection_triggered"]),
        "mean_num_eps_min": safe_mean(d["num_eps_min"]),
        "mean_num_eps_max": safe_mean(d["num_eps_max"]),
    }


def _plot_global_utility(metrics_by_strategy: Dict[str, pd.DataFrame], out_dir: str, title_prefix: str):
    # Accuracy
    plt.figure()
    for st in STRATEGY_ORDER:
        if st not in metrics_by_strategy:
            continue
        df = metrics_by_strategy[st].sort_values("round")
        plt.plot(df["round"], df["global_acc"], label=STRATEGY_DISPLAY.get(st, st))
    plt.xlabel("Round")
    plt.ylabel("Test Accuracy")
    plt.title((title_prefix + " Accuracy vs Round").strip())
    plt.legend()
    plt.savefig(os.path.join(out_dir, "global_utility_accuracy.png"), dpi=160, bbox_inches="tight")

    # Loss
    plt.figure()
    for st in STRATEGY_ORDER:
        if st not in metrics_by_strategy:
            continue
        df = metrics_by_strategy[st].sort_values("round")
        plt.plot(df["round"], df["global_loss"], label=STRATEGY_DISPLAY.get(st, st))
    plt.xlabel("Round")
    plt.ylabel("Test Loss")
    plt.title((title_prefix + " Loss vs Round").strip())
    plt.legend()
    plt.savefig(os.path.join(out_dir, "global_utility_loss.png"), dpi=160, bbox_inches="tight")


def _plot_privacy_constraints(metrics_by_strategy: Dict[str, pd.DataFrame], out_dir: str, title_prefix: str):
    # Pool usage U_t
    plt.figure()
    for st in STRATEGY_ORDER:
        if st not in metrics_by_strategy:
            continue
        df = metrics_by_strategy[st].sort_values("round")
        if df["pool_usage"].isna().all():
            # fallback: plot budget_used if pool_usage missing
            plt.plot(df["round"], df["budget_used"], label=STRATEGY_DISPLAY.get(st, st))
            ylab = "Budget Used (fallback)"
        else:
            plt.plot(df["round"], df["pool_usage"], label=STRATEGY_DISPLAY.get(st, st))
            ylab = "Pool Usage $U_t$"
    plt.xlabel("Round")
    plt.ylabel(ylab)
    plt.title((title_prefix + " Budget Pool Usage vs Round").strip())
    plt.legend()
    plt.savefig(os.path.join(out_dir, "privacy_pool_usage.png"), dpi=160, bbox_inches="tight")

    # Boundary hits (min/max) — plot each strategy as two lines (min & max)
    plt.figure()
    for st in STRATEGY_ORDER:
        if st not in metrics_by_strategy:
            continue
        df = metrics_by_strategy[st].sort_values("round")
        if df["num_eps_min"].isna().all() and df["num_eps_max"].isna().all():
            continue
        plt.plot(df["round"], df["num_eps_min"], label=f"{STRATEGY_DISPLAY.get(st, st)} @ eps_min")
        plt.plot(df["round"], df["num_eps_max"], label=f"{STRATEGY_DISPLAY.get(st, st)} @ eps_max")
    plt.xlabel("Round")
    plt.ylabel("#Clients on Boundary")
    plt.title((title_prefix + " Boundary Hits (eps_min/eps_max) vs Round").strip())
    plt.legend()
    plt.savefig(os.path.join(out_dir, "privacy_boundary_hits.png"), dpi=160, bbox_inches="tight")

    # Projection trigger cumulative count
    plt.figure()
    for st in STRATEGY_ORDER:
        if st not in metrics_by_strategy:
            continue
        df = metrics_by_strategy[st].sort_values("round")
        if df["projection_triggered"].isna().all():
            continue
        trig = pd.to_numeric(df["projection_triggered"], errors="coerce").fillna(0.0).to_numpy()
        csum = np.cumsum(trig)
        plt.plot(df["round"], csum, label=STRATEGY_DISPLAY.get(st, st))
    plt.xlabel("Round")
    plt.ylabel("Cumulative Projection Triggers")
    plt.title((title_prefix + " Projection Trigger (Cumsum) vs Round").strip())
    plt.legend()
    plt.savefig(os.path.join(out_dir, "privacy_projection_cumsum.png"), dpi=160, bbox_inches="tight")


def run_one_dataset(root_dir: str, dataset: str, ignore_round0: bool):
    ds_dir = os.path.join(root_dir, dataset)
    os.makedirs(ds_dir, exist_ok=True)

    metrics_by_strategy: Dict[str, pd.DataFrame] = {}
    summary_rows: List[Dict[str, object]] = []

    cols_defaults = {
        "budget_used": float("nan"),
        "pool_usage": float("nan"),
        "num_eps_min": float("nan"),
        "num_eps_max": float("nan"),
        "projection_triggered": float("nan"),
        "global_acc": float("nan"),
        "global_loss": float("nan"),
    }

    for st in STRATEGY_ORDER:
        st_dir = os.path.join(ds_dir, st)
        if not os.path.isdir(st_dir):
            continue
        df = _read_metrics(st_dir)
        df = _ensure_cols(df, cols_defaults)

        # If pool_usage missing but budget_used exists, keep NaN (we don't know B_total reliably here).
        metrics_by_strategy[st] = df

        summ = _compute_summary(df, ignore_round0=ignore_round0)
        summary_rows.append(
            {
                "dataset": dataset,
                "strategy": st,
                "strategy_display": STRATEGY_DISPLAY.get(st, st),
                **summ,
            }
        )

    if not metrics_by_strategy:
        raise SystemExit(f"No strategy folders found under: {ds_dir}")

    title_prefix = dataset.upper()

    _plot_global_utility(metrics_by_strategy, out_dir=ds_dir, title_prefix=title_prefix)
    _plot_privacy_constraints(metrics_by_strategy, out_dir=ds_dir, title_prefix=title_prefix)

    summary_df = pd.DataFrame(summary_rows).sort_values(["dataset", "strategy_display"])
    summary_df.to_csv(os.path.join(ds_dir, "summary_36_2.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Comparison root dir, e.g., runs/cifar_comp or runs/mnist_comp",
    )
    ap.add_argument(
        "--ignore_round0",
        action="store_true",
        help="Ignore round=0 when computing mean metrics (recommended).",
    )
    args = ap.parse_args()

    root_dir = args.root_dir
    datasets = _find_datasets(root_dir)
    if not datasets:
        raise SystemExit(f"No dataset folders found under: {root_dir}")

    for ds in datasets:
        run_one_dataset(root_dir=root_dir, dataset=ds, ignore_round0=bool(args.ignore_round0))


if __name__ == "__main__":
    main()

