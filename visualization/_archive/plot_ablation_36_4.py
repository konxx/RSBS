from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default


def _find_run_dirs(dataset_root: str) -> List[str]:
    """
    Find directories that contain metrics_round.csv under dataset_root (recursive).
    """
    run_dirs = []
    for r, _, files in os.walk(dataset_root):
        if "metrics_round.csv" in files:
            run_dirs.append(r)
    return sorted(run_dirs)


def _load_meta_constants(run_dir: str) -> Tuple[float, float, float]:
    meta_path = os.path.join(run_dir, "meta.json")
    if not os.path.exists(meta_path):
        return np.nan, np.nan, np.nan
    meta = _read_json(meta_path)
    priv = meta.get("privacy_config", {})
    budget = priv.get("budget", {})
    B_total = _safe_float(budget.get("B_total", np.nan))
    eps_min = _safe_float(budget.get("epsilon_min", np.nan))
    eps_max = _safe_float(budget.get("epsilon_max", np.nan))
    return B_total, eps_min, eps_max


def _ensure_cols(df: pd.DataFrame, defaults: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for c, d in defaults.items():
        if c not in out.columns:
            out[c] = d
    return out


def _tail_mean(series: pd.Series, tail_k: int) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return float("nan")
    if len(s) <= tail_k:
        return float(s.mean())
    return float(s.iloc[-tail_k:].mean())


def _compute_perf_table(metrics: pd.DataFrame, tail_k: int, ignore_round0: bool = True) -> Dict[str, float]:
    d = metrics.sort_values("round").copy()
    if ignore_round0 and (d["round"] == 0).any():
        d = d[d["round"] > 0]

    d_acc = pd.to_numeric(d["global_acc"], errors="coerce")
    d_loss = pd.to_numeric(d["global_loss"], errors="coerce")

    out = {
        "mean_acc": float(d_acc.mean()) if len(d_acc) else float("nan"),
        "mean_loss": float(d_loss.mean()) if len(d_loss) else float("nan"),
        "tail_acc_mean": _tail_mean(d["global_acc"], tail_k=tail_k),
        "tail_loss_mean": _tail_mean(d["global_loss"], tail_k=tail_k),
        "final_acc": float(d_acc.iloc[-1]) if len(d_acc) else float("nan"),
        "final_loss": float(d_loss.iloc[-1]) if len(d_loss) else float("nan"),
    }

    # client distribution stats are only filled at final round (otherwise NaN)
    last = d.iloc[-1] if len(d) else None
    if last is not None:
        out.update(
            {
                "client_acc_mean": _safe_float(last.get("client_acc_mean", np.nan)),
                "client_acc_worst": _safe_float(last.get("client_acc_worst", np.nan)),
                "client_acc_p10": _safe_float(last.get("client_acc_p10", np.nan)),
                "client_acc_p50": _safe_float(last.get("client_acc_p50", np.nan)),
                "client_acc_p90": _safe_float(last.get("client_acc_p90", np.nan)),
            }
        )
    else:
        out.update(
            {
                "client_acc_mean": np.nan,
                "client_acc_worst": np.nan,
                "client_acc_p10": np.nan,
                "client_acc_p50": np.nan,
                "client_acc_p90": np.nan,
            }
        )

    return out


def _compute_privacy_table(metrics: pd.DataFrame, B_total: float, ignore_round0: bool = True) -> Dict[str, float]:
    d = metrics.sort_values("round").copy()
    if ignore_round0 and (d["round"] == 0).any():
        d = d[d["round"] > 0]

    budget_used = pd.to_numeric(d["budget_used"], errors="coerce").fillna(0.0).to_numpy()
    pool_usage = pd.to_numeric(d["pool_usage"], errors="coerce")

    # if pool_usage missing, derive if B_total known
    if pool_usage.isna().all() and np.isfinite(B_total) and B_total > 0:
        pool_usage = pd.Series(budget_used / B_total)

    proj = pd.to_numeric(d["projection_triggered"], errors="coerce").fillna(0.0).to_numpy()
    nmin = pd.to_numeric(d["num_eps_min"], errors="coerce").fillna(0.0).to_numpy()
    nmax = pd.to_numeric(d["num_eps_max"], errors="coerce").fillna(0.0).to_numpy()

    if np.isfinite(B_total) and B_total > 0:
        violations = (budget_used > (B_total + 1e-12)).astype(float)
        violation_rate = float(np.mean(violations)) if len(violations) else 0.0
        violation_rounds = float(np.sum(violations)) if len(violations) else 0.0
    else:
        violation_rate = float("nan")
        violation_rounds = float("nan")

    return {
        "mean_pool_usage": float(pool_usage.mean()) if len(pool_usage.dropna()) else float("nan"),
        "max_pool_usage": float(pool_usage.max()) if len(pool_usage.dropna()) else float("nan"),
        "pool_violation_rate": float(violation_rate),
        "pool_violation_rounds": float(violation_rounds),
        "projection_trigger_rate": float(np.mean(proj)) if len(proj) else float("nan"),
        "mean_num_eps_min": float(np.mean(nmin)) if len(nmin) else float("nan"),
        "mean_num_eps_max": float(np.mean(nmax)) if len(nmax) else float("nan"),
    }


def _pretty_name(run_dir: str, dataset_root: str) -> str:
    # use relative path as variant name
    rel = os.path.relpath(run_dir, dataset_root)
    return rel.replace("\\", "/")  # stable across OS


def _barplot(series: pd.Series, title: str, ylabel: str, out_path: str):
    plt.figure()
    plt.bar(series.index.astype(str), series.values)
    plt.xticks(rotation=30, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str, required=True, help=r"e.g., runs/mnist_abl")
    ap.add_argument("--dataset", type=str, default="mnist", help="dataset subfolder under root_dir")
    ap.add_argument("--tail_k", type=int, default=5, help="tail rounds for convergence metric")
    ap.add_argument("--ignore_round0", action="store_true", help="ignore round 0 in averages")
    args = ap.parse_args()

    dataset_root = os.path.join(args.root_dir, args.dataset)
    run_dirs = _find_run_dirs(dataset_root)
    if not run_dirs:
        raise SystemExit(f"No metrics_round.csv found under: {dataset_root}")

    perf_rows = []
    priv_rows = []

    for rd in run_dirs:
        metrics_path = os.path.join(rd, "metrics_round.csv")
        metrics = pd.read_csv(metrics_path)

        metrics = _ensure_cols(
            metrics,
            {
                "round": np.nan,
                "global_acc": np.nan,
                "global_loss": np.nan,
                "budget_used": 0.0,
                "pool_usage": np.nan,
                "num_eps_min": 0.0,
                "num_eps_max": 0.0,
                "projection_triggered": 0.0,
                "client_acc_mean": np.nan,
                "client_acc_worst": np.nan,
                "client_acc_p10": np.nan,
                "client_acc_p50": np.nan,
                "client_acc_p90": np.nan,
            },
        )

        B_total, eps_min, eps_max = _load_meta_constants(rd)

        name = _pretty_name(rd, dataset_root)
        perf = _compute_perf_table(metrics, tail_k=int(args.tail_k), ignore_round0=bool(args.ignore_round0))
        priv = _compute_privacy_table(metrics, B_total=B_total, ignore_round0=bool(args.ignore_round0))

        perf_rows.append({"variant": name, **perf})
        priv_rows.append(
            {
                "variant": name,
                "B_total": B_total,
                "eps_min": eps_min,
                "eps_max": eps_max,
                **priv,
            }
        )

    perf_df = pd.DataFrame(perf_rows).sort_values("variant")
    priv_df = pd.DataFrame(priv_rows).sort_values("variant")

    out1 = os.path.join(dataset_root, "ablation_final_perf_tail.csv")
    out2 = os.path.join(dataset_root, "ablation_privacy_constraints.csv")
    perf_df.to_csv(out1, index=False)
    priv_df.to_csv(out2, index=False)

    # Optional figures (helpful for thesis)
    # 1) pool violation rate (highlight w/o projection)
    _barplot(
        priv_df.set_index("variant")["pool_violation_rate"],
        title="Ablation: Pool Violation Rate (budget_used > B_total)",
        ylabel="Violation Rate",
        out_path=os.path.join(dataset_root, "ablation_pool_violation_rate.png"),
    )

    # 2) pool usage mean/max
    plt.figure()
    x = np.arange(len(priv_df))
    plt.bar(priv_df["variant"], priv_df["mean_pool_usage"], label="mean_pool_usage")
    plt.bar(priv_df["variant"], priv_df["max_pool_usage"], label="max_pool_usage")
    plt.xticks(rotation=30, ha="right")
    plt.title("Ablation: Pool Usage (Mean/Max)")
    plt.ylabel("Pool Usage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_root, "ablation_pool_usage_mean_max.png"), dpi=160, bbox_inches="tight")

    # 3) boundary hits (mean)
    plt.figure()
    plt.bar(priv_df["variant"], priv_df["mean_num_eps_min"], label="mean # at eps_min")
    plt.bar(priv_df["variant"], priv_df["mean_num_eps_max"], label="mean # at eps_max")
    plt.xticks(rotation=30, ha="right")
    plt.title("Ablation: Boundary Hits (Mean)")
    plt.ylabel("#Clients")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_root, "ablation_boundary_hits_mean.png"), dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    main()
