"""
plot_paper_figures.py
=====================
统一生成论文/毕设所需的 6 张关键图片：

1. fig_noniid_heterogeneity.png   — Non-IID/异构设置示意
2. fig_comparison_accuracy.png    — 全局 Acc vs Round
3. fig_comparison_fairness.png    — 最差客户端/方差 vs Round
4. fig_pool_usage.png             — 预算池使用率曲线
5. fig_client_budget_evolution.png— 代表客户端预算演化曲线
6. fig_ablation_comparison.png    — 消融对比柱状图（多子图）

Usage:
    python -m visualization.plot_paper_figures \\
        --comp_dir runs/mnist_comp \\
        --dyn_dir runs/mnist_dyn/mnist/rsbs_track \\
        --abl_dir runs/mnist_abl \\
        --dataset mnist \\
        --out_dir figures/
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 策略顺序与显示名称
STRATEGY_ORDER = ["fixed", "linear_decay", "heuristic", "rsbs"]
STRATEGY_DISPLAY = {
    "fixed": "Fixed",
    "linear_decay": "Linear Decay",
    "heuristic": "Heuristic",
    "rsbs": "RS-BS (Ours)",
}
STRATEGY_COLORS = {
    "fixed": "#7f7f7f",
    "linear_decay": "#ff7f0e",
    "heuristic": "#2ca02c",
    "rsbs": "#1f77b4",
}

# ============================================================================
# Helper functions
# ============================================================================

def _read_metrics(strategy_dir: str) -> pd.DataFrame:
    path = os.path.join(strategy_dir, "metrics_round.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"metrics_round.csv not found: {path}")
    return pd.read_csv(path)


def _ensure_cols(df: pd.DataFrame, defaults: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for c, d in defaults.items():
        if c not in out.columns:
            out[c] = d
    return out


def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _tail_mean(series: pd.Series, tail_k: int = 5) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return float("nan")
    if len(s) <= tail_k:
        return float(s.mean())
    return float(s.iloc[-tail_k:].mean())


# ============================================================================
# Figure 1: Non-IID / Heterogeneity Diagram
# ============================================================================

def plot_noniid_heterogeneity(config_dir: str, out_dir: str, num_clients: int = 10, num_classes: int = 10):
    """
    生成 Non-IID 数据分布及客户端异构性示意图。
    使用 Dirichlet 分布模拟标签分布，并可视化敏感度和资源异构性。
    """
    # Read config
    hetero_path = os.path.join(config_dir, "hetero_config.yaml")
    if os.path.exists(hetero_path):
        import yaml
        with open(hetero_path, "r", encoding="utf-8") as f:
            hetero = yaml.safe_load(f)
    else:
        hetero = {}

    alpha = hetero.get("non_iid", {}).get("dirichlet_alpha", 0.5)
    sens_levels = hetero.get("sensitivity", {}).get("levels", [0.2, 0.5, 0.8])
    sens_ratio = hetero.get("sensitivity", {}).get("ratio", [5, 3, 2])

    # Simulate Dirichlet label distribution
    np.random.seed(42)
    label_dist = np.random.dirichlet([alpha] * num_classes, size=num_clients)

    # Assign sensitivity levels based on ratio
    total_ratio = sum(sens_ratio)
    sens_counts = [int(num_clients * r / total_ratio) for r in sens_ratio]
    sens_counts[-1] = num_clients - sum(sens_counts[:-1])  # Fix rounding
    client_sensitivity = []
    for lvl, cnt in zip(sens_levels, sens_counts):
        client_sensitivity.extend([lvl] * cnt)
    np.random.shuffle(client_sensitivity)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Label distribution heatmap
    ax1 = axes[0]
    im = ax1.imshow(label_dist.T, aspect="auto", cmap="Blues")
    ax1.set_xlabel("Client ID")
    ax1.set_ylabel("Class Label")
    ax1.set_title(f"Non-IID Label Distribution (α={alpha})")
    ax1.set_xticks(range(num_clients))
    ax1.set_yticks(range(num_classes))
    plt.colorbar(im, ax=ax1, label="Proportion")

    # Subplot 2: Stacked bar chart for sensitivity + sample count
    ax2 = axes[1]
    sample_counts = np.random.randint(100, 500, size=num_clients)
    colors = plt.cm.Reds(np.array(client_sensitivity) / max(sens_levels))
    bars = ax2.bar(range(num_clients), sample_counts, color=colors, edgecolor="black")
    ax2.set_xlabel("Client ID")
    ax2.set_ylabel("Sample Count")
    ax2.set_title("Client Heterogeneity (bar color = sensitivity)")
    ax2.set_xticks(range(num_clients))
    # Add colorbar for sensitivity
    sm = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(vmin=min(sens_levels), vmax=max(sens_levels)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax2, label="Sensitivity")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_noniid_heterogeneity.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Saved: {out_path}")


# ============================================================================
# Figure 2 & 3: Comparison Curves (Accuracy + Fairness)
# ============================================================================

def plot_comparison_curves(comp_dir: str, dataset: str, out_dir: str, ignore_round0: bool = True):
    """
    生成主对比曲线：
    - fig_comparison_accuracy.png: 全局准确率 vs Round
    - fig_comparison_fairness.png: 最差客户端准确率 + 方差 vs Round
    """
    ds_dir = os.path.join(comp_dir, dataset)
    metrics_by_strategy: Dict[str, pd.DataFrame] = {}

    defaults = {
        "global_acc": np.nan,
        "worst_client_acc": np.nan,
        "client_acc_var": np.nan,
        "fairness_var": np.nan,
    }

    for st in STRATEGY_ORDER:
        st_dir = os.path.join(ds_dir, st)
        if not os.path.isdir(st_dir):
            continue
        df = _read_metrics(st_dir)
        df = _ensure_cols(df, defaults)
        if ignore_round0 and (df["round"] == 0).any():
            df = df[df["round"] > 0]
        metrics_by_strategy[st] = df.sort_values("round")

    if not metrics_by_strategy:
        print(f"[!] No strategy data found in {ds_dir}")
        return

    # Figure 2: Accuracy curve
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for st in STRATEGY_ORDER:
        if st not in metrics_by_strategy:
            continue
        df = metrics_by_strategy[st]
        ax1.plot(df["round"], df["global_acc"], 
                 label=STRATEGY_DISPLAY.get(st, st),
                 color=STRATEGY_COLORS.get(st, None),
                 linewidth=2)
    ax1.set_xlabel("Communication Round", fontsize=12)
    ax1.set_ylabel("Global Test Accuracy", fontsize=12)
    ax1.set_title(f"{dataset.upper()}: Accuracy Comparison", fontsize=14)
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    out1 = os.path.join(out_dir, "fig_comparison_accuracy.png")
    fig1.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close(fig1)
    print(f"[✓] Saved: {out1}")

    # Figure 3: Fairness curve (fairness_var)
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    for st in STRATEGY_ORDER:
        if st not in metrics_by_strategy:
            continue
        df = metrics_by_strategy[st]
        # Use fairness_var which has data (not worst_client_acc/client_acc_var which are mostly NaN)
        y = df["fairness_var"].dropna()
        x = df.loc[y.index, "round"]
        ax2.plot(x, y,
                  label=STRATEGY_DISPLAY.get(st, st),
                  color=STRATEGY_COLORS.get(st, None),
                  linewidth=2)

    ax2.set_xlabel("Communication Round", fontsize=12)
    ax2.set_ylabel("Fairness Variance (Reward Proxy)", fontsize=12)
    ax2.set_title(f"{dataset.upper()}: Fairness Comparison", fontsize=14)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out2 = os.path.join(out_dir, "fig_comparison_fairness.png")
    fig2.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"[✓] Saved: {out2}")


# ============================================================================
# Figure 4: Pool Usage Curve
# ============================================================================

def plot_pool_usage(comp_dir: str, dataset: str, out_dir: str, ignore_round0: bool = True):
    """
    生成预算池使用率曲线。
    """
    ds_dir = os.path.join(comp_dir, dataset)
    metrics_by_strategy: Dict[str, pd.DataFrame] = {}

    defaults = {"pool_usage": np.nan, "budget_used": 0.0}

    for st in STRATEGY_ORDER:
        st_dir = os.path.join(ds_dir, st)
        if not os.path.isdir(st_dir):
            continue
        df = _read_metrics(st_dir)
        df = _ensure_cols(df, defaults)
        if ignore_round0 and (df["round"] == 0).any():
            df = df[df["round"] > 0]
        metrics_by_strategy[st] = df.sort_values("round")

    if not metrics_by_strategy:
        print(f"[!] No strategy data found in {ds_dir}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for st in STRATEGY_ORDER:
        if st not in metrics_by_strategy:
            continue
        df = metrics_by_strategy[st]
        y = df["pool_usage"] if not df["pool_usage"].isna().all() else df["budget_used"]
        ax.plot(df["round"], y,
                label=STRATEGY_DISPLAY.get(st, st),
                color=STRATEGY_COLORS.get(st, None),
                linewidth=2)

    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Budget Pool Usage ($U_t$)", fontsize=12)
    ax.set_title(f"{dataset.upper()}: Budget Pool Usage", fontsize=14)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1, label="Budget Limit")

    out_path = os.path.join(out_dir, "fig_pool_usage.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Saved: {out_path}")


# ============================================================================
# Figure 5: Client Budget Evolution
# ============================================================================

def plot_client_budget_evolution(dyn_dir: str, out_dir: str, title: str = ""):
    """
    生成代表客户端的预算演化曲线。
    优先从 tracked_clients.json 读取客户端列表。
    """
    clients_path = os.path.join(dyn_dir, "clients_round.csv")
    if not os.path.exists(clients_path):
        print(f"[!] clients_round.csv not found in {dyn_dir}")
        return

    df = pd.read_csv(clients_path)
    
    # Try to read tracked_clients.json first
    tracked_json_path = os.path.join(dyn_dir, "tracked_clients.json")
    tracked_cids = []
    tracked_labels = {}
    if os.path.exists(tracked_json_path):
        try:
            with open(tracked_json_path, "r", encoding="utf-8") as f:
                tracked_data = json.load(f)
            tracked_cids = [int(cid) for cid in tracked_data.get("track_list", [])]
            # Get role labels for legend
            clients_info = tracked_data.get("clients", {})
            for cid_str, info in clients_info.items():
                role = info.get("role", "")
                # Translate role to readable label
                role_map = {
                    "low_sens_high_reward": "Low-S High-R",
                    "low_sens_low_reward": "Low-S Low-R",
                    "high_sens_high_reward": "High-S High-R",
                    "high_sens_low_reward": "High-S Low-R",
                }
                tracked_labels[int(cid_str)] = role_map.get(role, f"Client {cid_str}")
        except Exception as e:
            print(f"[!] Failed to read tracked_clients.json: {e}")
    
    # Fallback: use tracked column or first 4 unique clients
    if not tracked_cids:
        tracked_df = df[df.get("tracked", 0) == 1]
        if not tracked_df.empty:
            tracked_cids = tracked_df["cid"].unique().tolist()
        else:
            tracked_cids = df["cid"].unique()[:4].tolist()
    
    if not tracked_cids:
        print(f"[!] No client data found in {dyn_dir}")
        return
    
    # Filter to tracked clients
    tracked = df[df["cid"].isin(tracked_cids)].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.tab10
    for i, cid in enumerate(tracked_cids):
        g = tracked[tracked["cid"] == cid].sort_values("round")
        if g.empty:
            continue
        label = tracked_labels.get(cid, f"Client {cid}")
        ax.plot(g["round"], g["epsilon_total"], 
                label=label,
                color=cmap(i % 10),
                linewidth=2)

    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel(r"Privacy Budget $\epsilon_k^t$", fontsize=12)
    ax.set_title(f"{title} Representative Clients: Budget Evolution".strip(), fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(out_dir, "fig_client_budget_evolution.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Saved: {out_path}")


# ============================================================================
# Figure 6: Ablation Bar Chart (Subplots)
# ============================================================================

def plot_ablation_bar(abl_dir: str, dataset: str, out_dir: str, tail_k: int = 5, ignore_round0: bool = True):
    """
    生成消融实验的多子图柱状图：
    - 子图1: 性能 (tail_acc_mean)
    - 子图2: 公平性 (worst_client_acc)
    - 子图3: 代价 (mean_pool_usage)
    """
    dataset_root = os.path.join(abl_dir, dataset)
    run_dirs = []
    for r, _, files in os.walk(dataset_root):
        if "metrics_round.csv" in files:
            run_dirs.append(r)
    run_dirs = sorted(run_dirs)

    if not run_dirs:
        print(f"[!] No ablation runs found in {dataset_root}")
        return

    results = []
    for rd in run_dirs:
        metrics = pd.read_csv(os.path.join(rd, "metrics_round.csv"))
        metrics = _ensure_cols(metrics, {
            "global_acc": np.nan,
            "worst_client_acc": np.nan,
            "pool_usage": np.nan,
            "budget_used": 0.0,
        })

        d = metrics.sort_values("round").copy()
        if ignore_round0 and (d["round"] == 0).any():
            d = d[d["round"] > 0]

        # Get variant name from folder
        variant = os.path.basename(rd)

        # Compute metrics
        tail_acc = _tail_mean(d["global_acc"], tail_k)
        # Worst client at final round
        last_row = d.iloc[-1] if len(d) > 0 else None
        worst_acc = _safe_float(last_row["worst_client_acc"]) if last_row is not None else np.nan
        # Pool usage mean
        pool_usage = d["pool_usage"] if not d["pool_usage"].isna().all() else d["budget_used"]
        mean_pool = float(pool_usage.mean()) if len(pool_usage) > 0 else np.nan

        results.append({
            "variant": variant,
            "tail_acc": tail_acc,
            "worst_acc": worst_acc,
            "mean_pool": mean_pool,
        })

    df = pd.DataFrame(results).sort_values("variant")

    # Create 3-subplot figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Subplot 1: Performance
    ax1 = axes[0]
    colors = ["#1f77b4" if "rsbs_full" in v else "#7f7f7f" for v in df["variant"]]
    ax1.bar(range(len(df)), df["tail_acc"], color=colors, edgecolor="black")
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df["variant"], rotation=30, ha="right")
    ax1.set_ylabel("Tail-K Accuracy", fontsize=11)
    ax1.set_title("(a) Performance", fontsize=12)
    ax1.grid(True, axis="y", alpha=0.3)

    # Subplot 2: Fairness
    ax2 = axes[1]
    ax2.bar(range(len(df)), df["worst_acc"], color=colors, edgecolor="black")
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels(df["variant"], rotation=30, ha="right")
    ax2.set_ylabel("Worst Client Accuracy", fontsize=11)
    ax2.set_title("(b) Fairness", fontsize=12)
    ax2.grid(True, axis="y", alpha=0.3)

    # Subplot 3: Cost
    ax3 = axes[2]
    ax3.bar(range(len(df)), df["mean_pool"], color=colors, edgecolor="black")
    ax3.set_xticks(range(len(df)))
    ax3.set_xticklabels(df["variant"], rotation=30, ha="right")
    ax3.set_ylabel("Mean Pool Usage", fontsize=11)
    ax3.set_title("(c) Budget Cost", fontsize=12)
    ax3.axhline(y=1.0, color="red", linestyle="--", linewidth=1)
    ax3.grid(True, axis="y", alpha=0.3)

    plt.suptitle(f"{dataset.upper()}: Ablation Study Comparison", fontsize=14, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "fig_ablation_comparison.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Saved: {out_path}")


# ============================================================================
# Main entry
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Generate paper/thesis figures for RS-BS project.")
    ap.add_argument("--comp_dir", type=str, default="runs/mnist_comp",
                    help="Comparison experiment root dir (e.g., runs/mnist_comp)")
    ap.add_argument("--dyn_dir", type=str, default="runs/mnist_dyn/mnist/rsbs_track",
                    help="Dynamics experiment run dir")
    ap.add_argument("--abl_dir", type=str, default="runs/mnist_abl",
                    help="Ablation experiment root dir")
    ap.add_argument("--config_dir", type=str, default="config",
                    help="Config directory for hetero_config.yaml")
    ap.add_argument("--dataset", type=str, default="mnist",
                    help="Dataset subfolder name")
    ap.add_argument("--out_dir", type=str, default="figures",
                    help="Output directory for figures")
    ap.add_argument("--ignore_round0", action="store_true", default=True,
                    help="Ignore round 0 in metrics")
    ap.add_argument("--tail_k", type=int, default=5,
                    help="Tail K rounds for convergence metrics")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("Generating paper figures...")
    print("=" * 60)

    # Figure 1: Non-IID / Heterogeneity
    print("\n[1/6] Generating Non-IID heterogeneity diagram...")
    try:
        plot_noniid_heterogeneity(args.config_dir, args.out_dir)
    except Exception as e:
        print(f"[!] Error: {e}")

    # Figure 2 & 3: Comparison curves
    print("\n[2-3/6] Generating comparison curves...")
    try:
        plot_comparison_curves(args.comp_dir, args.dataset, args.out_dir, args.ignore_round0)
    except Exception as e:
        print(f"[!] Error: {e}")

    # Figure 4: Pool usage
    print("\n[4/6] Generating pool usage curve...")
    try:
        plot_pool_usage(args.comp_dir, args.dataset, args.out_dir, args.ignore_round0)
    except Exception as e:
        print(f"[!] Error: {e}")

    # Figure 5: Client budget evolution
    print("\n[5/6] Generating client budget evolution curve...")
    try:
        plot_client_budget_evolution(args.dyn_dir, args.out_dir, args.dataset.upper())
    except Exception as e:
        print(f"[!] Error: {e}")

    # Figure 6: Ablation bar chart
    print("\n[6/6] Generating ablation bar chart...")
    try:
        plot_ablation_bar(args.abl_dir, args.dataset, args.out_dir, args.tail_k, args.ignore_round0)
    except Exception as e:
        print(f"[!] Error: {e}")

    print("\n" + "=" * 60)
    print(f"Done! Figures saved to: {args.out_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
