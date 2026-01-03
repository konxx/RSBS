from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from experiments.common import load_yaml, build_world
from core.server import StrategyConfig


def auto_model_for_dataset(dataset: str) -> str:
    dataset = dataset.lower().strip()
    if dataset == "mnist":
        return "lenet"
    if dataset == "cifar10":
        return "simple_cnn"
    raise ValueError(f"Unsupported dataset: {dataset}")


def _entropy_from_counts(counts: np.ndarray) -> float:
    s = float(np.sum(counts))
    if s <= 0:
        return 0.0
    p = counts / s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-12)))


def _client_quality_score(client) -> float:
    """
    Reward proxy for selecting "high/low reward" clients:
      score = (#samples) * label_entropy
    Works when train_loader.dataset is a torch.utils.data.Subset from torchvision datasets.
    """
    ds = getattr(client.train_loader, "dataset", None)
    if ds is None:
        return 0.0

    # Subset case (expected)
    indices = getattr(ds, "indices", None)
    base = getattr(ds, "dataset", None)
    if indices is None or base is None:
        # fallback: only sample size
        try:
            return float(len(ds))
        except Exception:
            return 0.0

    # labels from torchvision dataset
    targets = getattr(base, "targets", None)
    if targets is None:
        # expensive fallback: iterate indices
        labels = []
        for i in indices:
            labels.append(int(base[i][1]))
        labels = np.asarray(labels, dtype=int)
    else:
        labels = np.asarray(targets, dtype=int)[np.asarray(indices, dtype=int)]

    num_classes = int(np.max(labels)) + 1 if labels.size > 0 else 10
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    ent = _entropy_from_counts(counts)
    score = float(len(indices)) * ent
    return score


def _pick_typical_clients(server) -> Tuple[List[int], Dict[int, Dict]]:
    """
    Pick 4 typical clients:
      - low sens + high score
      - low sens + low score
      - high sens + high score
      - high sens + low score

    Returns: (cids, meta_per_cid)
    """
    cids = list(server.clients.keys())
    sens = {cid: float(server.clients[cid].hetero.sensitivity) for cid in cids}
    scores = {cid: _client_quality_score(server.clients[cid]) for cid in cids}

    sens_values = sorted({sens[c] for c in cids})
    low_s = sens_values[0]
    high_s = sens_values[-1]

    low_group = [c for c in cids if abs(sens[c] - low_s) < 1e-12]
    high_group = [c for c in cids if abs(sens[c] - high_s) < 1e-12]

    def pick_two(group: List[int]) -> Tuple[int, int]:
        if not group:
            # fallback: use all
            group = cids
        group_sorted = sorted(group, key=lambda x: scores[x])
        low = group_sorted[0]
        high = group_sorted[-1]
        if low == high and len(group_sorted) >= 2:
            low = group_sorted[0]
            high = group_sorted[-1]
        return high, low  # (high_score, low_score)

    low_high, low_low = pick_two(low_group)
    high_high, high_low = pick_two(high_group)

    picked = []
    for cid in [low_high, low_low, high_high, high_low]:
        if cid not in picked:
            picked.append(cid)

    # if duplicates reduced count, fill from remaining by score extremes
    if len(picked) < 4:
        remaining = [c for c in cids if c not in picked]
        remaining_sorted = sorted(remaining, key=lambda x: scores[x])
        while len(picked) < 4 and remaining_sorted:
            # alternate add high then low
            picked.append(remaining_sorted.pop(-1))
            if len(picked) < 4 and remaining_sorted:
                picked.append(remaining_sorted.pop(0))
        picked = picked[:4]

    # label roles
    role = {
        low_high: "low_sens_high_reward",
        low_low: "low_sens_low_reward",
        high_high: "high_sens_high_reward",
        high_low: "high_sens_low_reward",
    }
    meta = {}
    for cid in picked:
        meta[cid] = {
            "cid": int(cid),
            "sensitivity": float(sens[cid]),
            "score": float(scores[cid]),
            "role": role.get(cid, "extra"),
        }
    return picked, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    ap.add_argument("--model", type=str, default=None, choices=["lenet", "simple_cnn"])
    ap.add_argument("--out", type=str, default="runs/mnist_dyn")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--clients_per_round", type=int, default=10)
    args = ap.parse_args()

    base_cfg = load_yaml("config/base_config.yaml")
    hetero_cfg = load_yaml("config/hetero_config.yaml")
    priv_cfg = load_yaml("config/privacy_config.yaml")

    base_cfg["dataset"]["name"] = args.dataset
    base_cfg["model"]["name"] = args.model or auto_model_for_dataset(args.dataset)

    base_cfg["federated"]["rounds"] = int(args.rounds)
    base_cfg["federated"]["clients_per_round"] = int(args.clients_per_round)

    out_dir = os.path.join(args.out, args.dataset, "rsbs_track")
    os.makedirs(out_dir, exist_ok=True)

    server, _ = build_world(base_cfg, hetero_cfg, priv_cfg, out_dir=out_dir)

    # pick 4 typical clients based on sensitivity and reward proxy score
    track, track_meta = _pick_typical_clients(server)

    # persist selection for visualization
    with open(os.path.join(out_dir, "tracked_clients.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"dataset": args.dataset, "clients": track_meta, "track_list": track},
            f,
            ensure_ascii=False,
            indent=2,
        )

    st = StrategyConfig(name="rsbs")

    # IMPORTANT: force include these clients each round so epsilon/eta curves are continuous
    server.run(
        strategy=st,
        rounds=int(args.rounds),
        clients_per_round=int(args.clients_per_round),
        track_clients=track,
        force_include_clients=track,
    )


if __name__ == "__main__":
    main()
