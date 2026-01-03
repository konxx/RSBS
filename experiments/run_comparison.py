from __future__ import annotations

import argparse
import os

from experiments.common import load_yaml, build_world
from core.server import StrategyConfig


def auto_model_for_dataset(dataset: str) -> str:
    dataset = dataset.lower().strip()
    if dataset == "mnist":
        return "lenet"
    if dataset == "cifar10":
        return "simple_cnn"
    raise ValueError(f"Unsupported dataset: {dataset}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    ap.add_argument("--model", type=str, default=None, choices=["lenet", "simple_cnn"],
                    help="Optional override. If not set, auto-select based on dataset.")
    ap.add_argument("--out", type=str, default="runs/comparison")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--clients_per_round", type=int, default=None)
    args = ap.parse_args()

    base_cfg = load_yaml("config/base_config.yaml")
    hetero_cfg = load_yaml("config/hetero_config.yaml")
    priv_cfg = load_yaml("config/privacy_config.yaml")

    # Set dataset
    base_cfg["dataset"]["name"] = args.dataset

    # Auto-select correct model for dataset unless user overrides
    base_cfg["model"]["name"] = args.model or auto_model_for_dataset(args.dataset)

    if args.rounds is not None:
        base_cfg["federated"]["rounds"] = args.rounds
    if args.clients_per_round is not None:
        base_cfg["federated"]["clients_per_round"] = args.clients_per_round

    rounds = int(base_cfg["federated"]["rounds"])
    cpr = int(base_cfg["federated"]["clients_per_round"])

    # Run 4 strategies in 4 subfolders
    strategies = [
        StrategyConfig(name="fixed", fixed_epsilon=2.0),
        StrategyConfig(name="linear_decay", linear_start=4.0, linear_end=1.0),
        StrategyConfig(name="heuristic"),
        StrategyConfig(name="rsbs"),
    ]

    for st in strategies:
        out_dir = os.path.join(args.out, args.dataset, st.name)
        server, _ = build_world(base_cfg, hetero_cfg, priv_cfg, out_dir=out_dir)
        server.run(strategy=st, rounds=rounds, clients_per_round=cpr)


if __name__ == "__main__":
    main()
