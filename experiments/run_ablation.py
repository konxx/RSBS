from __future__ import annotations

import argparse
import os
import copy

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
    ap.add_argument("--out", type=str, default="runs/ablation")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--clients_per_round", type=int, default=10)
    args = ap.parse_args()

    base_cfg = load_yaml("config/base_config.yaml")
    hetero_cfg = load_yaml("config/hetero_config.yaml")
    priv_cfg = load_yaml("config/privacy_config.yaml")

    base_cfg["dataset"]["name"] = args.dataset
    base_cfg["model"]["name"] = args.model or auto_model_for_dataset(args.dataset)

    base_cfg["federated"]["rounds"] = args.rounds
    base_cfg["federated"]["clients_per_round"] = args.clients_per_round

    ablations = []

    # Full RS-BS
    ablations.append(("rsbs_full", dict(enable_projection=True, enable_two_stage=True, tweak={})))

    # beta = 0 (remove sensitivity term)
    ablations.append(("no_sensitivity_beta0", dict(enable_projection=True, enable_two_stage=True, tweak={"rsbs.beta": 0.0})))

    # lambda_c = 0 (remove cost penalty)
    ablations.append(("no_cost_lambda0", dict(enable_projection=True, enable_two_stage=True, tweak={"rsbs.lambda_c": 0.0})))

    # remove projection
    ablations.append(("no_projection", dict(enable_projection=False, enable_two_stage=True, tweak={})))

    # remove two-stage split
    ablations.append(("no_two_stage", dict(enable_projection=True, enable_two_stage=False, tweak={})))

    for name, cfg in ablations:
        priv_cfg2 = copy.deepcopy(priv_cfg)

        # apply tweaks like "rsbs.beta"
        for k, v in cfg["tweak"].items():
            parts = k.split(".")
            if len(parts) == 2 and parts[0] in priv_cfg2 and parts[1] in priv_cfg2[parts[0]]:
                priv_cfg2[parts[0]][parts[1]] = v

        out_dir = os.path.join(args.out, args.dataset, name)
        server, _ = build_world(
            base_cfg, hetero_cfg, priv_cfg2, out_dir=out_dir,
            enable_projection=cfg["enable_projection"],
            enable_two_stage=cfg["enable_two_stage"],
        )
        st = StrategyConfig(name="rsbs")
        server.run(strategy=st, rounds=args.rounds, clients_per_round=args.clients_per_round)


if __name__ == "__main__":
    main()

