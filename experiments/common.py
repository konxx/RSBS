from __future__ import annotations

import os
import json
import random
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from data import dirichlet_partition_indices, get_torchvision_dataset, make_dataloaders_for_clients
from models import build_model
from core.client import FederatedClient, ClientHeterogeneity
from core.server import FederatedServer, StrategyConfig
from utils.logger import ExperimentLogger


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_device(use_cuda: bool) -> torch.device:
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_world(
    base_cfg: Dict,
    hetero_cfg: Dict,
    priv_cfg: Dict,
    out_dir: str,
    enable_projection: bool = True,
    enable_two_stage: bool = True,
) -> Tuple[FederatedServer, Dict]:
    seed = int(base_cfg.get("seed", 42))
    set_all_seeds(seed)

    device = make_device(bool(base_cfg["device"]["use_cuda"]))

    ds_name = base_cfg["dataset"]["name"]
    ds_root = base_cfg["dataset"]["root"]
    bundle = get_torchvision_dataset(ds_name, ds_root)

    num_clients = int(base_cfg["federated"]["num_clients"])
    alpha = float(hetero_cfg["non_iid"]["dirichlet_alpha"])

    # labels extraction (torchvision datasets expose targets)
    if hasattr(bundle.train, "targets"):
        labels = np.array(bundle.train.targets)
    else:
        # fallback
        labels = np.array([bundle.train[i][1] for i in range(len(bundle.train))])

    client_indices = dirichlet_partition_indices(
        labels=labels,
        num_clients=num_clients,
        alpha=alpha,
        num_classes=bundle.num_classes,
        seed=seed,
    )

    batch_size = int(base_cfg["federated"]["batch_size"])
    client_loaders, test_loader = make_dataloaders_for_clients(
        bundle.train, bundle.test, client_indices, batch_size=batch_size
    )

    # sensitivity heterogeneity: levels with ratio
    levels = list(map(float, hetero_cfg["sensitivity"]["levels"]))
    ratio = list(map(int, hetero_cfg["sensitivity"]["ratio"]))
    sens_list = []
    for lv, r in zip(levels, ratio):
        sens_list.extend([lv] * r)
    # pad/trim to num_clients
    while len(sens_list) < num_clients:
        sens_list.append(levels[-1])
    sens_list = sens_list[:num_clients]
    random.shuffle(sens_list)

    nu_lo, nu_hi = map(float, hetero_cfg["resources"]["nu_range"])
    om_lo, om_hi = map(float, hetero_cfg["resources"]["omega_range"])

    clients = {}
    for cid in range(num_clients):
        hetero = ClientHeterogeneity(
            sensitivity=sens_list[cid],
            nu=float(np.random.uniform(nu_lo, nu_hi)),
            omega=float(np.random.uniform(om_lo, om_hi)),
        )
        # Create eval_loader from client's train data for fairness metrics
        eval_loader = DataLoader(
            client_loaders[cid].dataset,
            batch_size=256,
            shuffle=False,
            drop_last=False,
        )
        clients[cid] = FederatedClient(
            cid=cid,
            train_loader=client_loaders[cid],
            hetero=hetero,
            device=device,
            eval_loader=eval_loader,
        )

    model = build_model(base_cfg["model"]["name"], num_classes=bundle.num_classes)

    dp_delta = float(priv_cfg["dp"]["delta"])
    clip_norm = float(priv_cfg["dp"]["clip_norm"])

    B_total = float(priv_cfg["budget"]["B_total"])
    eps_min = float(priv_cfg["budget"]["epsilon_min"])
    eps_max = float(priv_cfg["budget"]["epsilon_max"])

    rsbs_params = dict(priv_cfg.get("rsbs", {}))

    train_params = dict(
        lr=float(base_cfg["federated"]["lr"]),
        momentum=float(base_cfg["federated"]["momentum"]),
        weight_decay=float(base_cfg["federated"]["weight_decay"]),
        local_epochs_up=int(base_cfg["federated"]["local_epochs_up"]),
        local_epochs_ft=int(base_cfg["federated"]["local_epochs_ft"]),
    )

    meta = dict(
        base_config=base_cfg,
        hetero_config=hetero_cfg,
        privacy_config=priv_cfg,
        out_dir=out_dir,
        enable_projection=enable_projection,
        enable_two_stage=enable_two_stage,
        device=str(device),
    )
    logger = ExperimentLogger(out_dir=out_dir, meta=meta)

    server = FederatedServer(
        model=model,
        clients=clients,
        test_loader=test_loader,
        device=device,
        logger=logger,
        dp_delta=dp_delta,
        clip_norm=clip_norm,
        B_total=B_total,
        epsilon_min=eps_min,
        epsilon_max=eps_max,
        rsbs_params=rsbs_params,
        base_train_params=train_params,
        seed=seed,
        enable_projection=enable_projection,
        enable_two_stage=enable_two_stage,
    )
    return server, meta
