from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from .dp_mechanisms import DPGaussianMechanism, ConservativeRDPAccountant, epsilon_to_noise_multiplier, gaussian_mechanism_for_scalar


@dataclass
class ClientHeterogeneity:
    sensitivity: float  # s_k in {0.2, 0.5, 0.8}
    nu: float           # compute throughput factor
    omega: float        # bandwidth factor


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    correct, total = 0, 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce(logits, y)
            loss_sum += float(loss.item()) * y.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.size(0))
    acc = correct / max(total, 1)
    avg_loss = loss_sum / max(total, 1)
    return acc, avg_loss


class FederatedClient:
    """
    Two-stage training:
      - stage 1 (upload): DP-SGD using epsilon_up; produces update for server
      - stage 2 (fine-tune): DP-SGD using epsilon_ft; improves local model only (not uploaded)
    """

    def __init__(
        self,
        cid: int,
        train_loader: DataLoader,
        hetero: ClientHeterogeneity,
        device: torch.device,
        eval_loader: Optional[DataLoader] = None,
    ):
        self.cid = int(cid)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.hetero = hetero
        self.device = device

        # RS-BS signals
        self.reward_ema = 0.0  # for RS-BS (legacy)
        self.rho_ema = 0.0     # consistency signal EMA
        self.g_ema = 0.0       # gradient norm signal EMA
        self.m_ema = 0.0       # loss proxy signal EMA
        self.c_ema = 0.0       # cost signal EMA
        
        # For loss proxy reporting
        self.prev_loss = None  # previous round loss for Δℓ calculation

    def local_train_two_stage(
        self,
        global_model: nn.Module,
        epsilon_total: float,
        eta: float,
        dp_delta: float,
        clip_norm: float,
        lr: float,
        momentum: float,
        weight_decay: float,
        local_epochs_up: int,
        local_epochs_ft: int,
        global_update_norm: Optional[float] = None,
        loss_proxy_params: Optional[Dict] = None,
    ) -> Dict:
        """
        Returns dict with:
          - update_state_dict: model delta after stage1
          - num_samples
          - raw_reward: legacy reward proxy
          - epsilon_up, epsilon_ft, eta
          - cost: resource cost proxy
          - update_norm: ||Δθ_k||_2
          - loss_proxy: DP-protected Δℓ for RS-BS
          - rho: consistency signal (cosine similarity with global update)
          - g: gradient norm signal
        """
        model = copy.deepcopy(global_model).to(self.device)
        model.train()

        epsilon_total = float(max(epsilon_total, 1e-12))
        eta = float(min(max(eta, 0.0), 1.0))
        epsilon_up = epsilon_total * eta
        epsilon_ft = epsilon_total * (1.0 - eta)

        # Calculate base loss for loss proxy
        ce = nn.CrossEntropyLoss()
        base_loss = None
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                base_loss = float(ce(model(x), y).item())
            break
        if base_loss is None:
            base_loss = 0.0

        # ---- Stage 1: upload DP-SGD
        steps_up = max(local_epochs_up, 1) * max(len(self.train_loader), 1)
        sigma_up = epsilon_to_noise_multiplier(epsilon_up, dp_delta, steps_up) if epsilon_up > 0 else 1e6
        mech_up = DPGaussianMechanism(clip_norm=clip_norm, noise_multiplier=sigma_up)
        acc_up = ConservativeRDPAccountant(delta=dp_delta)

        opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        for _ in range(local_epochs_up):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = ce(logits, y)
                loss.backward()
                mech_up.clip_and_noise_(model.parameters())
                opt.step()
                acc_up.step(sigma_up)

        # delta (stage1) for server aggregation
        update_state_dict = {}
        with torch.no_grad():
            gsd = global_model.state_dict()
            msd = model.state_dict()
            for k in gsd.keys():
                update_state_dict[k] = (msd[k] - gsd[k]).detach().cpu()

        # Calculate update norm (g signal)
        update_norm = 0.0
        for v in update_state_dict.values():
            update_norm += torch.sum(v ** 2).item()
        update_norm = float(np.sqrt(max(update_norm, 1e-12)))

        # ---- Stage 2: fine-tune (not uploaded)
        if local_epochs_ft > 0 and epsilon_ft > 0:
            steps_ft = local_epochs_ft * max(len(self.train_loader), 1)
            sigma_ft = epsilon_to_noise_multiplier(epsilon_ft, dp_delta, steps_ft)
            mech_ft = DPGaussianMechanism(clip_norm=clip_norm, noise_multiplier=sigma_ft)
            opt2 = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

            for _ in range(local_epochs_ft):
                for x, y in self.train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    opt2.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = ce(logits, y)
                    loss.backward()
                    mech_ft.clip_and_noise_(model.parameters())
                    opt2.step()

        # Calculate post-loss for loss proxy
        post_loss = None
        model.eval()
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                post_loss = float(ce(model(x), y).item())
            break
        if post_loss is None:
            post_loss = base_loss

        # Calculate loss proxy Δℓ = base_loss - post_loss
        delta_loss = base_loss - post_loss
        
        # Apply DP protection to loss proxy if parameters provided
        loss_proxy = delta_loss
        if loss_proxy_params is not None:
            C_l = loss_proxy_params.get("C_l", 1.0)  # truncation threshold
            epsilon_l = loss_proxy_params.get("epsilon_l", 1.0)
            delta_l = loss_proxy_params.get("delta_l", 1e-5)
            
            # Truncate
            delta_loss_trunc = max(min(delta_loss, C_l), -C_l)
            # Sensitivity: 2C_l (difference between two adjacent datasets)
            sensitivity = 2.0 * C_l
            
            # Apply Gaussian mechanism
            loss_proxy = gaussian_mechanism_for_scalar(
                delta_loss_trunc, epsilon_l, delta_l, sensitivity
            )

        # Calculate cost proxy
        cost = (1.0 / max(self.hetero.nu, 1e-6)) + (1.0 / max(self.hetero.omega, 1e-6))

        num_samples = len(self.train_loader.dataset)

        # Legacy reward proxy (for backward compatibility)
        raw_reward = max(0.0, delta_loss)

        # Store current loss for next round's Δℓ calculation
        self.prev_loss = post_loss

        # Update EMA signals for RS-BS (fix for CSV zero values issue)
        alpha = 0.8  # EMA smoothing coefficient
        self.g_ema = alpha * self.g_ema + (1.0 - alpha) * update_norm
        self.m_ema = alpha * self.m_ema + (1.0 - alpha) * max(delta_loss, 0.0)
        self.c_ema = alpha * self.c_ema + (1.0 - alpha) * cost
        
        # Consistency signal: use normalized update norm as proxy
        # Higher norm relative to history indicates higher consistency
        if self.g_ema > 1e-8:
            rho = min(update_norm / (self.g_ema + 1e-8), 2.0) / 2.0
        else:
            rho = 0.5
        self.rho_ema = alpha * self.rho_ema + (1.0 - alpha) * rho

        return {
            "cid": self.cid,
            "update_state_dict": update_state_dict,
            "num_samples": num_samples,
            "raw_reward": raw_reward,
            "epsilon_total": epsilon_total,
            "epsilon_up": float(epsilon_up),
            "epsilon_ft": float(epsilon_ft),
            "eta": float(eta),
            "cost": float(cost),
            "sensitivity": float(self.hetero.sensitivity),
            "nu": float(self.hetero.nu),
            "omega": float(self.hetero.omega),
            "update_norm": update_norm,
            "loss_proxy": float(loss_proxy),
            "base_loss": float(base_loss),
            "post_loss": float(post_loss),
        }
