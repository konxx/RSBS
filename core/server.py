from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.metrics import compute_fairness_var
from utils.logger import ExperimentLogger

try:
    from core.client import evaluate as eval_on_loader  # type: ignore
except Exception:
    eval_on_loader = None  # type: ignore


@dataclass
class StrategyConfig:
    name: str
    fixed_epsilon: float = 2.0
    linear_start: float = 4.0
    linear_end: float = 1.0


class FederatedServer:
    def __init__(
        self,
        model: nn.Module,
        clients: Dict[int, object],
        test_loader,
        device: torch.device,
        logger: ExperimentLogger,
        dp_delta: float,
        clip_norm: float,
        B_total: float,
        epsilon_min: float,
        epsilon_max: float,
        rsbs_params: Dict,
        base_train_params: Dict,
        seed: int = 42,
        enable_projection: bool = True,
        enable_two_stage: bool = True,
    ):
        self.global_model = model.to(device)
        self.clients = clients
        self.test_loader = test_loader
        self.device = device
        self.logger = logger

        self.dp_delta = float(dp_delta)
        self.clip_norm = float(clip_norm)

        self.B_total = float(B_total)
        self.eps_min = float(epsilon_min)
        self.eps_max = float(epsilon_max)

        self.rsbs = dict(rsbs_params)
        self.train_params = dict(base_train_params)

        self.rng = random.Random(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.enable_projection = bool(enable_projection)
        self.enable_two_stage = bool(enable_two_stage)

        self.reward_ema: Dict[int, float] = {cid: 0.0 for cid in clients.keys()}
        self._last_projection_triggered: bool = False
        
        # 信号归一化的全局历史上下界统计（论文3.4.2节）
        # 为每类信号维护[L, U]对，使用递推更新：L ← min(L, x), U ← max(U, x)
        self.signal_bounds = {
            'rho': {'L': float('inf'), 'U': float('-inf')},  # 一致性信号
            'g': {'L': float('inf'), 'U': float('-inf')},    # 梯度范数信号
            'm': {'L': float('inf'), 'U': float('-inf')},    # 损失代理信号
            'c': {'L': float('inf'), 'U': float('-inf')},    # 代价信号
        }
        
        # 归一化参数
        self.norm_epsilon = float(self.rsbs.get("norm_epsilon", 1e-8))
        self.norm_clip_low = float(self.rsbs.get("norm_clip_low", 0.01))
        self.norm_clip_high = float(self.rsbs.get("norm_clip_high", 0.99))

    # ---------------- Evaluation ----------------

    def evaluate_global(self) -> Tuple[float, float]:
        self.global_model.eval()
        ce = nn.CrossEntropyLoss()
        correct, total = 0, 0
        loss_sum = 0.0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.global_model(x)
                loss = ce(logits, y)
                loss_sum += float(loss.item()) * y.size(0)
                pred = logits.argmax(dim=1)
                correct += int((pred == y).sum().item())
                total += int(y.size(0))
        return float(correct / max(total, 1)), float(loss_sum / max(total, 1))

    def _evaluate_model_on_loader(self, loader: DataLoader) -> Tuple[float, float]:
        """Fallback evaluator if core.client.evaluate is unavailable."""
        self.global_model.eval()
        ce = nn.CrossEntropyLoss()
        correct, total = 0, 0
        loss_sum = 0.0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.global_model(x)
                loss = ce(logits, y)
                loss_sum += float(loss.item()) * y.size(0)
                pred = logits.argmax(dim=1)
                correct += int((pred == y).sum().item())
                total += int(y.size(0))
        return float(correct / max(total, 1)), float(loss_sum / max(total, 1))

    def _get_client_eval_loader(self, client) -> Optional[DataLoader]:
        """
        Prefer client.eval_loader if exists.
        Else build a deterministic eval loader from client.train_loader.dataset.
        """
        ev = getattr(client, "eval_loader", None)
        if ev is not None:
            try:
                if hasattr(ev, "dataset") and len(ev.dataset) == 0:
                    ev = None
            except Exception:
                pass
        if ev is not None:
            return ev

        tr = getattr(client, "train_loader", None)
        if tr is None:
            return None
        ds = getattr(tr, "dataset", None)
        if ds is None:
            return None
        try:
            if len(ds) == 0:
                return None
        except Exception:
            pass

        # build eval loader (no shuffle) from local dataset
        return DataLoader(ds, batch_size=256, shuffle=False, drop_last=False)

    def _client_acc_distribution_stats(self) -> Dict[str, float]:
        """
        Evaluate global model on each client's local eval dataset and compute:
          mean, worst, p10, p50, p90
        Used for ablation table (3.6.4).
        """
        accs: List[float] = []
        self.global_model.eval()

        for _, client in self.clients.items():
            loader = self._get_client_eval_loader(client)
            if loader is None:
                continue
            if eval_on_loader is not None and getattr(client, "eval_loader", None) is not None:
                a, _ = eval_on_loader(self.global_model, loader, self.device)  # type: ignore
            else:
                a, _ = self._evaluate_model_on_loader(loader)
            accs.append(float(a))

        if len(accs) == 0:
            return {
                "client_acc_mean": float("nan"),
                "client_acc_worst": float("nan"),
                "client_acc_p10": float("nan"),
                "client_acc_p50": float("nan"),
                "client_acc_p90": float("nan"),
            }

        arr = np.asarray(accs, dtype=float)
        return {
            "client_acc_mean": float(np.mean(arr)),
            "client_acc_worst": float(np.min(arr)),
            "client_acc_p10": float(np.percentile(arr, 10)),
            "client_acc_p50": float(np.percentile(arr, 50)),
            "client_acc_p90": float(np.percentile(arr, 90)),
        }

    def _evaluate_clients_fairness(self) -> Tuple[float, float]:
        # keep optional fairness curve support (not required for your 3.6.4 tables)
        if eval_on_loader is None:
            return float("nan"), float("nan")
        accs: List[float] = []
        self.global_model.eval()
        for _, client in self.clients.items():
            eval_loader = getattr(client, "eval_loader", None)
            if eval_loader is None:
                continue
            try:
                if hasattr(eval_loader, "dataset") and len(eval_loader.dataset) == 0:
                    continue
            except Exception:
                pass
            a, _ = eval_on_loader(self.global_model, eval_loader, self.device)  # type: ignore
            accs.append(float(a))
        if len(accs) == 0:
            return float("nan"), float("nan")
        arr = np.asarray(accs, dtype=float)
        return float(np.var(arr)), float(np.min(arr))

    # ---------------- Client selection ----------------

    def sample_clients(self, clients_per_round: int) -> List[int]:
        all_cids = list(self.clients.keys())
        self.rng.shuffle(all_cids)
        return all_cids[:clients_per_round]

    def _select_clients_with_forced(
        self,
        clients_per_round: int,
        force_include_clients: Optional[List[int]],
    ) -> List[int]:
        force = [int(c) for c in (force_include_clients or []) if int(c) in self.clients]
        force = list(dict.fromkeys(force))  # unique keep order

        if clients_per_round <= 0:
            return []
        if len(force) >= clients_per_round:
            return force[:clients_per_round]

        remaining_need = clients_per_round - len(force)
        pool = [cid for cid in self.clients.keys() if cid not in force]
        self.rng.shuffle(pool)
        return force + pool[:remaining_need]

    # ---------------- Budget projection & stats ----------------

    def _project_budget_pool(self, eps: Dict[int, float]) -> Dict[int, float]:
        """
        实现带下界的预算池可行化映射（论文算法3-1步骤5）
        公式：ε_k^{t+1} = ε_min + (R / Σ_k u_k) * u_k
        其中 u_k = max(ε̂_k^{t+1} - ε_min, 0), R = B_total - K·ε_min
        """
        self._last_projection_triggered = False
        
        # 步骤1：区间裁剪 ε̂_k^{t+1} = clip(ε̃_k^{t+1}, ε_min, ε_max)
        clamped = {cid: float(min(max(v, self.eps_min), self.eps_max)) for cid, v in eps.items()}
        
        # 检查是否满足可行性前提：B_total ≥ K·ε_min
        K = len(clamped)
        if self.B_total < K * self.eps_min - 1e-12:
            # 如果不可行，返回裁剪后的值（但这种情况不应该发生）
            return clamped
        
        # 计算剩余可分配预算 R = B_total - K·ε_min
        R = self.B_total - K * self.eps_min
        
        # 计算增量 u_k = ε̂_k^{t+1} - ε_min
        u_k = {cid: max(v - self.eps_min, 0.0) for cid, v in clamped.items()}
        sum_u = sum(u_k.values())
        
        # 如果 Σ_k u_k ≤ R，则不需要投影
        if sum_u <= R + 1e-12:
            return clamped
        
        # 需要投影，触发标志
        self._last_projection_triggered = True
        
        # 执行带下界的比例缩放投影
        if sum_u > 0:
            scale = R / sum_u
            projected = {cid: self.eps_min + scale * u_k[cid] for cid in clamped.keys()}
        else:
            # 所有u_k都为0，直接返回ε_min
            projected = {cid: self.eps_min for cid in clamped.keys()}
        
        return projected

    def _count_boundary_hits(self, eps_alloc: Dict[int, float]) -> Tuple[int, int]:
        tol = 1e-10
        n_min, n_max = 0, 0
        for v in eps_alloc.values():
            if abs(v - self.eps_min) <= tol:
                n_min += 1
            if abs(v - self.eps_max) <= tol:
                n_max += 1
        return n_min, n_max

    # ---------------- Signal normalization ----------------

    def _norm_signal(self, x: float, signal_type: str) -> float:
        """
        实现论文中的信号归一化公式（论文3.4.2节）
        公式：norm(x) = clip((x - L) / (U - L + ε), clip_low, clip_high)
        其中：
          - L, U: 信号的全局历史下界和上界
          - ε: 数值稳定项
          - clip_low, clip_high: 截断边界
        """
        if signal_type not in self.signal_bounds:
            return x
        
        bounds = self.signal_bounds[signal_type]
        L = bounds['L']
        U = bounds['U']
        
        # 如果上下界未初始化或相等，返回默认值
        if L == float('inf') or U == float('-inf') or U - L <= 0:
            return 0.5  # 返回中间值
        
        # 计算归一化值
        norm_val = (x - L) / (U - L + self.norm_epsilon)
        
        # 应用截断
        norm_val = min(max(norm_val, self.norm_clip_low), self.norm_clip_high)
        
        return float(norm_val)

    def _update_signal_bounds(self, signal_type: str, value: float):
        """
        更新信号的全局历史上下界统计（论文3.4.2节）
        递推更新：L ← min(L, x), U ← max(U, x)
        """
        if signal_type not in self.signal_bounds:
            return
        
        bounds = self.signal_bounds[signal_type]
        
        # 更新下界
        if value < bounds['L']:
            bounds['L'] = value
        
        # 更新上界
        if value > bounds['U']:
            bounds['U'] = value

    # ---------------- RS-BS split eta ----------------

    def _compute_eta(self, cid: int, reward: float, sensitivity: float) -> float:
        """
        实现论文中的拆分系数η动态更新公式（论文3.4.3节）
        公式：η_k^{t+1} = clip(η0 + η1·ρ̄_k^t - η2·s_k, 0, 1)
        其中：
          - η0, η1, η2: 可调超参数
          - ρ̄_k^t: 平滑一致性信号（从客户端获取）
          - s_k: 客户端敏感度
        """
        # 获取超参数
        eta0 = float(self.rsbs.get("eta0", 0.5))  # 基础拆分系数
        eta1 = float(self.rsbs.get("eta1", 0.3))  # 一致性信号权重
        eta2 = float(self.rsbs.get("eta2", 0.2))  # 敏感度抑制权重
        
        # 获取客户端对象
        client = self.clients.get(cid)
        if client is None:
            # 如果客户端不存在，使用默认值
            return float(self.rsbs.get("eta_default", 0.5))
        
        # 获取平滑一致性信号ρ̄_k^t
        rho_ema = getattr(client, "rho_ema", 0.0)
        
        # 获取客户端敏感度s_k
        s_k = float(sensitivity)
        
        # 计算拆分系数：η = η0 + η1·ρ̄ - η2·s
        eta = eta0 + eta1 * rho_ema - eta2 * s_k
        
        # 应用裁剪到[0, 1]区间
        eta = min(max(eta, 0.0), 1.0)
        
        return float(eta)

    # ---------------- Allocation strategies ----------------

    def _alloc_fixed(self, selected: List[int], fixed_eps: float) -> Dict[int, float]:
        eps = {cid: float(fixed_eps) for cid in selected}
        return self._project_budget_pool(eps)

    def _alloc_linear_decay(self, selected: List[int], t: int, T: int, start: float, end: float) -> Dict[int, float]:
        if T <= 1:
            val = end
        else:
            val = start + (end - start) * (t / (T - 1))
        eps = {cid: float(val) for cid in selected}
        return self._project_budget_pool(eps)

    def _alloc_heuristic(self, selected: List[int]) -> Dict[int, float]:
        scores = []
        for cid in selected:
            scores.append(max(1e-6, float(self.reward_ema.get(cid, 0.0)) + 1e-3))
        ssum = sum(scores)
        free = max(0.0, self.B_total - self.eps_min * len(selected))
        eps = {}
        for cid, sc in zip(selected, scores):
            eps[cid] = self.eps_min + free * (sc / max(ssum, 1e-12))
        return self._project_budget_pool(eps)

    def _alloc_rsbs(self, selected: List[int]) -> Dict[int, float]:
        """
        实现RS-BS算法的预算分配（论文算法3-1步骤5）
        更新规则：ε̃_k^{t+1} = ε_k^t + γ·(r_k^t - β·s_k)
        其中：
          - γ: 步长 (self.rsbs.get("gamma", 0.1))
          - r_k^t: 回报信号 (从客户端获取的完整RS-BS信号)
          - β: 敏感度权重 (self.rsbs.get("beta", 1.0))
          - s_k: 客户端敏感度
        """
        # 获取RS-BS参数
        gamma = float(self.rsbs.get("gamma", 0.1))  # 步长
        beta = float(self.rsbs.get("beta", 1.0))    # 敏感度权重
        lambda_l = float(self.rsbs.get("lambda_l", 0.3))  # 损失代理权重
        lambda_c = float(self.rsbs.get("lambda_c", 0.5))  # 代价惩罚系数
        
        # 初始化预算字典（包含所有客户端，不仅仅是selected）
        eps_tilde = {}
        
        # 对每个客户端应用更新规则
        for cid in self.clients.keys():
            client = self.clients[cid]
            
            # 获取当前预算（如果不存在则使用初始值）
            current_eps = getattr(client, "current_epsilon", self.eps_min)
            
            if cid in selected:
                # 获取客户端敏感度
                s_k = float(getattr(client, "hetero").sensitivity)
                
                # 获取RS-BS信号（从客户端返回的结果中）
                rho_ema = getattr(client, "rho_ema", 0.0)  # 一致性信号EMA
                g_ema = getattr(client, "g_ema", 0.0)      # 梯度范数信号EMA
                m_ema = getattr(client, "m_ema", 0.0)      # 损失代理信号EMA
                c_ema = getattr(client, "c_ema", 0.0)      # 代价信号EMA
                
                # 使用完整的归一化逻辑（论文3.4.2节）
                g_norm = self._norm_signal(g_ema, 'g')      # 归一化梯度范数信号
                m_norm = self._norm_signal(m_ema, 'm')      # 归一化损失代理信号
                c_norm = self._norm_signal(c_ema, 'c')      # 归一化代价信号
                
                # 计算回报信号 r_k^t = ρ̄_k^t + norm(ḡ_k^t) + λ_ℓ·norm(m̄_k^t) - λ_c·norm(c_k^t)
                # 注意：ρ̄_k^t已经是[0,1]范围内的值，不需要额外归一化
                r_k_t = rho_ema + g_norm + lambda_l * m_norm - lambda_c * c_norm
                
                # 应用更新规则：ε̃_k^{t+1} = ε_k^t + γ·(r_k^t - β·s_k)
                eps_tilde[cid] = current_eps + gamma * (r_k_t - beta * s_k)
            else:
                # 未参与训练的客户端保持预算不变
                eps_tilde[cid] = current_eps
        
        # 应用区间裁剪和预算池投影
        return self._project_budget_pool(eps_tilde)

    # ---------------- Main training loop ----------------

    def run(
        self,
        strategy: StrategyConfig,
        rounds: int,
        clients_per_round: int,
        track_clients: Optional[List[int]] = None,
        force_include_clients: Optional[List[int]] = None,
    ):
        track_clients = track_clients or []
        smooth = float(self.rsbs.get("reward_smooth", 0.8))

        # round 0
        acc0, loss0 = self.evaluate_global()
        client_var0, worst0 = self._evaluate_clients_fairness()
        self.logger.log_round_metrics(
            rnd=0,
            strategy=strategy.name,
            global_acc=acc0,
            global_loss=loss0,
            budget_used=0.0,
            fairness_var=float("nan"),
            client_acc_var=client_var0,
            worst_client_acc=worst0,
            pool_usage=0.0,
            num_eps_min=0.0,
            num_eps_max=0.0,
            projection_triggered=0.0,
        )

        for t in range(1, rounds + 1):
            selected = self._select_clients_with_forced(
                clients_per_round=clients_per_round,
                force_include_clients=force_include_clients,
            )

            if strategy.name == "fixed":
                eps_alloc = self._alloc_fixed(selected, strategy.fixed_epsilon)
            elif strategy.name == "linear_decay":
                eps_alloc = self._alloc_linear_decay(selected, t - 1, rounds, strategy.linear_start, strategy.linear_end)
            elif strategy.name == "heuristic":
                eps_alloc = self._alloc_heuristic(selected)
            elif strategy.name == "rsbs":
                eps_alloc = self._alloc_rsbs(selected)
            else:
                raise ValueError(f"Unknown strategy: {strategy.name}")

            budget_used = float(sum(eps_alloc.values()))
            pool_usage = float(budget_used / max(self.B_total, 1e-12))
            n_min, n_max = self._count_boundary_hits(eps_alloc)
            proj = 1.0 if self._last_projection_triggered else 0.0

            eta_alloc = {}
            for cid in selected:
                if not self.enable_two_stage:
                    eta_alloc[cid] = 1.0
                else:
                    client = self.clients[cid]
                    sens = float(getattr(client, "hetero").sensitivity)
                    eta_alloc[cid] = self._compute_eta(
                        cid=cid,
                        reward=float(self.reward_ema.get(cid, 0.0)),
                        sensitivity=sens,
                    )

            updates = []
            client_rows = []
            for cid in selected:
                client = self.clients[cid]
                res = client.local_train_two_stage(
                    global_model=self.global_model,
                    epsilon_total=eps_alloc[cid],
                    eta=eta_alloc[cid],
                    dp_delta=self.dp_delta,
                    clip_norm=self.clip_norm,
                    lr=float(self.train_params["lr"]),
                    momentum=float(self.train_params["momentum"]),
                    weight_decay=float(self.train_params["weight_decay"]),
                    local_epochs_up=int(self.train_params["local_epochs_up"]),
                    local_epochs_ft=int(self.train_params["local_epochs_ft"]) if self.enable_two_stage else 0,
                )
                updates.append(res)

                raw = float(res["raw_reward"])
                prev = float(self.reward_ema.get(cid, 0.0))
                self.reward_ema[cid] = smooth * prev + (1.0 - smooth) * raw
                
                # 更新信号边界（论文3.4.2节）
                # 从客户端获取RS-BS信号并更新全局历史上下界
                if hasattr(client, "rho_ema"):
                    self._update_signal_bounds('rho', float(client.rho_ema))
                if hasattr(client, "g_ema"):
                    self._update_signal_bounds('g', float(client.g_ema))
                if hasattr(client, "m_ema"):
                    self._update_signal_bounds('m', float(client.m_ema))
                if hasattr(client, "c_ema"):
                    self._update_signal_bounds('c', float(client.c_ema))

                client_rows.append(
                    dict(
                        round=t,
                        cid=cid,
                        epsilon_total=float(res["epsilon_total"]),
                        epsilon_up=float(res["epsilon_up"]),
                        epsilon_ft=float(res["epsilon_ft"]),
                        eta=float(res["eta"]),
                        raw_reward=float(res["raw_reward"]),
                        reward_ema=float(self.reward_ema[cid]),
                        sensitivity=float(res["sensitivity"]),
                        cost=float(res["cost"]),
                        nu=float(res["nu"]),
                        omega=float(res["omega"]),
                        tracked=int(cid in track_clients),
                    )
                )

            self._aggregate(updates)

            gacc, gloss = self.evaluate_global()
            fairness_var = compute_fairness_var([row["raw_reward"] for row in client_rows])
            client_acc_var, worst_client_acc = self._evaluate_clients_fairness()

            # ---- ablation table stats: only compute at final round to save time
            client_stats = {}
            if t == rounds:
                client_stats = self._client_acc_distribution_stats()

            self.logger.log_round_metrics(
                rnd=t,
                strategy=strategy.name,
                global_acc=gacc,
                global_loss=gloss,
                budget_used=budget_used,
                fairness_var=fairness_var,
                client_acc_var=client_acc_var,
                worst_client_acc=worst_client_acc,
                pool_usage=pool_usage,
                num_eps_min=float(n_min),
                num_eps_max=float(n_max),
                projection_triggered=proj,
                client_acc_mean=float(client_stats.get("client_acc_mean", float("nan"))),
                client_acc_p10=float(client_stats.get("client_acc_p10", float("nan"))),
                client_acc_p50=float(client_stats.get("client_acc_p50", float("nan"))),
                client_acc_p90=float(client_stats.get("client_acc_p90", float("nan"))),
                client_acc_worst=float(client_stats.get("client_acc_worst", float("nan"))),
            )
            self.logger.log_client_rows(client_rows)

    # ---------------- Aggregation ----------------

    def _aggregate(self, updates: List[Dict]):
        if not updates:
            return
        total = sum(int(u["num_samples"]) for u in updates)
        total = max(total, 1)

        new_state = copy.deepcopy(self.global_model.state_dict())
        for k in new_state.keys():
            agg_delta = None
            for u in updates:
                w = float(int(u["num_samples"]) / total)
                d = u["update_state_dict"][k].to(self.device)
                if agg_delta is None:
                    agg_delta = w * d
                else:
                    agg_delta = agg_delta + w * d
            new_state[k] = (new_state[k] + agg_delta).detach()

        self.global_model.load_state_dict(new_state)
