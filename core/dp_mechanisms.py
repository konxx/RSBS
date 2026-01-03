from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple, List, Optional, Union
from enum import Enum
import math
import torch
import numpy as np


class DPChannelType(Enum):
    """差分隐私通道类型枚举"""
    TRAINING_UPDATE = "training_update"  # 训练更新通道
    LOSS_PROXY = "loss_proxy"           # 损失代理通道
    VALIDATION = "validation"           # 验证通道
    OTHER = "other"                     # 其他通道


@dataclass
class DPMechanismRecord:
    """差分隐私机制记录"""
    channel_type: DPChannelType
    epsilon: float  # 隐私预算ε
    delta: float    # 失效概率δ
    sigma: float    # 噪声倍率
    steps: int      # 步数
    timestamp: int  # 时间戳（轮次）


class AdaptiveRDPAccountant:
    """
    自适应组合RDP会计器，支持多通道和动态预算更新
    
    根据论文3.5.1节要求实现：
    1. 支持动态预算更新的自适应组合
    2. 预算更新所依赖的信号必须为差分隐私机制的输出或其后处理
    3. 损失代理的发布引入了额外通道，其隐私损耗需与训练更新的隐私损耗分别记账
    4. 在跨轮次统计时按自适应组合原则共同累计
    
    实现基于Renyi差分隐私(RDP)的自适应组合会计
    """
    
    def __init__(self, delta: float = 1e-5):
        """
        初始化自适应RDP会计器
        
        Args:
            delta: 失效概率δ
        """
        self.delta = float(delta)
        self.records: List[DPMechanismRecord] = []
        self.orders = [1.5, 2, 4, 8, 16, 32, 64]  # RDP阶数α
        
    def add_mechanism(self, 
                     channel_type: DPChannelType,
                     epsilon: float,
                     sigma: float,
                     steps: int = 1,
                     timestamp: Optional[int] = None) -> None:
        """
        添加差分隐私机制记录
        
        Args:
            channel_type: 通道类型
            epsilon: 隐私预算ε
            sigma: 噪声倍率
            steps: 步数（默认为1）
            timestamp: 时间戳（轮次），如果为None则使用记录数量
        """
        if timestamp is None:
            timestamp = len(self.records)
            
        record = DPMechanismRecord(
            channel_type=channel_type,
            epsilon=float(epsilon),
            delta=self.delta,
            sigma=float(sigma),
            steps=int(steps),
            timestamp=int(timestamp)
        )
        self.records.append(record)
        
    def get_rdp_for_gaussian(self, sigma: float, alpha: float) -> float:
        """
        计算高斯机制的RDP值
        
        Args:
            sigma: 噪声倍率
            alpha: RDP阶数
            
        Returns:
            RDP值ε(α)
        """
        if sigma <= 0:
            return float('inf')
        return alpha / (2.0 * sigma ** 2)
    
    def get_composition_rdp(self, alpha: float, channel_filter: Optional[List[DPChannelType]] = None) -> float:
        """
        计算组合RDP值
        
        Args:
            alpha: RDP阶数
            channel_filter: 通道过滤器，如果为None则包含所有通道
            
        Returns:
            组合RDP值
        """
        total_rdp = 0.0
        
        for record in self.records:
            if channel_filter is not None and record.channel_type not in channel_filter:
                continue
                
            # 对于每个记录，计算其RDP贡献
            # 注意：这里假设每个记录是独立的机制
            record_rdp = self.get_rdp_for_gaussian(record.sigma, alpha) * record.steps
            total_rdp += record_rdp
            
        return total_rdp
    
    def get_epsilon_delta(self, channel_filter: Optional[List[DPChannelType]] = None) -> Tuple[float, float]:
        """
        计算(ε, δ)隐私保证
        
        Args:
            channel_filter: 通道过滤器，如果为None则包含所有通道
            
        Returns:
            (ε, δ)元组
        """
        best_epsilon = float('inf')
        
        for alpha in self.orders:
            rdp = self.get_composition_rdp(alpha, channel_filter)
            
            # 从RDP转换到(ε, δ)
            # ε = RDP(α) + log(1/δ)/(α-1)
            epsilon = rdp + math.log(1.0 / self.delta) / (alpha - 1.0)
            
            if epsilon < best_epsilon:
                best_epsilon = epsilon
                
        return best_epsilon, self.delta
    
    def get_channel_summary(self) -> Dict[str, Dict]:
        """
        获取各通道的隐私消耗摘要
        
        Returns:
            通道摘要字典
        """
        summary = {}
        
        for channel_type in DPChannelType:
            channel_records = [r for r in self.records if r.channel_type == channel_type]
            if not channel_records:
                continue
                
            total_epsilon = sum(r.epsilon for r in channel_records)
            total_steps = sum(r.steps for r in channel_records)
            
            # 计算该通道的组合RDP
            epsilon, delta = self.get_epsilon_delta([channel_type])
            
            summary[channel_type.value] = {
                "record_count": len(channel_records),
                "total_epsilon_sum": total_epsilon,
                "total_steps": total_steps,
                "composed_epsilon": epsilon,
                "composed_delta": delta
            }
            
        return summary
    
    def get_total_summary(self) -> Dict[str, float]:
        """
        获取总隐私消耗摘要
        
        Returns:
            总摘要字典
        """
        total_epsilon_sum = sum(r.epsilon for r in self.records)
        total_steps = sum(r.steps for r in self.records)
        
        epsilon, delta = self.get_epsilon_delta()
        
        return {
            "total_records": len(self.records),
            "total_epsilon_sum": total_epsilon_sum,
            "total_steps": total_steps,
            "composed_epsilon": epsilon,
            "composed_delta": delta
        }
    
    def clear(self) -> None:
        """清除所有记录"""
        self.records.clear()


@dataclass
class DPGaussianMechanism:
    """
    DP-SGD高斯机制实现
    
    支持：
    - 逐参数梯度裁剪（全局范数）
    - 高斯噪声添加
    - 与自适应RDP会计器集成
    """
    clip_norm: float
    noise_multiplier: float  # sigma
    accountant: Optional[AdaptiveRDPAccountant] = None
    channel_type: DPChannelType = DPChannelType.TRAINING_UPDATE

    def clip_and_noise_(self, parameters: Iterable[torch.nn.Parameter], steps: int = 1) -> Tuple[float, float]:
        """
        就地裁剪梯度并添加噪声
        
        Args:
            parameters: 模型参数
            steps: 训练步数（用于会计）
            
        Returns:
            (裁剪前的梯度范数, 裁剪因子)
        """
        grads = []
        for p in parameters:
            if p.grad is not None:
                grads.append(p.grad.detach().view(-1))
        if not grads:
            return 0.0, 1.0

        flat = torch.cat(grads)
        grad_norm = torch.norm(flat, p=2).item()
        clip_factor = min(1.0, self.clip_norm / (grad_norm + 1e-12))

        for p in parameters:
            if p.grad is None:
                continue
            p.grad.mul_(clip_factor)
            if self.noise_multiplier > 0:
                noise = torch.randn_like(p.grad) * (self.noise_multiplier * self.clip_norm)
                p.grad.add_(noise)

        # 记录到会计器（如果提供）
        if self.accountant is not None and self.noise_multiplier > 0:
            # 计算等效隐私预算
            # 对于高斯机制，ε ≈ α/(2σ²) for RDP order α
            # 这里我们使用一个保守的估计，实际ε由会计器计算
            conservative_epsilon = 1.0 / (2.0 * self.noise_multiplier ** 2) if self.noise_multiplier > 0 else float('inf')
            
            self.accountant.add_mechanism(
                channel_type=self.channel_type,
                epsilon=conservative_epsilon,
                sigma=self.noise_multiplier,
                steps=steps
            )

        return float(grad_norm), float(clip_factor)


class LossProxyDPMechanism:
    """
    损失代理差分隐私机制
    
    用于保护损失变化率Δℓ的上报，根据论文3.4.2节：
    - 客户端计算Δℓ = ℓ_k(θ^t) - ℓ_k(θ^{t+1})
    - 对Δℓ进行截断和噪声注入
    - 使用(ε_ℓ, δ_ℓ)-DP高斯机制上报
    """
    
    def __init__(self, 
                 clip_threshold: float,
                 epsilon: float,
                 delta: float,
                 accountant: Optional[AdaptiveRDPAccountant] = None):
        """
        初始化损失代理DP机制
        
        Args:
            clip_threshold: 截断阈值C_ℓ
            epsilon: 隐私预算ε_ℓ
            delta: 失效概率δ_ℓ
            accountant: 自适应RDP会计器
        """
        self.clip_threshold = float(clip_threshold)
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.accountant = accountant
        
        # 计算高斯噪声的标准差
        # 对于标量高斯机制，灵敏度 = 2 * clip_threshold
        # σ = 灵敏度 * sqrt(2 * log(1.25/δ)) / ε
        sensitivity = 2.0 * self.clip_threshold
        self.sigma = sensitivity * math.sqrt(2.0 * math.log(1.25 / self.delta)) / self.epsilon
        
    def apply(self, delta_loss: float) -> float:
        """
        应用差分隐私保护
        
        Args:
            delta_loss: 原始损失变化率Δℓ
            
        Returns:
            加噪后的损失变化率
        """
        # 截断
        clipped = max(min(delta_loss, self.clip_threshold), -self.clip_threshold)
        
        # 添加高斯噪声
        noise = np.random.normal(0, self.sigma)
        noisy_delta_loss = clipped + noise
        
        # 记录到会计器（如果提供）
        if self.accountant is not None:
            self.accountant.add_mechanism(
                channel_type=DPChannelType.LOSS_PROXY,
                epsilon=self.epsilon,
                sigma=self.sigma,
                steps=1
            )
            
        return float(noisy_delta_loss)


def epsilon_to_noise_multiplier(epsilon: float, delta: float, steps: int, alpha: float = 16.0) -> float:
    """
    将隐私预算ε转换为噪声倍率σ
    
    使用RDP框架进行转换：
    ε ≈ steps * α/(2σ²) + log(1/δ)/(α-1)
    => σ ≈ sqrt( steps * α / (2*(ε - log(1/δ)/(α-1))) )
    
    Args:
        epsilon: 隐私预算ε
        delta: 失效概率δ
        steps: 训练步数
        alpha: RDP阶数α
        
    Returns:
        噪声倍率σ
    """
    if epsilon <= 0:
        return float('inf')
        
    c = math.log(1.0 / delta) / (alpha - 1.0)
    budget = float(epsilon) - c
    
    if budget <= 1e-10:
        return 1e6  # 返回很大的噪声倍率表示强隐私保护
        
    sigma2 = steps * alpha / (2.0 * budget)
    return float(math.sqrt(max(sigma2, 1e-12)))


def noise_multiplier_to_epsilon(sigma: float, delta: float, steps: int, alpha: float = 16.0) -> float:
    """
    将噪声倍率σ转换为隐私预算ε
    
    使用RDP框架进行转换：
    ε ≈ steps * α/(2σ²) + log(1/δ)/(α-1)
    
    Args:
        sigma: 噪声倍率σ
        delta: 失效概率δ
        steps: 训练步数
        alpha: RDP阶数α
        
    Returns:
        隐私预算ε
    """
    if sigma <= 0:
        return float('inf')
        
    rdp_term = steps * alpha / (2.0 * sigma ** 2)
    c = math.log(1.0 / delta) / (alpha - 1.0)
    
    epsilon = rdp_term + c
    return float(epsilon)


class ConservativeRDPAccountant:
    """
    保守RDP会计器（用于向后兼容）
    
    包装AdaptiveRDPAccountant，提供step方法
    """
    
    def __init__(self, delta: float = 1e-5):
        """
        初始化保守RDP会计器
        
        Args:
            delta: 失效概率δ
        """
        self.delta = float(delta)
        self.accountant = AdaptiveRDPAccountant(delta=delta)
        self.step_count = 0
        
    def step(self, sigma: float) -> None:
        """
        记录一个训练步骤
        
        Args:
            sigma: 噪声倍率
        """
        self.step_count += 1
        # 添加一个机制记录
        self.accountant.add_mechanism(
            channel_type=DPChannelType.TRAINING_UPDATE,
            epsilon=1.0 / (2.0 * sigma ** 2) if sigma > 0 else float('inf'),
            sigma=sigma,
            steps=1,
            timestamp=self.step_count
        )
        
    def get_epsilon_delta(self) -> Tuple[float, float]:
        """
        获取(ε, δ)隐私保证
        
        Returns:
            (ε, δ)元组
        """
        return self.accountant.get_epsilon_delta()


def gaussian_mechanism_for_scalar(value: float, epsilon: float, delta: float, sensitivity: float) -> float:
    """
    对标量应用高斯差分隐私机制
    
    Args:
        value: 原始标量值
        epsilon: 隐私预算ε
        delta: 失效概率δ
        sensitivity: 灵敏度
        
    Returns:
        加噪后的标量值
    """
    # 计算高斯噪声的标准差
    # σ = 灵敏度 * sqrt(2 * log(1.25/δ)) / ε
    sigma = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
    
    # 添加高斯噪声
    noise = np.random.normal(0, sigma)
    noisy_value = value + noise
    
    return float(noisy_value)
