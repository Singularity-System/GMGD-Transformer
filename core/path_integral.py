"""
路径积分分析器：检测闭合误差，触发新群创建

核心功能：
1. 计算路径积分闭合误差 ΔG = G_final^{-1} · G_target
2. 分析闭合误差的代数特性
3. 当误差超过阈值时，触发新群创建流程
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PathIntegralAnalysis:
    """路径积分分析报告"""
    deviation_from_identity: float  # 与单位阵的偏差
    symmetric_norm: float  # 对称部分范数
    skew_symmetric_norm: float  # 反对称部分范数
    determinant: float  # 行列式
    needs_new_group: bool  # 是否需要新群
    candidate_generator: torch.Tensor  # 候选生成元矩阵 (d, d)


class PathIntegralAnalyzer(nn.Module):
    """
    路径积分分析器

    通过分析最后一层输出的全局群状态与目标状态的差异，
    判断是否需要创建新群。

    参数：
        group_d: 群表示维度
        threshold: 闭合误差阈值（超过此值触发新群创建）
    """

    def __init__(self, group_d: int, threshold: float = 0.5):
        super().__init__()

        self.group_d = group_d
        self.threshold = threshold

        # 缓冲器：存储历史闭合误差
        self.error_history: List[float] = []
        self.max_history = 100

        # 候选生成元缓冲
        self.candidate_generators: List[torch.Tensor] = []

    def forward(
        self,
        G_final: torch.Tensor,
        G_target: torch.Tensor
    ) -> PathIntegralAnalysis:
        """
        分析路径积分闭合状态

        Args:
            G_final: 当前全局群状态 (B, d, d)
            G_target: 目标群状态 (B, d, d)

        Returns:
            PathIntegralAnalysis: 分析报告
        """
        B, d, _ = G_final.shape
        device = G_final.device

        # 1. 计算缺失变换 ΔG = G_final^{-1} · G_target
        try:
            G_final_inv = torch.linalg.inv(G_final)
        except torch.linalg.LinAlgError:
            # 如果 G_final 奇异，使用伪逆
            G_final_inv = torch.linalg.pinv(G_final)

        delta_missing = G_final_inv @ G_target  # (B, d, d)

        # 2. 计算与单位阵的偏差
        I = torch.eye(d, device=device).unsqueeze(0).expand(B, -1, -1)
        deviation = torch.norm(delta_missing - I, dim=(-2, -1)).mean().item()

        # 记录历史误差
        self.error_history.append(deviation)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)

        # 3. 分析 delta_missing 的代数特性
        # 对称部分 vs 反对称部分
        symmetric_part = 0.5 * (delta_missing + delta_missing.transpose(-2, -1))
        skew_symmetric_part = 0.5 * (delta_missing - delta_missing.transpose(-2, -1))

        symmetric_norm = torch.norm(symmetric_part, dim=(-2, -1)).mean().item()
        skew_symmetric_norm = torch.norm(skew_symmetric_part, dim=(-2, -1)).mean().item()

        # 4. 行列式分析
        det = torch.linalg.det(delta_missing).abs().mean().item()

        # 5. 判断是否需要新群
        # 条件：偏差超过阈值，且误差呈上升趋势
        needs_new_group = deviation > self.threshold

        # 检查误差趋势（如果有足够历史）
        if len(self.error_history) >= 10:
            recent_trend = sum(self.error_history[-5:]) - sum(self.error_history[-10:-5])
            if recent_trend > 0.1:  # 误差在上升
                needs_new_group = True

        # 6. 提取候选生成元（取 batch 平均）
        candidate_generator = delta_missing.mean(dim=0).detach()  # (d, d)

        # 保存候选生成元用于后续分析
        self.candidate_generators.append(candidate_generator.clone())
        if len(self.candidate_generators) > 10:
            self.candidate_generators.pop(0)

        return PathIntegralAnalysis(
            deviation_from_identity=deviation,
            symmetric_norm=symmetric_norm,
            skew_symmetric_norm=skew_symmetric_norm,
            determinant=det,
            needs_new_group=needs_new_group,
            candidate_generator=candidate_generator
        )

    def test_algebraic_compatibility(
        self,
        candidate: torch.Tensor,
        existing_generators: List[torch.Tensor],
        relations: List[str] = None
    ) -> Tuple[bool, float]:
        """
        测试候选生成元与现有群的代数相容性

        Args:
            candidate: 候选生成元矩阵 (d, d)
            existing_generators: 现有生成元列表
            relations: 要测试的关系（如 'commute', 'associate'）

        Returns:
            compatible: 是否相容
            relation_loss: 关系损失值
        """
        if relations is None:
            relations = ['commute']  # 默认测试交换律

        device = candidate.device
        d = candidate.shape[0]

        relation_losses = []

        for relation in relations:
            if relation == 'commute':
                # 测试与所有现有生成元的交换性
                for gen in existing_generators:
                    # [A, B] = AB - BA = 0（若交换）
                    commutator = candidate @ gen - gen @ candidate
                    loss = torch.norm(commutator).item()
                    relation_losses.append(loss)

            elif relation == 'orthogonal':
                # 测试正交性：R^T R = I
                RT_R = candidate.transpose(-2, -1) @ candidate
                I = torch.eye(d, device=device)
                loss = torch.norm(RT_R - I).item()
                relation_losses.append(loss)

        # 平均关系损失
        avg_loss = sum(relation_losses) / len(relation_losses) if relation_losses else 0.0

        # 相容性判断：损失小于阈值
        compatible = avg_loss < 0.1

        return compatible, avg_loss

    def reset(self):
        """重置分析器状态"""
        self.error_history.clear()
        self.candidate_generators.clear()

    def get_statistics(self) -> dict:
        """获取统计信息"""
        if not self.error_history:
            return {'mean_error': 0.0, 'max_error': 0.0, 'min_error': 0.0}

        return {
            'mean_error': sum(self.error_history) / len(self.error_history),
            'max_error': max(self.error_history),
            'min_error': min(self.error_history),
            'num_candidates': len(self.candidate_generators)
        }
