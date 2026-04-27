"""
MetaGroup: 元群

管理多个 SubGroup 的索引，监控路径积分闭合，决定何时扩张。
元群不管子群的生成元——子群自己管。

核心逻辑：
- 路径积分误差持续过大 → "人手不够解决问题了" → add_subgroup()
- 子群负责各自的投影
"""

import torch
import torch.nn as nn
from typing import List, Optional


def cayley_exp(A: torch.Tensor) -> torch.Tensor:
    """Cayley 变换计算矩阵指数（正交群）"""
    A_skew = 0.5 * (A - A.transpose(-1, -2))
    norm_A = torch.linalg.norm(A_skew, ord='fro', dim=(-2, -1), keepdim=True)
    scale = torch.clamp(norm_A, min=1e-6)
    A_skew = torch.where(scale > 1.0, A_skew / scale, A_skew)
    d = A.shape[-1]
    I = torch.eye(d, device=A.device, dtype=A.dtype)
    if A.dim() == 4:
        I_expanded = I.unsqueeze(0).unsqueeze(0)
    elif A.dim() == 3:
        I_expanded = I.unsqueeze(0)
    else:
        I_expanded = I
    half_A = 0.5 * A_skew
    numerator = I_expanded + half_A
    denominator = I_expanded - half_A + 1e-6 * I_expanded
    if A.dim() == 4:
        B, N = A_skew.shape[:2]
        numerator = numerator.view(B * N, A.shape[-2], A.shape[-1])
        denominator = denominator.view(B * N, A.shape[-2], A.shape[-1])
        result = torch.linalg.solve(denominator, numerator)
        return result.view(B, N, A.shape[-2], A.shape[-1])
    return torch.linalg.solve(denominator, numerator)


def stable_matrix_exp(A: torch.Tensor, max_spectral_radius: float = 3.0) -> torch.Tensor:
    """数值稳定的矩阵指数（备用）"""
    spectral_radius = torch.linalg.norm(A, ord='fro', dim=(-2, -1), keepdim=True)
    scale_factor = torch.clamp(spectral_radius / max_spectral_radius, min=1.0)
    return torch.linalg.matrix_exp(A / scale_factor)


class MetaGroup(nn.Module):
    """
    元群：管理 SubGroup 索引，监控扩张

    参数：
        d: 群表示维度
        group_type: 'orthogonal' 或 'general_linear'
        hidden_dim: Transformer 隐层维度
        expansion_threshold: 路径积分误差超过此值触发扩张
        initial_subgroups: 初始子群数量
        initial_generators_per_subgroup: 每个子群的初始生成元数
    """

    def __init__(
        self,
        d: int = 16,
        group_type: str = 'orthogonal',
        hidden_dim: int = 768,
        expansion_threshold: float = 0.5,
        initial_subgroups: int = 1,
        initial_generators_per_subgroup: int = 6,
    ):
        super().__init__()
        self.d = d
        self.group_type = group_type
        self.hidden_dim = hidden_dim
        self.expansion_threshold = expansion_threshold

        self.subgroups = nn.ModuleList()
        for _ in range(initial_subgroups):
            self._add_subgroup(initial_generators_per_subgroup)

        self.last_expansion_step = -1000
        self.min_steps_between_expansion = 50

    def _add_subgroup(self, num_generators: int):
        """创建并添加子群（延迟导入避免循环）"""
        from .subgroup import SubGroup
        sg = SubGroup(
            num_generators=num_generators,
            d=self.d,
            group_type=self.group_type,
            hidden_dim=self.hidden_dim,
        )
        self.subgroups.append(sg)
        print(f"[MetaGroup] 新增子群 #{len(self.subgroups) - 1} (generators={num_generators})")

    def check_expansion(self, path_integral_error: float, global_step: int) -> bool:
        """
        检查是否需要扩张：路径积分误差过大 → 人手不够 → 加人
        """
        if (path_integral_error > self.expansion_threshold
                and global_step - self.last_expansion_step >= self.min_steps_between_expansion
                and len(self.subgroups) < 10):
            self._add_subgroup(num_generators=1)
            self.last_expansion_step = global_step
            return True
        return False

    def project_all(
        self,
        X: torch.Tensor,
        num_steps: int = 1,
        lr: float = 0.1,
    ) -> List[tuple]:
        """
        所有子群独立投影

        Returns:
            List[(smoothed_X, manifold_dist)] 每子群一个元组
        """
        results = []
        for sg in self.subgroups:
            smooth = sg.project_to_manifold_batch(X, num_steps=num_steps, lr=lr)
            dist = torch.norm(smooth - X, dim=(-2, -1))
            results.append((smooth, dist))
        return results

    def extra_repr(self) -> str:
        return (f'd={self.d}, type={self.group_type}, '
                f'subgroups={len(self.subgroups)}, threshold={self.expansion_threshold}')
