"""
SubGroup: 子群

每个子群独立管理自己的生成元和投影层。
MetaGroup 只管索引和扩张决策，不管子群的生成元。
"""

import torch
import torch.nn as nn

from .meta_group import cayley_exp


class SubGroup(nn.Module):
    """
    子群：独立管理一组生成元 + 投影层

    参数：
        num_generators: 生成元个数
        d: 群表示维度
        group_type: 'orthogonal' 或 'general_linear'
        hidden_dim: Transformer 隐层维度
        init_noise: 初始化噪声标准差
    """

    def __init__(
        self,
        num_generators: int = 6,
        d: int = 16,
        group_type: str = 'orthogonal',
        hidden_dim: int = 768,
        init_noise: float = 0.01,
    ):
        super().__init__()
        self.num_generators = num_generators
        self.d = d
        self.group_type = group_type
        self.hidden_dim = hidden_dim

        # 生成元
        if group_type == 'orthogonal':
            self.generator_params = nn.Parameter(
                torch.randn(num_generators, d, d) * init_noise
            )
            with torch.no_grad():
                self.generator_params.copy_(
                    self.generator_params - self.generator_params.transpose(-1, -2)
                )
        else:
            self.generator_params = nn.Parameter(
                torch.randn(num_generators, d, d) * init_noise
            )

        # 投影层：hidden_dim → d*d
        matrix_dim = d * d
        self.proj_to = nn.Linear(hidden_dim, matrix_dim, bias=True)
        self.proj_from = nn.Linear(matrix_dim, hidden_dim, bias=True)

        # 初始化策略
        import math
        nn.init.kaiming_uniform_(self.proj_to.weight, a=math.sqrt(5))
        with torch.no_grad():
            self.proj_to.weight.mul_(0.1)
        nn.init.zeros_(self.proj_to.bias)
        nn.init.zeros_(self.proj_from.weight)
        nn.init.zeros_(self.proj_from.bias)

    def all_generators(self) -> torch.Tensor:
        """获取所有生成元矩阵 (k, d, d)"""
        if self.group_type == 'orthogonal':
            return cayley_exp(self.generator_params)
        return self.generator_params

    def project_to_manifold_batch(
        self,
        X: torch.Tensor,
        num_steps: int = 1,
        lr: float = 0.1,
    ) -> torch.Tensor:
        """
        批量投影到本子群的流形

        Args:
            X: (N, d, d)
            num_steps: 投影步数
            lr: 学习率

        Returns:
            投影后矩阵 (N, d, d)
        """
        gens = self.all_generators()  # (k, d, d)
        X_proj = X.clone()

        for _ in range(num_steps):
            coeffs = torch.einsum('nij,kij->nk', X_proj, gens)
            approx = torch.einsum('nk,kij->nij', coeffs, gens)
            X_proj = X_proj - lr * (X_proj - approx)

        return X_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_steps: int = 1,
        lr: float = 0.1,
    ) -> tuple:
        """
        子群前向传播：投影 → 流形投影 → 反投影

        Args:
            hidden_states: (B, S, H) 或 (N, H)
            num_steps: 投影步数
            lr: 学习率

        Returns:
            (output, manifold_dist, delta)
            - output: (N, H) 平滑后的隐状态
            - manifold_dist: (N,) 流形距离向量
            - delta: (N, d, d) 群增量
        """
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        B, S, _ = hidden_states.shape
        N = B * S

        # 投影到矩阵空间
        matrix_flat = self.proj_to(hidden_states)  # (B, S, d*d)
        matrices = matrix_flat.view(N, self.d, self.d)

        # 裁剪输入矩阵
        norms = torch.linalg.norm(matrices, dim=(-2, -1), keepdim=True)
        max_norm = 100.0
        scaling = torch.clamp(norms, max=max_norm) / (norms + 1e-8)
        matrices = matrices * scaling

        # 流形投影
        smoothed = self.project_to_manifold_batch(matrices, num_steps=num_steps, lr=lr)

        # 流形距离
        dist = torch.norm(smoothed - matrices, dim=(-2, -1))  # (N,)

        # 群增量
        delta = smoothed - matrices  # (N, d, d)

        # 重塑 + 反投影
        smooth_flat = smoothed.view(B, S, -1)
        output = self.proj_from(smooth_flat)  # (B, S, H)

        if squeeze_output:
            output = output.squeeze(0)
            dist = dist.squeeze(0) if dist.dim() > 1 else dist
            delta = delta.squeeze(0)

        return output, dist, delta

    def extra_repr(self) -> str:
        return f'num_generators={self.num_generators}, d={self.d}, type={self.group_type}, hidden_dim={self.hidden_dim}'
