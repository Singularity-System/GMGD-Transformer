"""
Group: 群的数学基础实现

核心功能：
- 生成元管理
- Cayley 变换
- 流形投影
- 不涉及 Transformer，纯矩阵操作
"""

import torch
import torch.nn as nn


def cayley_exp(A: torch.Tensor) -> torch.Tensor:
    """Cayley 变换：反对称矩阵 → 正交矩阵
    (I + A/2)(I - A/2)^{-1}

    Args:
        A: 反对称矩阵 (..., d, d)

    Returns:
        正交矩阵 (..., d, d)
    """
    half_A = 0.5 * A
    I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)

    while I.dim() < A.dim():
        I = I.unsqueeze(0)

    numerator = I + half_A
    denominator = I - half_A + 1e-6 * I
    return torch.linalg.solve(denominator, numerator)


class Group:
    """纯群操作类

    参数：
        d: 群表示维度
        num_generators: 生成元个数
        is_identity: 是否为透明群（恒等变换）
    """

    def __init__(self, d: int = 16, num_generators: int = 6, is_identity: bool = False):
        self.d = d
        self.num_generators = num_generators
        self.is_identity = is_identity

        if is_identity:
            self.generators = torch.zeros(num_generators, d, d)
        else:
            # 随机反对称初始化
            A = torch.randn(num_generators, d, d)
            self.generators = 0.5 * (A - A.transpose(-1, -2))

    def get_generator_matrices(self) -> torch.Tensor:
        """获取群生成元矩阵（Cayley 变换后）

        Returns:
            (num_generators, d, d) 正交矩阵
        """
        if self.is_identity:
            return torch.zeros(self.num_generators, self.d, self.d)
        return cayley_exp(self.generators)

    def project_to_manifold(self, X: torch.Tensor, num_steps: int = 1, lr: float = 0.1) -> torch.Tensor:
        """投影到群流形

        Args:
            X: 输入矩阵 (N, d, d)
            num_steps: 投影步数
            lr: 学习率

        Returns:
            投影后矩阵 (N, d, d)
        """
        if self.is_identity:
            N = X.shape[0]
            return torch.eye(self.d, device=X.device, dtype=X.dtype).unsqueeze(0).expand(N, -1, -1)

        gens = self.get_generator_matrices()  # (k, d, d)
        X_proj = X.clone()

        for _ in range(num_steps):
            coeffs = torch.einsum('nij,kij->nk', X_proj, gens)
            approx = torch.einsum('nk,kij->nij', coeffs, gens)
            X_proj = X_proj - lr * (X_proj - approx)

        return X_proj

    def get_correction(self, X: torch.Tensor, num_steps: int = 1, lr: float = 0.1) -> tuple:
        """群增量（修正量）

        Args:
            X: 输入矩阵 (N, d, d)

        Returns:
            (smoothed, delta, dist)
            - smoothed: 投影后矩阵 (N, d, d)
            - delta: 群增量 = smoothed - X (N, d, d)
            - dist: 流形距离 ||delta|| (N,)
        """
        smoothed = self.project_to_manifold(X, num_steps, lr)
        delta = smoothed - X
        dist = torch.norm(delta, dim=(-2, -1))  # (N,)
        return smoothed, delta, dist
