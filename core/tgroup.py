"""
TGroup: Transformer 群操作

继承 Group（纯群操作），加上与 Transformer hidden_states 的交互：
- proj_to: hidden_states → (d,d) 矩阵
- proj_from: (d,d) 修正量 → hidden_states 修正
- forward: 润色 hidden_states

不涉及训练和推理逻辑。
"""

import torch
import torch.nn as nn

from .group import Group, cayley_exp


class TGroup(Group, nn.Module):
    """Transformer 群操作类

    继承 Group 的群操作，加上与 hidden_states 的交互。

    参数：
        d: 群表示维度
        hidden_dim: Transformer 隐层维度
        num_generators: 生成元个数
        is_identity: 是否为透明群
    """

    def __init__(
        self,
        d: int = 16,
        hidden_dim: int = 768,
        num_generators: int = 6,
        is_identity: bool = False,
    ):
        nn.Module.__init__(self)
        Group.__init__(self, d, num_generators, is_identity)

        self.hidden_dim = hidden_dim

        # Group.__init__ 创建了 plain tensor，需转为 nn.Parameter 使其可训练
        del self.generators
        if not is_identity:
            A = torch.randn(num_generators, d, d)
            self.register_parameter(
                'generators',
                nn.Parameter(0.5 * (A - A.transpose(-1, -2)))
            )
        else:
            self.register_parameter(
                'generators',
                nn.Parameter(torch.zeros(num_generators, d, d))
            )

        if not is_identity:
            import math
            matrix_dim = d * d
            self.proj_to = nn.Linear(hidden_dim, matrix_dim, bias=True)
            self.proj_from = nn.Linear(matrix_dim, hidden_dim, bias=True)

            # proj_to: Kaiming * 0.1
            nn.init.kaiming_uniform_(self.proj_to.weight, a=math.sqrt(5))
            with torch.no_grad():
                self.proj_to.weight.mul_(0.1)
            nn.init.zeros_(self.proj_to.bias)

            # proj_from: 零初始化
            nn.init.zeros_(self.proj_from.weight)
            nn.init.zeros_(self.proj_from.bias)

    def forward(self, hidden_states: torch.Tensor, num_steps: int = 1, lr: float = 0.1) -> tuple:
        """润色 hidden_states

        Args:
            hidden_states: (B, S, H)

        Returns:
            (output, delta, dist)
            - output: 润色后 hidden_states (B, S, H)
            - delta: 群增量 (B, S, d, d)
            - dist: 流形距离 (B, S)
        """
        if self.is_identity:
            B, S, _ = hidden_states.shape
            return (
                hidden_states,
                torch.zeros(B, S, self.d, self.d, device=hidden_states.device, dtype=hidden_states.dtype),
                torch.zeros(B, S, device=hidden_states.device, dtype=hidden_states.dtype),
            )

        B, S, _ = hidden_states.shape
        N = B * S

        # 投影到矩阵空间
        matrix = self.proj_to(hidden_states)  # (B, S, d*d)
        matrix = matrix.view(N, self.d, self.d)

        # 裁剪
        norms = torch.linalg.norm(matrix, dim=(-2, -1), keepdim=True)
        scaling = torch.clamp(norms, max=100.0) / (norms + 1e-8)
        matrix = matrix * scaling

        # 群增量
        _, delta, dist = self.get_correction(matrix, num_steps, lr)

        # 修正量映射回 H 空间
        correction = self.proj_from(delta.view(B, S, -1))  # (B, S, H)

        output = hidden_states + correction

        return output, delta.view(B, S, self.d, self.d), dist.view(B, S)
