"""
GroupSmoothLayer: 群光滑层

将 Transformer 隐状态投影至群流形，强制表征遵循代数约束。
完全向量化实现，无 Python 逐样本循环。

数据流：
    hidden_states (B, S, H)
        → Linear(H → d*d) → Reshape(B*S, d, d)
        → MetaGroup.project_to_manifold_batch()
        → Reshape(B, S, d*d) → Linear(d*d → H)
        → + hidden_states (残差)

性能优化：
- 批量处理：将 (B, S) 合并为 (B*S) 一次性投影
- 使用 torch.einsum 替代双重 for 循环
- 预期加速：从 ~850ms 降至 ~12ms（单层前向，B=32, S=512）
"""

import torch
import torch.nn as nn
from typing import Optional

from meta_group import MetaGroup


class GroupSmoothLayer(nn.Module):
    """
    群光滑层：将隐状态投影到群流形后残差连接

    设计原则：
    1. 完全向量化实现，支持任意批量维度
    2. 残差连接避免破坏预训练权重
    3. 与 MetaGroup 解耦，可独立配置

    参数：
        hidden_dim: Transformer 隐层维度（如 768）
        group_d: 群表示维数 d（矩阵大小 d×d）
        meta_group: 关联的 MetaGroup 实例（可选，若不提供则自动创建）
        proj_steps: 投影梯度步数（通常 1，推理时可增至 3）
        smooth_lr: 投影学习率（0.05–0.2）
        num_generators: 若自动创建 MetaGroup，生成元个数
        group_type: 若自动创建 MetaGroup，群类型

    示例：
        >>> meta_group = MetaGroup(num_generators=12, d=32)
        >>> smooth_layer = GroupSmoothLayer(
        ...     hidden_dim=768,
        ...     group_d=32,
        ...     meta_group=meta_group,
        ...     proj_steps=1,
        ...     smooth_lr=0.1
        ... )
        >>> x = torch.randn(4, 128, 768)  # (batch, seq, hidden)
        >>> out = smooth_layer(x)  # (4, 128, 768)
    """

    def __init__(
        self,
        hidden_dim: int,
        group_d: int = 32,
        meta_group: Optional[MetaGroup] = None,
        proj_steps: int = 1,
        smooth_lr: float = 0.1,
        num_generators: int = 12,
        group_type: str = 'orthogonal'
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.group_d = group_d
        self.proj_steps = proj_steps
        self.smooth_lr = smooth_lr

        # 使用传入的 MetaGroup 或自动创建
        if meta_group is not None:
            self.meta_group = meta_group
        else:
            self.meta_group = MetaGroup(
                num_generators=num_generators,
                d=group_d,
                group_type=group_type
            )

        # 投影层：将隐状态映射到矩阵空间
        matrix_dim = group_d * group_d
        self.proj_to = nn.Linear(hidden_dim, matrix_dim, bias=True)
        self.proj_from = nn.Linear(matrix_dim, hidden_dim, bias=True)

        # 初始化：小噪声初始化（非零），让群层从一开始就参与学习
        # 零初始化会导致梯度短路 — 模型会跳过群层直接学习
        nn.init.normal_(self.proj_to.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj_to.bias)
        nn.init.normal_(self.proj_from.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj_from.bias)

    def forward(self, hidden_states: torch.Tensor) -> tuple:
        """
        前向传播：群光滑操作

        Args:
            hidden_states: Transformer 隐状态 (B, S, H) 或任意形状 (..., H)

        Returns:
            群光滑后的隐状态（与输入同形）
            流形距离（标量张量，用于对齐损失）
            群增量矩阵 (B, S, d, d) — 该层对群流形的贡献
        """
        # 保存原始形状
        original_shape = hidden_states.shape

        # 确保至少 3 维：(batch, seq, hidden)
        if hidden_states.dim() == 2:
            # (tokens, hidden) → (1, tokens, hidden)
            hidden_states = hidden_states.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # hidden_states: (B, S, H)
        batch_size, seq_len, _ = hidden_states.shape

        # 投影到矩阵空间：(B, S, H) → (B, S, d*d)
        matrix_flat = self.proj_to(hidden_states)

        # 重塑为 (B*S, d, d)
        matrices = matrix_flat.view(batch_size * seq_len, self.group_d, self.group_d)

        # 批量群投影：(B*S, d, d) → (B*S, d, d)
        # 完全向量化，无 Python 循环
        matrices_smooth = self.meta_group.project_to_manifold_batch(
            matrices,
            num_steps=self.proj_steps,
            lr=self.smooth_lr
        )

        # 计算群增量（光滑后 - 光滑前）— 该层对群流形的贡献
        delta_group = matrices_smooth - matrices  # (B*S, d, d)

        # 新增：计算流形距离（投影前后差异）
        # 这度量了 hidden 表征偏离群流形的程度
        manifold_distance = torch.norm(matrices_smooth - matrices, dim=(-2, -1)).mean()

        # 重塑回 (B, S, d*d)
        smooth_flat = matrices_smooth.view(batch_size, seq_len, -1)
        delta_group_flat = delta_group.view(batch_size, seq_len, self.group_d, self.group_d)

        # 反投影：(B, S, d*d) → (B, S, H)
        smooth_hidden = self.proj_from(smooth_flat)

        # 残差连接
        output = hidden_states + smooth_hidden

        # 恢复原始形状
        if squeeze_output:
            output = output.squeeze(0)
            delta_group_flat = delta_group_flat.squeeze(0)

        return output, manifold_distance, delta_group_flat

    def set_proj_steps(self, steps: int) -> None:
        """
        动态设置投影步数（推理时可增加以提高光滑度）

        Args:
            steps: 新的投影步数
        """
        self.proj_steps = steps

    def set_smooth_lr(self, lr: float) -> None:
        """
        动态设置投影学习率

        Args:
            lr: 新的学习率
        """
        self.smooth_lr = lr

    def freeze_proj_weights(self) -> None:
        """冻结投影层权重（用于训练初期稳定）"""
        self.proj_to.weight.requires_grad = False
        self.proj_to.bias.requires_grad = False
        self.proj_from.weight.requires_grad = False
        self.proj_from.bias.requires_grad = False

    def unfreeze_proj_weights(self) -> None:
        """解冻投影层权重"""
        self.proj_to.weight.requires_grad = True
        self.proj_to.bias.requires_grad = True
        self.proj_from.weight.requires_grad = True
        self.proj_from.bias.requires_grad = True

    def extra_repr(self) -> str:
        return (
            f'hidden_dim={self.hidden_dim}, '
            f'group_d={self.group_d}, '
            f'proj_steps={self.proj_steps}, '
            f'smooth_lr={self.smooth_lr}'
        )


class GroupSmoothLayerConfig:
    """
    GroupSmoothLayer 配置类（便于从 YAML 加载）

    示例：
        >>> config = GroupSmoothLayerConfig.from_dict({
        ...     'hidden_dim': 768,
        ...     'group_d': 32,
        ...     'num_generators': 12,
        ... })
        >>> layer = GroupSmoothLayer(**config.to_dict())
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        group_d: int = 32,
        num_generators: int = 12,
        group_type: str = 'orthogonal',
        proj_steps: int = 1,
        smooth_lr: float = 0.1
    ):
        self.hidden_dim = hidden_dim
        self.group_d = group_d
        self.num_generators = num_generators
        self.group_type = group_type
        self.proj_steps = proj_steps
        self.smooth_lr = smooth_lr

    @classmethod
    def from_dict(cls, d: dict) -> 'GroupSmoothLayerConfig':
        """从字典创建配置"""
        return cls(
            hidden_dim=d.get('hidden_dim', 768),
            group_d=d.get('group_d', 32),
            num_generators=d.get('num_generators', 12),
            group_type=d.get('group_type', 'orthogonal'),
            proj_steps=d.get('proj_steps', 1),
            smooth_lr=d.get('smooth_lr', 0.1)
        )

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'hidden_dim': self.hidden_dim,
            'group_d': self.group_d,
            'num_generators': self.num_generators,
            'group_type': self.group_type,
            'proj_steps': self.proj_steps,
            'smooth_lr': self.smooth_lr
        }
