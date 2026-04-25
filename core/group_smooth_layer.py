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

import math
import torch
import torch.nn as nn
from typing import Optional

from .meta_group import MetaGroup


class GroupSmoothLayer(nn.Module):
    """
    群光滑层：将隐状态投影到群流形后残差连接（带自适应机制）

    设计原则：
    1. 完全向量化实现，支持任意批量维度
    2. 残差连接避免破坏预训练权重
    3. 与 MetaGroup 解耦，可独立配置
    4. **自适应学习**: 根据流形距离动态调整投影强度和残差权重

    自适应机制：
    - 可学习的残差权重 α = sigmoid(weight_alpha)，控制群光滑影响
    - 自适应学习率：lr_adaptive = base_lr * (1 + manifold_distance)
    - 门控投影：根据输入特征动态调整投影强度

    参数：
        hidden_dim: Transformer 隐层维度（如 768）
        group_d: 群表示维数 d（矩阵大小 d×d）
        meta_group: 关联的 MetaGroup 实例（可选，若不提供则自动创建）
        proj_steps: 投影梯度步数（通常 1，推理时可增至 3）
        smooth_lr: 投影基础学习率（0.05–0.2）
        num_generators: 若自动创建 MetaGroup，生成元个数
        group_type: 若自动创建 MetaGroup，群类型
        adaptive: 是否启用自适应机制（默认 True）

    示例：
        >>> meta_group = MetaGroup(num_generators=12, d=32)
        >>> smooth_layer = GroupSmoothLayer(
        ...     hidden_dim=768,
        ...     group_d=32,
        ...     meta_group=meta_group,
        ...     proj_steps=1,
        ...     smooth_lr=0.1,
        ...     adaptive=True
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
        group_type: str = 'orthogonal',
        adaptive: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.group_d = group_d
        self.proj_steps = proj_steps
        self.smooth_lr = smooth_lr
        self.adaptive = adaptive

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

        # 投影层初始化策略：
        # - proj_to: Kaiming 初始化，gain=0.1（小但不为零）
        #   提供非零初始梯度信号，同时避免 alpha=1 时激活爆炸
        #   gain=0.1 时投影范数 ≈ 0.92（旧方案 std=0.02 范数=8.87 的 1/10）
        #   每层注入噪声 ≈ 0.05，12 层总计 ≈ 0.6，远小于预训练激活
        # - proj_from: 零初始化，避免群路径在训练初期注入噪声
        #   零初始化时 proj_from 仍可从流形损失获得非零梯度
        # - alpha warm-up: 初始 alpha=1.0 提供强梯度信号
        #   配合小 gain 的 proj_to，激活不会爆炸
        nn.init.kaiming_uniform_(self.proj_to.weight, a=math.sqrt(5))
        with torch.no_grad():
            self.proj_to.weight.mul_(0.1)  # 缩小到 1/10，避免 warm-up 时激活爆炸
        nn.init.zeros_(self.proj_to.bias)
        nn.init.zeros_(self.proj_from.weight)
        nn.init.zeros_(self.proj_from.bias)

        # === 自适应机制 ===
        if self.adaptive:
            # Alpha warm-up 机制：
            # - 训练初期 (step < warmup_steps): alpha=0.2（提供足够梯度信号，
            #   同时限制噪声注入避免激活爆炸）
            # - 训练后期 (step >= warmup_steps): alpha 由模型自行学习
            #   sigmoid(logit_alpha) 通常收敛到合理值
            #
            # 为什么需要 warm-up:
            # proj_to 使用缩放 Kaiming 初始化 (范数 ≈ 0.92)，提供非零初始投影。
            # proj_from 零初始化，从流形损失获得非零梯度。
            # alpha=0.2 平衡：提供足够梯度信号推动学习，同时限制噪声注入。
            #
            # 旧方案 (random init std=0.02, alpha=0.5):
            #   投影层范数锁定在 8.87，群层变成随机噪声
            # 新方案 (scaled Kaiming, alpha=0.2 warmup):
            #   投影层可学习，群层从静默逐步激活
            self.logit_alpha = nn.Parameter(torch.tensor(-8.6))  # sigmoid(-8.6) ≈ 0.0002
            self.register_buffer('warmup_steps', torch.tensor(2000))  # 延长 warm-up 步数
            self.register_buffer('step_counter', torch.tensor(0))
            self.register_buffer('warmup_alpha', torch.tensor(0.1))  # warm-up 阶段 alpha（控制噪声）

            # 门控网络：根据输入特征动态调整投影强度
            # 输入：隐状态的统计特征 (mean, std)，输出：门控值 g ∈ (0, 1)
            self.gate_network = nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )

    def forward(self, hidden_states: torch.Tensor) -> tuple:
        """
        前向传播：自适应群光滑操作（带 Alpha Warm-up）

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

        # === 自适应机制 ===
        if self.adaptive:
            # 1. 裁剪输入矩阵避免数值溢出（Frobenius 范数限制）
            matrix_norms = torch.linalg.norm(matrices, dim=(-2, -1), keepdim=True)
            max_norm = 100.0
            scaling = torch.clamp(matrix_norms, max=max_norm) / (matrix_norms + 1e-8)
            matrices = matrices * scaling

            # 2. 门控网络
            input_mean = matrices.mean(dim=(-2, -1), keepdim=True)
            input_std = matrices.std(dim=(-2, -1), keepdim=True)
            input_mean_norm = input_mean / (input_std + 1e-6)
            gate_input = torch.cat([
                input_mean_norm.view(batch_size * seq_len, 1),
                input_std.view(batch_size * seq_len, 1)
            ], dim=-1)
            gate_value = self.gate_network(gate_input).view(batch_size * seq_len, 1, 1)
            matrices_gated = gate_value * matrices

            # 3. 自适应学习率（限制上限防止爆炸）
            base_manifold_dist = torch.norm(matrices, dim=(-2, -1)).mean().detach()
            adaptive_lr_factor = 1.0 + base_manifold_dist
            adaptive_lr_factor = torch.clamp(adaptive_lr_factor, min=1.0, max=50.0)

        else:
            matrices_gated = matrices
            adaptive_lr_factor = 1.0

        # 批量群投影：(B*S, d, d) → (B*S, d, d)
        matrices_smooth = self.meta_group.project_to_manifold_batch(
            matrices_gated,
            num_steps=self.proj_steps,
            lr=self.smooth_lr * adaptive_lr_factor
        )

        # 计算群增量（光滑后 - 光滑前）— 该层对群流形的贡献
        delta_group = matrices_smooth - matrices_gated

        # 流形距离（投影前后差异，度量 hidden 表征偏离群流形的程度）
        manifold_distance = torch.norm(matrices_smooth - matrices_gated, dim=(-2, -1)).mean()

        # 重塑回 (B, S, d*d)
        smooth_flat = matrices_smooth.view(batch_size, seq_len, -1)
        delta_group_flat = delta_group.view(batch_size, seq_len, self.group_d, self.group_d)

        # 反投影：(B, S, d*d) → (B, S, H)
        smooth_hidden = self.proj_from(smooth_flat)

        # === Alpha Warm-up 残差连接 ===
        if self.adaptive:
            # Alpha warm-up 机制：
            # - 训练初期 (step < warmup_steps): alpha=0.2（适中信号）
            #   帮助 proj_to/proj_from 快速逃离零初始化状态
            # - 训练后期 (step >= warmup_steps): alpha 由模型自行学习
            #   sigmoid(logit_alpha) 通常收敛到合理值
            #
            # 为什么需要 warm-up:
            # proj_to 使用 He 初始化提供非零初始投影，proj_from 零初始化。
            # 训练初期 alpha=1 提供强梯度信号：
            #   dL/dW_proj_from ∝ alpha * (matrices_smooth - matrices_gated)  ← 非零
            #   dL/dW_proj_to   ∝ alpha * dL/dW_proj_from * gate * ...       ← 非零
            # 随着生成元学习群公理，流形距离提供额外信号。
            if self.training:
                with torch.no_grad():
                    self.step_counter.add_(1)

            warmup_progress = (self.step_counter.float() / self.warmup_steps.float()).clamp(0.0, 1.0)
            learned_alpha = torch.sigmoid(self.logit_alpha)
            # warm-up 阶段：alpha 从 warmup_alpha (0.2) 线性衰减到 learned_alpha
            alpha = (1.0 - warmup_progress) * self.warmup_alpha + warmup_progress * learned_alpha

            output = hidden_states + alpha * smooth_hidden
        else:
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
        repr_str = (
            f'hidden_dim={self.hidden_dim}, '
            f'group_d={self.group_d}, '
            f'proj_steps={self.proj_steps}, '
            f'smooth_lr={self.smooth_lr}'
        )
        if self.adaptive:
            warmup_progress = min(self.step_counter.item() / max(self.warmup_steps.item(), 1), 1.0)
            learned_alpha = torch.sigmoid(self.logit_alpha).item()
            warmup_a = self.warmup_alpha.item()
            effective_alpha = (1.0 - warmup_progress) * warmup_a + warmup_progress * learned_alpha
            repr_str += f', adaptive=True, alpha={effective_alpha:.3f}, learned={learned_alpha:.3f}, warmup={warmup_progress:.1%}'
        return repr_str


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
