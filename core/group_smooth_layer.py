"""
GroupSmoothLayer: 多群注意力光滑层

用 per-token 注意力聚合多个子群的流形投影。

数据流（每子群独立）：
    hidden_states (B, S, H)
        → attention: (B, S, K) 每个 token 对每个子群的权重
        → 对每个子群 g:
            output_g, dist_g, delta_g = g.forward(hidden_states)
                # 内部: proj_to → 流形投影 → proj_from
        → weighted_output = Σ attention[t, g] * output_g[t]
        → output = hidden_states + alpha * weighted_output

注意力机制：
- Query 来自每个 token 的 hidden_states（per-token，不丢失时序）
- Key 来自每个子群的可学习参数
- Softmax 输出注意力权重（每个 token 对每个子群的软分配）
- 无需路由损失，注意力自然稀疏

性能优化：
- 每子群独立投影（各自的 proj_to/proj_from）
- 共享 attention 计算
- Alpha warm-up 保证训练初期不注入噪声
"""

import math
import torch
import torch.nn as nn
from typing import Optional

from .group_attention import GroupAttention


class GroupSmoothLayer(nn.Module):
    """
    多群注意力光滑层

    每个子群独立完成流形投影 + 反投影，通过 per-token 注意力加权聚合。

    参数：
        meta_group: MetaGroup 实例（管理多个 SubGroup）
        attention: GroupAttention 实例
        proj_steps: 每子群投影梯度步数
        smooth_lr: 投影基础学习率
    """

    def __init__(
        self,
        meta_group: nn.Module,
        attention: GroupAttention,
        proj_steps: int = 1,
        smooth_lr: float = 0.1,
    ):
        super().__init__()

        self.meta_group = meta_group
        self.attention = attention
        self.proj_steps = proj_steps
        self.smooth_lr = smooth_lr

        # Alpha warm-up 机制
        self.logit_alpha = nn.Parameter(torch.tensor(-8.6))
        self.register_buffer('warmup_steps', torch.tensor(2000))
        self.register_buffer('step_counter', torch.tensor(0))
        self.register_buffer('warmup_alpha', torch.tensor(0.1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        global_step: int = 0,
    ) -> tuple:
        """
        前向传播：多群注意力光滑

        Args:
            hidden_states: (B, S, H)
            global_step: 当前训练步数

        Returns:
            (output, manifold_distance, delta_group_flat)
            - output: (B, S, H) 光滑后隐状态
            - manifold_distance: 标量，平均流形距离
            - delta_group_flat: (B, S, d, d) 群增量
        """
        original_shape = hidden_states.shape

        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, seq_len, _ = hidden_states.shape

        # 1. 计算注意力权重 (B, S, K) — per token
        attn_weights = self.attention(hidden_states)  # (B, S, K)

        # 2. 每子群独立投影
        num_subgroups = len(self.meta_group.subgroups)
        outputs_list = []
        dists_list = []
        deltas_list = []

        for sg in self.meta_group.subgroups:
            out_g, dist_g, delta_g = sg.forward(
                hidden_states, num_steps=self.proj_steps, lr=self.smooth_lr
            )
            outputs_list.append(out_g)
            dists_list.append(dist_g)
            deltas_list.append(delta_g)

        # Stack: outputs (K, B, S, H), dists (K, N), deltas (K, N, d, d)
        all_outputs = torch.stack(outputs_list, dim=0)   # (K, B, S, H)
        all_dists = torch.stack(dists_list, dim=0)        # (K, N)
        all_deltas = torch.stack(deltas_list, dim=0)      # (K, N, d, d)

        # 3. Per-token 加权聚合
        # attn_weights: (B, S, K) → (K, B, S)
        attn_t = attn_weights.permute(2, 0, 1)  # (K, B, S)
        attn_t = attn_t / (attn_t.sum(dim=0, keepdim=True) + 1e-8)

        # weighted_output: (B, S, H)
        weighted_output = torch.zeros_like(hidden_states)
        for i in range(num_subgroups):
            weighted_output += attn_t[i].unsqueeze(-1) * all_outputs[i]

        # 4. 流形距离（注意力加权）
        # dists: (K, N), attn: (B, S, K) → reshape to (K, N)
        manifold_distance = (all_dists * attn_weights.reshape(num_subgroups, -1)).sum()

        # 5. 群增量（注意力加权聚合）
        weighted_delta = torch.zeros_like(all_deltas[0])  # (N, d, d)
        for i in range(num_subgroups):
            weighted_delta += attn_weights.reshape(num_subgroups, -1)[i].unsqueeze(1).unsqueeze(2) * all_deltas[i]

        # 6. Alpha warm-up 残差
        if self.training:
            with torch.no_grad():
                self.step_counter.add_(1)

        warmup_progress = (self.step_counter.float() / self.warmup_steps.float()).clamp(0.0, 1.0)
        learned_alpha = torch.sigmoid(self.logit_alpha)
        alpha = (1.0 - warmup_progress) * self.warmup_alpha + warmup_progress * learned_alpha

        output = hidden_states + alpha * weighted_output

        if squeeze_output:
            output = output.squeeze(0)
            weighted_delta = weighted_delta.squeeze(0)

        delta_group_flat = weighted_delta.view(batch_size, seq_len, self.meta_group.d, self.meta_group.d)

        return output, manifold_distance, delta_group_flat

    def set_proj_steps(self, steps: int) -> None:
        self.proj_steps = steps

    def set_smooth_lr(self, lr: float) -> None:
        self.smooth_lr = lr

    def extra_repr(self) -> str:
        warmup_progress = min(self.step_counter.item() / max(self.warmup_steps.item(), 1), 1.0)
        learned_alpha = torch.sigmoid(self.logit_alpha).item()
        effective_alpha = (1.0 - warmup_progress) * self.warmup_alpha.item() + warmup_progress * learned_alpha
        return (f'proj_steps={self.proj_steps}, smooth_lr={self.smooth_lr}, '
                f'num_subgroups={len(self.meta_group.subgroups) if self.meta_group else 0}, '
                f'alpha={effective_alpha:.3f}')
