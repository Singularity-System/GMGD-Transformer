"""
全局群注册表：组合 MetaGroup + GroupAttention

核心功能：
1. 管理单个 MetaGroup（内含多个 SubGroup）
2. 管理共享的 GroupAttention
3. 动态扩张：添加新 SubGroup 时同步扩展注意力
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from .meta_group import MetaGroup
from .group_attention import GroupAttention


class GroupRegistry(nn.Module):
    """
    全局群注册表：组合 MetaGroup + GroupAttention

    设计原则：
    1. 所有层共享同一个注册表
    2. GroupAttention 是共享的（per-token query）
    3. MetaGroup 管理 SubGroup 索引和扩张

    参数：
        hidden_dim: Transformer 隐层维度
        initial_group_d: 初始群表示维度
        initial_num_generators: 初始生成元个数
        group_type: 群类型
    """

    def __init__(
        self,
        hidden_dim: int,
        initial_group_d: int = 16,
        initial_num_generators: int = 6,
        group_type: str = 'orthogonal',
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.group_type = group_type

        # 元群（管理多个子群）
        self.meta_group = MetaGroup(
            d=initial_group_d,
            group_type=group_type,
            hidden_dim=hidden_dim,
            initial_subgroups=1,
            initial_generators_per_subgroup=initial_num_generators,
        )

        # 共享注意力
        self.attention = GroupAttention(
            hidden_dim=hidden_dim,
            num_groups=1,
        )

    # === 向后兼容接口 ===

    @property
    def groups(self) -> nn.ModuleDict:
        """向后兼容：返回子群 ModuleDict"""
        result = nn.ModuleDict()
        for i, sg in enumerate(self.meta_group.subgroups):
            result[f"group_{i}"] = sg
        return result

    @property
    def group_metadata(self) -> Dict[str, dict]:
        """向后兼容：返回子群元数据"""
        return {
            f"group_{i}": {
                'group_d': self.meta_group.d,
                'num_generators': sg.num_generators,
            }
            for i, sg in enumerate(self.meta_group.subgroups)
        }

    @property
    def num_groups(self) -> int:
        return len(self.meta_group.subgroups)

    def get_group_ids(self) -> List[str]:
        return [f"group_{i}" for i in range(self.num_groups)]

    def generate_next_id(self) -> str:
        return f"group_{self.num_groups}"

    def register_new_group(
        self,
        group_id: str,
        group_d: int,
        num_generators: int,
        generator_init: Optional[torch.Tensor] = None,
    ):
        """
        注册新子群（同步扩展注意力）

        Args:
            group_id: 子群 ID（仅用于日志）
            group_d: 群维度
            num_generators: 生成元数量
            generator_init: 可选的初始生成元
        """
        self.meta_group._add_subgroup(num_generators)
        self.attention.expand_groups(self.num_groups)

        print(f"[GroupRegistry] 注册新子群：{group_id} (d={group_d}, k={num_generators})")

    # === 路由接口（向后兼容，新架构下不使用） ===

    def get_router_probs(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """获取注意力权重（向后兼容接口）"""
        return self.attention(hidden_states)

    def select_groups(
        self,
        hidden_states: torch.Tensor,
        top_k: int = 1,
    ) -> Tuple[List[str], torch.Tensor]:
        """
        为每个 token 选择主导子群

        Returns:
            selected_group_ids: 被选中的子群 ID 列表
            router_probs: 注意力概率 (B, S, num_groups)
        """
        probs = self.attention(hidden_states)
        ids = probs.argmax(dim=-1)
        selected_ids = []
        for i in range(self.num_groups):
            if (ids == i).any():
                selected_ids.append(f"group_{i}")
        return selected_ids, probs

    # === 前向传播（新架构下不使用，保留向后兼容） ===

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_group_state: Optional[torch.Tensor] = None,
        global_group_state: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        前向传播：注意力权重 + 群信息（向后兼容）
        """
        B, S, H = hidden_states.shape

        probs = self.attention(hidden_states)

        path_integral_error = None
        if target_group_state is not None and global_group_state is not None:
            path_integral_error = self._compute_path_integral_error(
                global_group_state, target_group_state
            )

        return {
            'router_probs': probs,
            'path_integral_error': path_integral_error,
        }

    def _compute_path_integral_error(
        self,
        G_final: torch.Tensor,
        G_target: torch.Tensor,
    ) -> torch.Tensor:
        """计算路径积分闭合误差"""
        try:
            G_final_inv = torch.linalg.inv(G_final)
            delta = G_final_inv @ G_target
            I = torch.eye(G_final.shape[-1], device=delta.device).unsqueeze(0).expand(G_final.shape[0], -1, -1)
            return torch.norm(delta - I, dim=(-2, -1)).mean()
        except RuntimeError:
            return torch.tensor(0.0, device=G_final.device)

    def analyze_path_integral(self, G_final: torch.Tensor, G_target: torch.Tensor) -> dict:
        """分析路径积分闭合状态"""
        B, d, _ = G_final.shape

        try:
            G_final_inv = torch.linalg.inv(G_final)
        except RuntimeError:
            G_final_inv = torch.linalg.pinv(G_final)

        delta_missing = G_final_inv @ G_target

        I = torch.eye(d, device=G_final.device).unsqueeze(0).expand(B, -1, -1)
        deviation = torch.norm(delta_missing - I, dim=(-2, -1)).mean().item()

        return {
            'deviation_from_identity': deviation,
            'needs_new_group': deviation > self.meta_group.expansion_threshold,
            'candidate_generator': delta_missing.mean(dim=0).detach(),
        }

    def get_statistics(self) -> dict:
        """获取所有子群的统计信息"""
        return {
            'num_groups': self.num_groups,
            'group_ids': self.get_group_ids(),
            'attention': self.attention.extra_repr(),
            'meta_group': self.meta_group.extra_repr(),
        }
