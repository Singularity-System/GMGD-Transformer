"""
全局群注册表：管理多个 MetaGroup，支持动态扩张

核心功能：
1. 管理多个 MetaGroup 实例（通过 SubGroupManager）
2. 多尺度上下文路由器（通过 DomainRouter）
3. 动态扩张：当检测到新域时，创建新群并扩展路由器
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from .meta_group import MetaGroup
from .subgroup_manager import SubGroupManager
from .domain_router import DomainRouter


class GroupRegistry(nn.Module):
    """
    全局群注册表：组合 SubGroupManager + DomainRouter

    设计原则：
    1. 所有层共享同一个注册表
    2. 路由器根据多尺度特征动态选择群
    3. 支持动态添加新群
    4. 向后兼容旧的单层 Linear 路由器接口

    参数：
        hidden_dim: Transformer 隐层维度
        initial_group_d: 初始群表示维度
        initial_num_generators: 初始生成元个数
        group_type: 群类型
        router_window_size: 路由器局部窗口大小
        router_mlp_hidden: 路由器 MLP 隐藏维度
    """

    def __init__(
        self,
        hidden_dim: int,
        initial_group_d: int = 16,
        initial_num_generators: int = 6,
        group_type: str = 'orthogonal',
        router_window_size: int = 16,
        router_mlp_hidden: int = 64
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.group_type = group_type

        # 子群管理器
        self.subgroup_manager = SubGroupManager(
            hidden_dim=hidden_dim,
            initial_group_d=initial_group_d,
            initial_num_generators=initial_num_generators,
            group_type=group_type
        )

        # 多尺度路由器
        self.router = DomainRouter(
            hidden_dim=hidden_dim,
            num_groups=1,  # 初始只有 1 个群
            window_size=router_window_size,
            mlp_hidden=router_mlp_hidden
        )

        # 路径积分闭合监控阈值
        self.path_integral_threshold = 0.5

    # === 向后兼容接口 ===

    @property
    def groups(self) -> nn.ModuleDict:
        """向后兼容：直接访问群字典"""
        return self.subgroup_manager.groups

    @property
    def group_metadata(self) -> Dict[str, dict]:
        """向后兼容：直接访问群元数据"""
        return self.subgroup_manager.group_metadata

    @property
    def num_groups(self) -> int:
        """返回当前注册的群数量"""
        return self.subgroup_manager.num_groups

    def get_group_ids(self) -> List[str]:
        """返回所有群 ID"""
        return self.subgroup_manager.get_all_group_ids()

    def generate_next_id(self) -> str:
        """生成下一个可用的群 ID"""
        return self.subgroup_manager.generate_next_id()

    def _register_group(self, group_id: str, group_d: int, num_generators: int):
        """内部方法：注册一个新群（向后兼容）"""
        return self.subgroup_manager.register_group(group_id, group_d, num_generators)

    def register_new_group(
        self,
        group_id: str,
        group_d: int,
        num_generators: int,
        generator_init: Optional[torch.Tensor] = None
    ):
        """
        注册一个新群（带初始化矩阵）

        Args:
            group_id: 群 ID
            group_d: 群维度
            num_generators: 生成元数量
            generator_init: 可选的初始生成元矩阵 (num_generators, d, d)
        """
        self.subgroup_manager.register_group(group_id, group_d, num_generators, generator_init)

        # 扩展路由器
        self.router.expand_router(self.num_groups)

        print(f"[GroupRegistry] 注册新群：{group_id} (d={group_d}, k={num_generators})")

    # === 路由接口 ===

    def get_router_probs(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        获取路由概率（向后兼容）

        Args:
            hidden_states: 隐状态 (B, S, hidden_dim)

        Returns:
            路由概率分布 (B, S, num_groups)
        """
        probs, indices, ids = self.router(hidden_states, force_hard=True)
        # 去掉 fallback 维度
        return probs[..., :-1]

    def select_groups(
        self,
        hidden_states: torch.Tensor,
        top_k: int = 1
    ) -> Tuple[List[str], torch.Tensor]:
        """
        为每个 batch 选择激活的群

        Returns:
            selected_group_ids: 选中的群 ID 列表
            router_probs: 路由概率 (B, S, num_groups)
        """
        probs, indices, ids = self.router(hidden_states, force_hard=not self.training)
        # 去掉 fallback 维度
        task_probs = probs[..., :-1]
        return ids, task_probs

    # === 前向传播 ===

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_group_state: Optional[torch.Tensor] = None,
        global_group_state: Optional[torch.Tensor] = None
    ) -> dict:
        """
        前向传播：路由 + 群信息

        Args:
            hidden_states: 输入隐状态 (B, S, hidden_dim)
            target_group_state: 目标群状态 (B, d, d)，用于路径积分闭合误差
            global_group_state: 全局群状态 (B, d, d)，路径积分累乘结果

        Returns:
            output: 路由结果 + 群信息字典
        """
        B, S, H = hidden_states.shape

        # 路由决策
        probs, indices, selected_group_ids = self.router(hidden_states, force_hard=not self.training)
        # 处理 fallback 情况（使用 group_0 作为默认）
        selected_group_id = selected_group_ids[0] if selected_group_ids else 'group_0'
        if selected_group_id == 'fallback' or selected_group_id not in self.groups:
            selected_group_id = 'group_0'

        # 获取对应的群
        group = self.subgroup_manager.get_group(selected_group_id)

        # 更新群激活计数
        self.group_metadata[selected_group_id]['activation_count'] += B * S

        # 路由辅助损失
        router_loss = self.router.compute_router_loss(probs)
        load_balance_loss = self.router.compute_load_balance_loss(probs)
        total_router_loss = router_loss + 0.1 * load_balance_loss

        # 路径积分闭合误差
        path_integral_error = None
        if target_group_state is not None and global_group_state is not None:
            path_integral_error = self._compute_path_integral_error(
                global_group_state, target_group_state, selected_group_id
            )

        return {
            'selected_group_id': selected_group_id,
            'group': group,
            'router_probs': probs[..., :-1],  # 去掉 fallback
            'router_loss': total_router_loss,
            'router_entropy': router_loss,
            'load_balance_loss': load_balance_loss,
            'path_integral_error': path_integral_error,
        }

    def _compute_path_integral_error(
        self,
        G_final: torch.Tensor,
        G_target: torch.Tensor,
        group_id: str
    ) -> torch.Tensor:
        """计算路径积分闭合误差"""
        try:
            G_final_inv = torch.linalg.inv(G_final)
            delta = G_final_inv @ G_target
            I = torch.eye(G_final.shape[-1], device=delta.device).unsqueeze(0).expand(G_final.shape[0], -1, -1)
            return torch.norm(delta - I, dim=(-2, -1)).mean()
        except RuntimeError:
            return torch.tensor(0.0, device=G_final.device)

    # === 路径积分分析 ===

    def analyze_path_integral(self, G_final: torch.Tensor, G_target: torch.Tensor) -> dict:
        """
        分析路径积分闭合状态，为可能的新群创建提供依据
        """
        B, d, _ = G_final.shape

        try:
            G_final_inv = torch.linalg.inv(G_final)
        except RuntimeError:
            G_final_inv = torch.linalg.pinv(G_final)

        delta_missing = G_final_inv @ G_target

        I = torch.eye(d, device=G_final.device).unsqueeze(0).expand(B, -1, -1)
        deviation = torch.norm(delta_missing - I, dim=(-2, -1)).mean().item()

        symmetric_part = 0.5 * (delta_missing + delta_missing.transpose(-2, -1))
        skew_symmetric_part = 0.5 * (delta_missing - delta_missing.transpose(-2, -1))

        symmetric_norm = torch.norm(symmetric_part, dim=(-2, -1)).mean().item()
        skew_symmetric_norm = torch.norm(skew_symmetric_part, dim=(-2, -1)).mean().item()
        det = torch.linalg.det(delta_missing).abs().mean().item()

        return {
            'deviation_from_identity': deviation,
            'symmetric_norm': symmetric_norm,
            'skew_symmetric_norm': skew_symmetric_norm,
            'determinant': det,
            'delta_missing_mean': delta_missing.mean().item(),
            'needs_new_group': deviation > self.path_integral_threshold,
            'candidate_generator': delta_missing.mean(dim=0).detach()
        }

    def get_statistics(self) -> dict:
        """获取所有群的统计信息"""
        return {
            'num_groups': self.num_groups,
            'group_ids': self.get_group_ids(),
            'subgroup_stats': self.subgroup_manager.get_statistics(),
            'router': self.router.extra_repr(),
        }
