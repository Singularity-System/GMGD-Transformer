"""
动态群扩张控制器：协调路径积分分析与群注册表

核心功能：
1. 监控路径积分闭合误差
2. 当误差超过阈值时，触发 MetaGroup.check_expansion()
3. 扩张时同步扩展 GroupAttention

改进点：
- 使用 GroupRegistry（含 MetaGroup + GroupAttention）
- 用注意力代替路由器
- 用 meta_group.check_expansion 触发扩张
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from .group_registry import GroupRegistry
from .path_integral import PathIntegralAnalyzer, PathIntegralAnalysis


class DynamicGroupExpander(nn.Module):
    """
    动态群扩张控制器

    通过逆向路径积分分析，自动检测并创建新子群。

    参数：
        hidden_dim: Transformer 隐层维度
        initial_group_d: 初始群维度
        initial_num_generators: 初始生成元数量
        group_type: 群类型
        threshold: 路径积分闭合误差阈值
        max_generators_per_group: 单子群最大生成元数量
        min_steps_between_expansion: 两次扩张的最小间隔步数
    """

    def __init__(
        self,
        hidden_dim: int,
        initial_group_d: int = 16,
        initial_num_generators: int = 6,
        group_type: str = 'orthogonal',
        threshold: float = 0.5,
        max_generators_per_group: int = 12,
        min_steps_between_expansion: int = 50
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.initial_group_d = initial_group_d
        self.initial_num_generators = initial_num_generators
        self.group_type = group_type
        self.max_generators_per_group = max_generators_per_group

        # 群注册表（MetaGroup + GroupAttention）
        self.registry = GroupRegistry(
            hidden_dim=hidden_dim,
            initial_group_d=initial_group_d,
            initial_num_generators=initial_num_generators,
            group_type=group_type
        )

        # 路径积分分析器
        self.analyzer = PathIntegralAnalyzer(
            group_d=initial_group_d,
            threshold=threshold
        )

        # 扩张历史
        self.expansion_history: List[dict] = []

        # 控制参数
        self.min_steps_between_expansion = min_steps_between_expansion
        self.last_expansion_step = -self.min_steps_between_expansion * 2

    @property
    def meta_group(self) -> nn.Module:
        """获取元群"""
        return self.registry.meta_group

    @property
    def attention(self) -> nn.Module:
        """获取注意力"""
        return self.registry.attention

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_group_state: Optional[torch.Tensor] = None,
        global_step: int = 0,
        global_group_state: Optional[torch.Tensor] = None
    ) -> dict:
        """
        前向传播：扩张检测

        Args:
            hidden_states: 输入隐状态 (B, S, hidden_dim)
            target_group_state: 目标群状态 (B, d, d)
            global_group_state: 全局群状态 (B, d, d)，路径积分累乘结果
            global_step: 当前训练步数

        Returns:
            output: 包含扩张信息和路径积分误差
        """
        expansion_triggered = False
        expansion_info = None
        path_integral_error = None

        if target_group_state is not None and global_group_state is not None:
            analysis = self.analyzer(global_group_state, target_group_state)

            if analysis.needs_new_group:
                # 用 MetaGroup.check_expansion 触发扩张
                if self.meta_group.check_expansion(
                    path_integral_error=analysis.deviation_from_identity,
                    global_step=global_step
                ):
                    # 同步扩展注意力
                    self.attention.expand_groups(self.registry.num_groups)

                    expansion_info = {
                        'step': global_step,
                        'action': 'add_subgroup',
                        'num_subgroups': self.registry.num_groups,
                    }
                    expansion_triggered = True
                    self.last_expansion_step = global_step
                    self.expansion_history.append(expansion_info)

                    print(f"[DynamicGroupExpander] 触发子群扩张！当前子群数量：{self.registry.num_groups}")

            path_integral_error = analysis.deviation_from_identity

        return {
            'expansion_triggered': expansion_triggered,
            'expansion_info': expansion_info,
            'path_integral_error': path_integral_error,
            'global_step': global_step,
            'num_groups': self.registry.num_groups,
        }

    def get_statistics(self) -> dict:
        """获取统计信息"""
        return {
            'num_groups': self.registry.num_groups,
            'group_ids': self.registry.get_group_ids(),
            'path_integral_stats': self.analyzer.get_statistics(),
            'num_expansions': len(self.expansion_history),
            'last_expansion_step': self.last_expansion_step,
            'registry_stats': self.registry.get_statistics(),
        }

    def reset(self):
        """重置扩张器状态"""
        self.analyzer.reset()
        self.expansion_history.clear()
        self.last_expansion_step = -self.min_steps_between_expansion
