"""
动态群扩张控制器：协调路径积分分析与群注册表

核心功能：
1. 监控路径积分闭合误差
2. 当误差超过阈值时，触发代数相容性测试
3. 根据测试结果：吸收（扩展现有群）或 分裂（创建新群）

改进点：
- 使用重构后的 GroupRegistry（含 SubGroupManager + DomainRouter）
- 改进扩张触发逻辑：结合路径积分误差和任务性能
- 新增群初始化时使用 QR 分解或反对称化确保正交性
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from .group_registry import GroupRegistry
from .path_integral import PathIntegralAnalyzer, PathIntegralAnalysis


class DynamicGroupExpander(nn.Module):
    """
    动态群扩张控制器

    通过逆向路径积分分析，自动检测并创建新群。

    参数：
        hidden_dim: Transformer 隐层维度
        initial_group_d: 初始群维度
        initial_num_generators: 初始生成元数量
        group_type: 群类型
        threshold: 路径积分闭合误差阈值
        max_generators_per_group: 单群最大生成元数量
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

        # 群注册表（已包含 SubGroupManager + DomainRouter）
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
        self.compatibility_threshold = 0.1
        self.min_steps_between_expansion = min_steps_between_expansion
        self.last_expansion_step = -self.min_steps_between_expansion * 2

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_group_state: Optional[torch.Tensor] = None,
        global_step: int = 0,
        global_group_state: Optional[torch.Tensor] = None
    ) -> dict:
        """
        前向传播：路由 + 可能的群扩张

        Args:
            hidden_states: 输入隐状态 (B, S, hidden_dim)
            target_group_state: 目标群状态 (B, d, d)
            global_group_state: 全局群状态 (B, d, d)，路径积分累乘结果
            global_step: 当前训练步数

        Returns:
            output: 包含路由结果和扩张信息
        """
        # 路由结果
        router_output = self.registry(
            hidden_states,
            target_group_state=target_group_state,
            global_group_state=global_group_state
        )

        # 扩张检测
        expansion_triggered = False
        expansion_info = None

        if target_group_state is not None and global_group_state is not None:
            analysis = self.analyzer(global_group_state, target_group_state)

            if analysis.needs_new_group:
                if global_step - self.last_expansion_step >= self.min_steps_between_expansion:
                    expansion_info = self._attempt_expansion(
                        analysis.candidate_generator,
                        global_step
                    )
                    expansion_triggered = True

        return {
            **router_output,
            'expansion_triggered': expansion_triggered,
            'expansion_info': expansion_info,
            'path_integral_error': router_output.get('path_integral_error'),
            'global_step': global_step,
            'num_groups': self.registry.num_groups
        }

    def _attempt_expansion(
        self,
        candidate_generator: torch.Tensor,
        global_step: int
    ) -> dict:
        """
        尝试群扩张：吸收或分裂

        Args:
            candidate_generator: 候选生成元矩阵 (d, d)
            global_step: 当前步数

        Returns:
            expansion_info: 扩张信息
        """
        d = candidate_generator.shape[0]
        device = candidate_generator.device

        # 获取现有生成元列表
        existing_generators = []
        for group_id, group in self.registry.groups.items():
            existing_generators.extend(group.all_generators().unbind(0))

        # 测试代数相容性
        compatible, relation_loss = self.analyzer.test_algebraic_compatibility(
            candidate_generator,
            existing_generators,
            relations=['commute', 'orthogonal']
        )

        expansion_info = {
            'step': global_step,
            'candidate_norm': torch.norm(candidate_generator).item(),
            'relation_loss': relation_loss,
            'compatible': compatible,
            'action': None
        }

        if compatible:
            # 吸收：加入生成元最少的群
            target_group_id = min(
                self.registry.groups.keys(),
                key=lambda gid: self.registry.group_metadata[gid]['num_generators']
            )
            target_group = self.registry.groups[target_group_id]
            current_n = len(target_group.generator_params)

            if current_n < self.max_generators_per_group:
                self._absorb_generator(target_group, candidate_generator)
                expansion_info['action'] = 'absorb'
                expansion_info['target_group'] = target_group_id
                print(f"[DynamicGroupExpander] 吸收候选生成元到 {target_group_id}")
            else:
                expansion_info['action'] = 'split'
                self._create_new_group(candidate_generator, global_step)
                print(f"[DynamicGroupExpander] 分裂：群已满，创建新群")
        else:
            expansion_info['action'] = 'split'
            self._create_new_group(candidate_generator, global_step)
            print(f"[DynamicGroupExpander] 分裂：候选与现有群不相容，创建新群")

        self.last_expansion_step = global_step
        self.expansion_history.append(expansion_info)
        return expansion_info

    def _absorb_generator(self, group: nn.Module, candidate: torch.Tensor):
        """吸收候选生成元到现有群"""
        new_param = nn.Parameter(candidate.clone())
        old_params = group.generator_params
        new_params = nn.Parameter(torch.cat([old_params.data, new_param.unsqueeze(0)], dim=0))
        group.generator_params = new_params

        group_id = [gid for gid, g in self.registry.groups.items() if g == group][0]
        self.registry.group_metadata[group_id]['num_generators'] += 1

    def _create_new_group(self, candidate: torch.Tensor, global_step: int):
        """创建新群（带正交初始化）"""
        d = candidate.shape[0]
        new_group_id = f"group_{self.registry.num_groups}"
        num_generators = 1

        # 注册新群（自动扩展路由器）
        self.registry.register_new_group(
            group_id=new_group_id,
            group_d=d,
            num_generators=num_generators
        )

        # 正交初始化
        new_group = self.registry.groups[new_group_id]
        try:
            Q, _ = torch.linalg.qr(candidate)
            new_group.generator_params.data[0] = Q
        except RuntimeError:
            skew = 0.5 * (candidate - candidate.transpose(-2, -1))
            new_group.generator_params.data[0] = skew / (skew.norm() + 1e-6) * 0.5

        self.registry.group_metadata[new_group_id]['created_at_step'] = global_step

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
