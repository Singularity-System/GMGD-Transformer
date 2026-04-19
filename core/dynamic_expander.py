"""
动态群扩张控制器：协调路径积分分析与群注册表

核心功能：
1. 监控路径积分闭合误差
2. 当误差超过阈值时，触发代数相容性测试
3. 根据测试结果：吸收（扩展现有群）或 分裂（创建新群）
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
    """

    def __init__(
        self,
        hidden_dim: int,
        initial_group_d: int = 16,
        initial_num_generators: int = 6,
        group_type: str = 'orthogonal',
        threshold: float = 0.5,
        max_generators_per_group: int = 12
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.initial_group_d = initial_group_d
        self.initial_num_generators = initial_num_generators
        self.group_type = group_type
        self.max_generators_per_group = max_generators_per_group

        # 群注册表
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
        self.compatibility_threshold = 0.1  # 代数相容性阈值
        self.min_steps_between_expansion = 50  # 两次扩张的最小间隔步数（降低以允许早期扩张）
        self.last_expansion_step = -self.min_steps_between_expansion * 2  # 初始延迟更长

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_group_state: Optional[torch.Tensor] = None,
        global_step: int = 0,
        global_group_state: Optional[torch.Tensor] = None  # 新增：路径积分累乘结果
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
        # 1. 获取当前路由结果
        router_output = self.registry(hidden_states, target_group_state)

        # 2. 如果提供目标状态，分析路径积分闭合
        expansion_triggered = False
        expansion_info = None

        if target_group_state is not None and global_group_state is not None:
            # 使用路径积分累乘结果进行分析
            G_final = global_group_state

            # 分析闭合误差
            analysis = self.analyzer(G_final, target_group_state)

            # 判断是否需要扩张
            if analysis.needs_new_group:
                # 检查间隔
                if global_step - self.last_expansion_step >= self.min_steps_between_expansion:
                    # 执行扩张流程
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

        # 1. 获取现有生成元列表
        existing_generators = []
        for group_id, group in self.registry.groups.items():
            existing_generators.extend(group.all_generators().unbind(0))

        # 2. 测试代数相容性
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
            # 吸收：将候选生成元加入现有群（如果还有空间）
            # 找到生成元最少的群
            target_group_id = min(
                self.registry.groups.keys(),
                key=lambda gid: self.registry.group_metadata[gid]['num_generators']
            )
            target_group = self.registry.groups[target_group_id]

            # 检查是否还能添加生成元
            current_num_generators = len(target_group.generator_params)
            if current_num_generators < self.max_generators_per_group:
                # 吸收逻辑：扩展生成元参数
                self._absorb_generator(target_group, candidate_generator)
                expansion_info['action'] = 'absorb'
                expansion_info['target_group'] = target_group_id
                print(f"[DynamicGroupExpander] 吸收：将候选生成元加入群 {target_group_id}")
            else:
                # 群已满，创建新群
                expansion_info['action'] = 'split'
                self._create_new_group(candidate_generator, global_step)
                print(f"[DynamicGroupExpander] 分裂：创建新群（群 {target_group_id} 已满）")
        else:
            # 分裂：创建新群
            expansion_info['action'] = 'split'
            self._create_new_group(candidate_generator, global_step)
            print(f"[DynamicGroupExpander] 分裂：候选生成元与现有群不相容，创建新群")

        # 更新最后扩张时间
        self.last_expansion_step = global_step

        # 记录历史
        self.expansion_history.append(expansion_info)

        return expansion_info

    def _absorb_generator(self, group: nn.Module, candidate: torch.Tensor):
        """吸收候选生成元到现有群"""
        d = candidate.shape[0]

        # 创建新的生成元参数
        new_param = nn.Parameter(candidate.clone())

        # 扩展 generator_params
        old_params = group.generator_params
        new_params = nn.Parameter(torch.cat([
            old_params.data,
            new_param.unsqueeze(0)
        ], dim=0))

        group.generator_params = new_params

        # 更新元数据
        group_id = [gid for gid, g in self.registry.groups.items() if g == group][0]
        self.registry.group_metadata[group_id]['num_generators'] += 1

    def _create_new_group(self, candidate: torch.Tensor, global_step: int):
        """创建新群"""
        d = candidate.shape[0]

        # 生成新群 ID
        new_group_id = f"group_{self.registry.num_groups}"

        # 以候选生成元为核心初始化新群
        # 简化实现：使用候选矩阵作为第一个生成元的基础
        num_generators = 1  # 新群初始只有 1 个生成元

        # 注册新群（会自动扩展路由器）
        self.registry.register_new_group(
            group_id=new_group_id,
            group_d=d,
            num_generators=num_generators
        )

        # 初始化新群的生成元
        new_group = self.registry.groups[new_group_id]

        # 使用候选矩阵初始化第一个生成元（正交化）
        # 通过 QR 分解获得正交矩阵
        try:
            Q, R = torch.linalg.qr(candidate)
            new_group.generator_params.data[0] = Q
        except:
            # QR 失败，使用反对称化
            skew = 0.5 * (candidate - candidate.transpose(-2, -1))
            new_group.generator_params.data[0] = skew / (torch.norm(skew) + 1e-6) * 0.5

        self.registry.group_metadata[new_group_id]['created_at_step'] = global_step

    def get_statistics(self) -> dict:
        """获取统计信息"""
        return {
            'num_groups': self.registry.num_groups,
            'group_ids': self.registry.get_group_ids(),
            'path_integral_stats': self.analyzer.get_statistics(),
            'num_expansions': len(self.expansion_history),
            'last_expansion_step': self.last_expansion_step
        }

    def reset(self):
        """重置扩张器状态"""
        self.analyzer.reset()
        self.expansion_history.clear()
        self.last_expansion_step = -self.min_steps_between_expansion
