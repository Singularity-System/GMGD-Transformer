"""
MetaGroup: 管理群

继承 Group（纯群操作），管理多个 TGroup 实例。
核心功能：索引 + 扩张。
"""

import torch
import torch.nn as nn

from .group import Group
from .tgroup import TGroup


class MetaGroup(Group, nn.Module):
    """管理多个 TGroup 实例

    参数：
        d: 群表示维度
        hidden_dim: Transformer 隐层维度
        expansion_threshold: 误差阈值
        initial_active_groups: 初始活跃群数量（不含透明群）
        num_generators_per_group: 每个活跃群的生成元数
    """

    def __init__(
        self,
        d: int = 16,
        hidden_dim: int = 768,
        expansion_threshold: float = 0.5,
        initial_active_groups: int = 1,
        num_generators_per_group: int = 6,
    ):
        Group.__init__(self, d=d, num_generators=1)
        nn.Module.__init__(self)

        self.hidden_dim = hidden_dim
        self.expansion_threshold = expansion_threshold
        self.last_expansion_step = -1000
        self.min_steps_between_expansion = 50

        self.subgroups = nn.ModuleList()

        # Group 0: 透明群
        self._add_group(is_identity=True)

        # Group 1+: 活跃群
        for _ in range(initial_active_groups):
            self._add_group(is_identity=False)

    def _add_group(self, is_identity: bool):
        """添加一个 TGroup"""
        sg = TGroup(
            d=self.d,
            hidden_dim=self.hidden_dim,
            num_generators=1 if is_identity else 6,
            is_identity=is_identity,
        )
        self.subgroups.append(sg)
        tag = " [透明]" if is_identity else ""
        print(f"[MetaGroup] 新增群 #{len(self.subgroups) - 1}{tag}")

    def check_expansion(self, error: float, step: int) -> bool:
        """误差过大 → 加群"""
        if (error > self.expansion_threshold
                and step - self.last_expansion_step >= self.min_steps_between_expansion
                and len(self.subgroups) < 10):
            self._add_group(is_identity=False)
            self.last_expansion_step = step
            return True
        return False
