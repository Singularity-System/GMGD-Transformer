"""
SubGroupManager: 子群管理器

管理 K 个子群的完整生命周期：注册、投影、统计、合并。

设计原则：
1. 每个 Transformer 层绑定一个群 ID
2. 群间投影通过批量 Cayley 变换实现
3. 支持动态新增群（路由器扩展时自动同步）
4. 群合并：生成元过于相似时自动合并
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from .meta_group import MetaGroup


class SubGroupManager(nn.Module):
    """
    管理多个 MetaGroup 实例，提供统一的注册、投影、统计接口。

    参数：
        hidden_dim: Transformer 隐层维度
        initial_group_d: 初始群表示维度 d
        initial_num_generators: 初始生成元个数
        group_type: 群类型 ('orthogonal' 或 'general_linear')
        max_generators_per_group: 单群最大生成元数
    """

    def __init__(
        self,
        hidden_dim: int,
        initial_group_d: int = 16,
        initial_num_generators: int = 6,
        group_type: str = 'orthogonal',
        max_generators_per_group: int = 12
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.group_type = group_type
        self.max_generators_per_group = max_generators_per_group

        # 群实例：group_id -> MetaGroup
        self.groups: Dict[str, MetaGroup] = nn.ModuleDict()

        # 群元数据
        self.group_metadata: Dict[str, dict] = {}

        # 层绑定：layer_idx -> group_id（默认全绑定到 'group_0'）
        self.layer_group_map: Dict[int, str] = {}

        # 注册初始群
        self._create_group('group_0', initial_group_d, initial_num_generators)

    def _create_group(self, group_id: str, group_d: int, num_generators: int):
        """内部方法：创建并注册一个新群"""
        if group_id in self.groups:
            raise ValueError(f"Group '{group_id}' already exists")

        group = MetaGroup(
            num_generators=num_generators,
            d=group_d,
            group_type=self.group_type
        )
        self.groups[group_id] = group

        self.group_metadata[group_id] = {
            'group_d': group_d,
            'num_generators': num_generators,
            'created_at_step': 0,
            'activation_count': 0,
            'total_manifold_dist': 0.0,
            'manifold_dist_count': 0,
        }

        print(f"[SubGroupManager] 注册群: {group_id} (d={group_d}, generators={num_generators})")

    def register_group(
        self,
        group_id: str,
        group_d: int,
        num_generators: int,
        generator_init: Optional[torch.Tensor] = None
    ) -> str:
        """
        注册新群（公开接口）

        Args:
            group_id: 群 ID
            group_d: 群维度 d
            num_generators: 生成元数量
            generator_init: 可选的初始生成元 (num_generators, d, d)

        Returns:
            注册的群 ID
        """
        self._create_group(group_id, group_d, num_generators)

        if generator_init is not None:
            group = self.groups[group_id]
            with torch.no_grad():
                # 确保初始化满足群约束
                if self.group_type == 'orthogonal':
                    # 通过 QR 分解获得正交初始化
                    try:
                        Q, _ = torch.linalg.qr(generator_init.view(-1, group_d * group_d).unsqueeze(0))
                        Q = Q.squeeze(0)  # (num_generators, d, d)
                        # 反对称化
                        skew = 0.5 * (Q - Q.transpose(-2, -1))
                        group.generator_params.copy_(skew)
                    except RuntimeError:
                        # QR 失败，直接反对称化
                        skew = 0.5 * (generator_init - generator_init.transpose(-2, -1))
                        group.generator_params.copy_(skew / (skew.norm() + 1e-6) * 0.5)
                else:
                    group.generator_params.copy_(generator_init)

        return group_id

    def get_group(self, group_id: str) -> MetaGroup:
        """获取指定群"""
        if group_id not in self.groups:
            raise KeyError(f"Group '{group_id}' not found. Available: {list(self.groups.keys())}")
        return self.groups[group_id]

    def get_group_d(self, group_id: str) -> int:
        """获取群的维度 d"""
        return self.group_metadata[group_id]['group_d']

    def project_to_manifold(
        self,
        group_id: str,
        matrices: torch.Tensor,
        num_steps: int = 1,
        lr: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将矩阵投影到指定群的流形

        Args:
            group_id: 群 ID
            matrices: 输入矩阵 (N, d, d)
            num_steps: 投影步数
            lr: 学习率

        Returns:
            (光滑后矩阵, 增量矩阵) — 均为 (N, d, d)
        """
        group = self.get_group(group_id)

        # 维度检查
        expected_d = self.group_metadata[group_id]['group_d']
        if matrices.shape[-1] != expected_d:
            raise ValueError(
                f"Matrix dimension {matrices.shape[-1]} != group d={expected_d}"
            )

        # 批量投影
        matrices_smooth = group.project_to_manifold_batch(matrices, num_steps=num_steps, lr=lr)
        delta = matrices_smooth - matrices

        # 更新统计
        meta = self.group_metadata[group_id]
        meta['activation_count'] += matrices.shape[0]
        manifold_dist = delta.norm(dim=(-2, -1)).mean().item()
        meta['total_manifold_dist'] += manifold_dist * matrices.shape[0]
        meta['manifold_dist_count'] += matrices.shape[0]

        return matrices_smooth, delta

    def bind_layer(self, layer_idx: int, group_id: str):
        """绑定某层到指定群"""
        if group_id not in self.groups:
            raise KeyError(f"Group '{group_id}' not found")
        self.layer_group_map[layer_idx] = group_id

    def get_layer_group(self, layer_idx: int) -> str:
        """获取某层绑定的群 ID"""
        if not self.layer_group_map:
            return 'group_0'  # 默认
        return self.layer_group_map.get(layer_idx, 'group_0')

    def get_statistics(self) -> dict:
        """获取所有群的统计信息"""
        stats = {}
        for gid, meta in self.group_metadata.items():
            avg_dist = (
                meta['total_manifold_dist'] / max(meta['manifold_dist_count'], 1)
            )
            stats[gid] = {
                'd': meta['group_d'],
                'num_generators': meta['num_generators'],
                'activation_count': meta['activation_count'],
                'avg_manifold_dist': avg_dist,
                'created_at_step': meta['created_at_step'],
            }
        return stats

    def merge_similar_groups(self, similarity_threshold: float = 0.95) -> Optional[Tuple[str, str, str]]:
        """
        检测并合并过于相似的两个群

        相似度度量：两个群的所有生成元之间的平均余弦相似度。

        Args:
            similarity_threshold: 超过此阈值则触发合并

        Returns:
            若发生合并：(group_a_id, group_b_id, merged_group_id)
            否则 None
        """
        group_ids = list(self.groups.keys())
        if len(group_ids) < 2:
            return None

        best_sim = -1.0
        best_pair = None

        for i in range(len(group_ids)):
            for j in range(i + 1, len(group_ids)):
                gid_a, gid_b = group_ids[i], group_ids[j]
                gen_a = self.groups[gid_a].all_generators()  # (k_a, d, d)
                gen_b = self.groups[gid_b].all_generators()  # (k_b, d, d)

                # 简化：比较生成元平均方向
                mean_a = gen_a.mean(dim=0)  # (d, d)
                mean_b = gen_b.mean(dim=0)  # (d, d)

                # 余弦相似度
                sim = (mean_a * mean_b).sum() / (mean_a.norm() * mean_b.norm() + 1e-8)
                sim = sim.abs().item()

                if sim > best_sim:
                    best_sim = sim
                    best_pair = (gid_a, gid_b)

        if best_sim > similarity_threshold and best_pair is not None:
            gid_a, gid_b = best_pair
            # 合并：保留 gid_a，将所有绑定到 gid_b 的层重绑定到 gid_a
            merged_id = gid_a
            for layer_idx, bound_gid in list(self.layer_group_map.items()):
                if bound_gid == gid_b:
                    self.layer_group_map[layer_idx] = merged_id

            # 删除 gid_b
            del self.groups[gid_b]
            del self.group_metadata[gid_b]

            print(f"[SubGroupManager] 合并群: {gid_b} -> {merged_id} (sim={best_sim:.4f})")
            return (gid_a, gid_b, merged_id)

        return None

    def generate_next_id(self) -> str:
        """生成下一个可用的群 ID"""
        idx = len(self.groups)
        return f"group_{idx}"

    @property
    def num_groups(self) -> int:
        """当前群数量"""
        return len(self.groups)

    def get_all_group_ids(self) -> List[str]:
        """所有群 ID"""
        return list(self.groups.keys())

    def forward(self, hidden_states: torch.Tensor) -> dict:
        """
        简化前向：根据隐状态统计选择最合适的群

        这是一个快捷接口，完整的路由逻辑由 DomainRouter 处理。

        Returns:
            {'selected_group_id': str, 'manager': self}
        """
        return {
            'selected_group_id': 'group_0',
            'manager': self,
        }
