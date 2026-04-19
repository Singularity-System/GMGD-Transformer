"""
全局群注册表：管理多个 MetaGroup，支持动态扩张

核心功能：
1. 注册多个群实例，每个群有唯一 ID
2. 路由器：根据输入特征选择激活哪些群
3. 动态扩张：当检测到新域时，创建新群并扩展路由器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from .meta_group import MetaGroup


class GroupRegistry(nn.Module):
    """
    全局群注册表：管理多个 MetaGroup 实例

    设计原则：
    1. 所有层共享同一个注册表
    2. 路由器根据输入特征动态选择群
    3. 支持动态添加新群

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
        group_type: str = 'orthogonal'
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.group_type = group_type

        # 群实例字典：group_id -> MetaGroup
        self.groups: Dict[str, MetaGroup] = nn.ModuleDict()

        # 群元数据：记录每个群的用途和统计信息
        self.group_metadata: Dict[str, dict] = {}

        # 路由器：输入隐状态 -> 群选择概率分布
        # 输出维度 = 当前群数量 + 1（+1 表示"无群激活"的 fallback）
        self.router = nn.Linear(hidden_dim, 2)  # 初始只有 1 个群，输出维度为 2

        # 注册初始群
        self._register_group('default', initial_group_d, initial_num_generators)

        # 路径积分闭合监控
        self.path_integral_threshold = 0.5  # 闭合误差阈值

    def _register_group(self, group_id: str, group_d: int, num_generators: int):
        """内部方法：注册一个新群"""
        if group_id in self.groups:
            raise ValueError(f"Group {group_id} already exists")

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
            'coherence_loss_history': []
        }

        print(f"[GroupRegistry] 注册新群：{group_id} (d={group_d}, k={num_generators})")

    def register_new_group(self, group_id: str, group_d: int, num_generators: int,
                           generator_init: Optional[torch.Tensor] = None):
        """
        注册一个新群（带初始化矩阵）

        Args:
            group_id: 群 ID
            group_d: 群维度
            num_generators: 生成元数量
            generator_init: 可选的初始生成元矩阵 (num_generators, d, d)
        """
        self._register_group(group_id, group_d, num_generators)

        # 扩展路由器输出维度
        old_num_groups = len(self.groups) - 1  # 减 1 是因为新群刚加入
        new_num_groups = len(self.groups)  # 包含新群

        # 获取旧权重
        old_weight = self.router.weight.data  # (old_num_groups + 1, hidden_dim)
        old_bias = self.router.bias.data if self.router.bias is not None else None

        # 创建新权重矩阵（增加一行）
        new_weight = torch.zeros(new_num_groups + 1, self.hidden_dim, device=old_weight.device)
        new_weight[:old_num_groups + 1] = old_weight

        # 新群的路由器权重初始化为小噪声
        nn.init.normal_(new_weight[new_num_groups], mean=0, std=0.01)

        # 更新路由器
        self.router = nn.Linear(self.hidden_dim, new_num_groups + 1, bias=self.router.bias is not None)
        self.router.weight.data = new_weight
        if old_bias is not None:
            new_bias = torch.zeros(new_num_groups + 1, device=old_bias.device)
            new_bias[:old_num_groups + 1] = old_bias
            self.router.bias.data = new_bias

        print(f"[GroupRegistry] 路由器扩展至 {new_num_groups + 1} 维")

    def get_router_probs(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        获取路由概率

        Args:
            hidden_states: 隐状态 (..., hidden_dim)

        Returns:
            路由概率分布 (..., num_groups + 1)
        """
        logits = self.router(hidden_states)
        return F.softmax(logits, dim=-1)

    def select_groups(self, hidden_states: torch.Tensor, top_k: int = 1) -> Tuple[List[str], torch.Tensor]:
        """
        为每个 token 选择激活的群

        Args:
            hidden_states: 隐状态 (B, S, hidden_dim)
            top_k: 选择 top-k 个群

        Returns:
            selected_group_ids: 选中的群 ID 列表
            router_probs: 路由概率 (B, S, num_groups)
        """
        # 获取路由概率
        router_probs = self.get_router_probs(hidden_states)  # (B, S, num_groups + 1)

        # 获取当前所有群 ID
        group_ids = list(self.groups.keys())

        # 取 top-k 群
        topk_probs, topk_indices = torch.topk(router_probs[:, :, :-1], min(top_k, len(group_ids)), dim=-1)

        # 计算平均激活，选择最可能的那个群
        avg_probs = topk_probs.mean(dim=(0, 1))  # (top_k,)
        selected_idx = topk_indices[0, 0, avg_probs.argmax()].item()

        selected_group_id = group_ids[selected_idx] if selected_idx < len(group_ids) else 'default'

        return [selected_group_id], router_probs[:, :, :-1]

    def forward(self, hidden_states: torch.Tensor, target_group_state: Optional[torch.Tensor] = None) -> dict:
        """
        前向传播：路由 + 群变换

        Args:
            hidden_states: 输入隐状态 (B, S, hidden_dim)
            target_group_state: 目标群状态 (B, d, d)，用于计算路径积分闭合误差

        Returns:
            output: 变换后的隐状态
            router_loss: 路由器辅助损失
            path_integral_error: 路径积分闭合误差
        """
        B, S, H = hidden_states.shape

        # 获取路由选择
        selected_group_ids, router_probs = self.select_groups(hidden_states)
        selected_group_id = selected_group_ids[0]

        # 获取对应的群
        group = self.groups[selected_group_id]

        # 更新群激活计数
        self.group_metadata[selected_group_id]['activation_count'] += B * S

        # 计算路径积分闭合误差（如果提供目标状态）
        path_integral_error = None
        if target_group_state is not None:
            # 获取当前群状态（简化：使用生成元的加权和）
            current_group_state = self._get_current_group_state(selected_group_id, B)

            # 计算闭合误差：||G_final^{-1} @ G_target - I||
            G_final_inv = torch.linalg.inv(current_group_state)  # (B, d, d)
            delta = G_final_inv @ target_group_state  # (B, d, d)
            I = torch.eye(group.d, device=delta.device).unsqueeze(0).expand(B, -1, -1)
            path_integral_error = torch.norm(delta - I, dim=(-2, -1)).mean()

        # 路由器辅助损失：鼓励稀疏激活（熵最小化）
        router_entropy = -(router_probs * torch.log(router_probs + 1e-8)).mean()
        router_loss = 0.1 * router_entropy  # 熵正则化权重

        return {
            'selected_group_id': selected_group_id,
            'group': group,
            'router_probs': router_probs,
            'router_loss': router_loss,
            'path_integral_error': path_integral_error
        }

    def _get_current_group_state(self, group_id: str, batch_size: int) -> torch.Tensor:
        """获取当前群状态（用于路径积分）"""
        group = self.groups[group_id]

        # 简化实现：使用所有生成元的乘积作为群状态
        # 实际应维护累积的路径积分
        generators = group.all_generators()  # (k, d, d)

        # 生成元连乘
        state = torch.eye(group.d, device=generators.device).unsqueeze(0).expand(batch_size, -1, -1)
        for gen in generators:
            state = state @ gen.unsqueeze(0).expand(batch_size, -1, -1)

        return state

    def analyze_path_integral(self, G_final: torch.Tensor, G_target: torch.Tensor) -> dict:
        """
        分析路径积分闭合状态，为可能的新群创建提供依据

        Args:
            G_final: 当前全局群状态 (B, d, d)
            G_target: 目标群状态 (B, d, d)

        Returns:
            analysis: 分析报告
        """
        B, d, _ = G_final.shape

        # 计算缺失变换
        G_final_inv = torch.linalg.inv(G_final)
        delta_missing = G_final_inv @ G_target  # (B, d, d)

        # 计算与单位阵的偏差
        I = torch.eye(d, device=G_final.device).unsqueeze(0).expand(B, -1, -1)
        deviation = torch.norm(delta_missing - I, dim=(-2, -1)).mean().item()

        # 分析 delta_missing 的代数特性
        # 1. 对称性：判断是正交变换还是其他
        symmetric_part = 0.5 * (delta_missing + delta_missing.transpose(-2, -1))
        skew_symmetric_part = 0.5 * (delta_missing - delta_missing.transpose(-2, -1))

        symmetric_norm = torch.norm(symmetric_part, dim=(-2, -1)).mean().item()
        skew_symmetric_norm = torch.norm(skew_symmetric_part, dim=(-2, -1)).mean().item()

        # 2. 行列式：判断缩放因子
        det = torch.linalg.det(delta_missing).abs().mean().item()

        return {
            'deviation_from_identity': deviation,
            'symmetric_norm': symmetric_norm,
            'skew_symmetric_norm': skew_symmetric_norm,
            'determinant': det,
            'delta_missing_mean': delta_missing.mean().item(),
            'needs_new_group': deviation > self.path_integral_threshold,
            'candidate_generator': delta_missing.mean(dim=0).detach()  # (d, d)
        }

    @property
    def num_groups(self) -> int:
        """返回当前注册的群数量"""
        return len(self.groups)

    def get_group_ids(self) -> List[str]:
        """返回所有群 ID"""
        return list(self.groups.keys())
