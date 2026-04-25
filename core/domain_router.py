"""
DomainRouter: 多尺度上下文路由器

根据输入隐状态的多尺度特征，动态选择激活哪个群。

多尺度特征提取：
- Token 级：当前 token 的隐状态
- 局部窗口：前后 N 个 token 的统计特征
- 序列级：全局统计特征

路由决策：
- 拼接的多尺度特征通过 2 层 MLP
- 输出 K+1 维概率分布（K 个群 + 1 个 fallback）
- 训练时加入熵正则鼓励稀疏分配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class DomainRouter(nn.Module):
    """
    多尺度上下文路由器

    参数：
        hidden_dim: Transformer 隐层维度
        num_groups: 当前群数量
        window_size: 局部窗口大小（用于局部统计）
        mlp_hidden: MLP 隐藏层维度
        temperature: softmax 温度（训练初期高温度鼓励探索）
    """

    def __init__(
        self,
        hidden_dim: int,
        num_groups: int = 1,
        window_size: int = 16,
        mlp_hidden: int = 64,
        temperature: float = 1.0
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_groups = num_groups
        self.window_size = window_size
        self.temperature = temperature

        # 多尺度特征维度：
        # - Token 级: hidden_dim
        # - 局部窗口: 2 * hidden_dim (mean + std) — 简化为全局统计
        # - 序列级: 2 * hidden_dim (mean + std)
        # 总计: 5 * hidden_dim
        feature_dim = hidden_dim + 2 * hidden_dim + 2 * hidden_dim  # 5 * hidden_dim

        # 路由 MLP
        self.router_mlp = nn.Sequential(
            nn.Linear(feature_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, num_groups + 1)  # +1 for fallback
        )

        # 初始化 MLP 权重为小值，避免初期硬分配
        with torch.no_grad():
            self.router_mlp[-1].weight.mul_(0.01)
            self.router_mlp[-1].bias.zero_()

    def _extract_token_features(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Token 级特征：每个 token 的隐状态（取序列平均）

        Args:
            hidden_states: (B, S, H)

        Returns:
            (B, H)
        """
        return hidden_states.mean(dim=1)  # 序列平均

    def _extract_local_window_stats(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        局部窗口统计：取序列前半部分的 mean + std

        Args:
            hidden_states: (B, S, H)

        Returns:
            (B, 2*H)
        """
        window = self.window_size
        local = hidden_states[:, :window, :]
        local_mean = local.mean(dim=1)   # (B, H)
        local_std = local.std(dim=1)      # (B, H)
        return torch.cat([local_mean, local_std], dim=-1)  # (B, 2H)

    def _extract_global_stats(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        序列级全局统计

        Args:
            hidden_states: (B, S, H)

        Returns:
            (B, 2*H)
        """
        global_mean = hidden_states.mean(dim=1)  # (B, H)
        global_std = hidden_states.std(dim=1)     # (B, H)
        return torch.cat([global_mean, global_std], dim=-1)  # (B, 2H)

    def forward(
        self,
        hidden_states: torch.Tensor,
        force_hard: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        前向传播：计算路由概率

        Args:
            hidden_states: (B, S, H)
            force_hard: 是否强制硬分配（argmax）

        Returns:
            probs: (B, S, num_groups + 1) 路由概率分布
            selected_indices: (B, S) 选中的群索引
            selected_group_ids: 选中的群 ID 列表
        """
        B, S, H = hidden_states.shape

        # 提取多尺度特征
        token_feat = self._extract_token_features(hidden_states)        # (B, H)
        local_feat = self._extract_local_window_stats(hidden_states)     # (B, 2H)
        global_feat = self._extract_global_stats(hidden_states)          # (B, 2H)

        # 拼接特征 (B, 5H)
        features = torch.cat([token_feat, local_feat, global_feat], dim=-1)

        # 路由 MLP: (B, num_groups + 1)
        logits = self.router_mlp(features) / self.temperature  # (B, num_groups + 1)

        # Softmax 获得概率
        probs = F.softmax(logits, dim=-1)  # (B, num_groups + 1)

        # 扩展到序列维度：所有 token 共享路由决策
        probs_seq = probs.unsqueeze(1).expand(-1, S, -1)  # (B, S, num_groups + 1)

        # 选择群
        if force_hard or not self.training:
            # 推理或硬模式：argmax
            selected_indices = probs.argmax(dim=-1)  # (B,)
        else:
            # 训练模式：采样（鼓励探索）
            selected_indices = torch.multinomial(probs, num_samples=1, replacement=True).squeeze(-1)  # (B,)

        # 扩展到序列维度
        selected_indices_seq = selected_indices.unsqueeze(1).expand(-1, S)  # (B, S)

        # 获取选中的群 ID
        group_ids = [f"group_{i}" for i in range(self.num_groups)]
        selected_group_ids = []
        for idx in selected_indices.tolist():
            if idx < self.num_groups:
                selected_group_ids.append(group_ids[idx])
            else:
                selected_group_ids.append('fallback')

        return probs_seq, selected_indices_seq, selected_group_ids

    def expand_router(self, new_num_groups: int):
        """
        扩展路由器以适应新增群

        Args:
            new_num_groups: 新的群数量
        """
        old_out_features = self.router_mlp[-1].out_features  # old K + 1
        new_out_features = new_num_groups + 1

        if new_out_features <= old_out_features:
            return  # 不需要扩展

        # 创建新的 Linear 层
        old_linear = self.router_mlp[-1]
        new_linear = nn.Linear(
            old_linear.in_features,
            new_out_features,
            bias=old_linear.bias is not None
        )

        # 复制旧权重
        with torch.no_grad():
            new_linear.weight[:old_out_features] = old_linear.weight
            if old_linear.bias is not None:
                new_linear.bias[:old_out_features] = old_linear.bias
            # 新输出初始化为小噪声
            new_linear.weight[old_out_features:].mul_(0.01)
            if new_linear.bias is not None:
                new_linear.bias[old_out_features:].zero_()

        self.router_mlp[-1] = new_linear
        self.num_groups = new_num_groups

        print(f"[DomainRouter] 扩展至 {new_num_groups} 个群 (输出维度 {new_out_features})")

    def compute_router_loss(self, probs: torch.Tensor) -> torch.Tensor:
        """
        计算路由辅助损失：熵正则化

        鼓励稀疏激活（低熵），避免群间混淆。

        Args:
            probs: (B, S, num_groups + 1) 路由概率

        Returns:
            标量损失
        """
        # 过滤掉 fallback 维度
        task_probs = probs[..., :-1]  # (B, S, num_groups)

        # 熵：-Σ p * log(p)
        entropy = -(task_probs * torch.log(task_probs + 1e-8)).sum(dim=-1)  # (B, S)

        # 平均熵
        return entropy.mean()

    def compute_load_balance_loss(self, probs: torch.Tensor) -> torch.Tensor:
        """
        计算负载均衡损失

        鼓励所有群被近似均匀使用，避免某些群被闲置。

        Args:
            probs: (B, S, num_groups + 1) 路由概率

        Returns:
            标量损失
        """
        #  marginal 概率分布（对所有 batch 和 seq 位置求平均）
        marginal = probs.mean(dim=(0, 1))  # (num_groups + 1,)

        # 与均匀分布的 KL 散度
        uniform = torch.full_like(marginal, 1.0 / marginal.shape[0])
        kl_loss = (marginal * (marginal / (uniform + 1e-8)).log()).sum()

        return kl_loss

    def set_temperature(self, temp: float):
        """设置 softmax 温度"""
        self.temperature = temp

    def extra_repr(self) -> str:
        return f'num_groups={self.num_groups}, window_size={self.window_size}, temperature={self.temperature}'
