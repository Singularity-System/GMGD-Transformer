"""
GroupAttention: 群注意力聚合器

用自注意力代替路由 MLP：
- 从 hidden_states 的每个 token 计算 query（不复用 mean+std）
- 从每个子群的可学习参数计算 key
- attention 权重作为软路由，加权聚合各群的平滑输出
- 无需显式路由损失，注意力本身就足够稀疏

数据流：
    hidden_states (B, S, H)
        → q = Linear(H → d_q)  [query per token]
        → K = group_keys (K, d_q)  [one key per subgroup]
        → A = softmax(q @ K^T / sqrt(d_q))  [attention weights (B, S, K)]
        → output = Σ A[t,g] · smooth_g[t]  [weighted aggregation]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class GroupAttention(nn.Module):
    """
    群注意力：根据每个 token 的隐状态计算各子群的注意力权重

    设计：
    - Query 来自每个 token 的 hidden_states（不复用 mean+std 丢失时序）
    - Key 来自每个子群的可学习参数（共享）
    - 输出注意力概率分布 (B, S, num_groups)

    参数：
        hidden_dim: Transformer 隐层维度
        num_groups: 初始子群数量
        query_dim: query/key 维度（默认 hidden_dim // 4）
    """

    def __init__(self, hidden_dim: int, num_groups: int = 1, query_dim: int = 64):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.query_dim = query_dim

        # Query: 从每个 token 的隐状态投影到 query_dim
        self.query_proj = nn.Linear(hidden_dim, query_dim)

        # Group keys: 每个子群一个可学习向量 (num_groups, query_dim)
        self.group_keys = nn.Parameter(torch.randn(num_groups, query_dim) * 0.02)

        self.scale = math.sqrt(query_dim)

        self._init_weights()

    def _init_weights(self):
        """初始化 query 投影层"""
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)

    def expand_groups(self, new_num_groups: int):
        """扩展子群数量（新增子群的 key 随机初始化）"""
        old_keys = self.group_keys.data
        new_keys = torch.randn(new_num_groups - old_keys.shape[0], self.query_dim) * 0.02
        self.group_keys = nn.Parameter(torch.cat([old_keys, new_keys], dim=0))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        计算群注意力权重

        Args:
            hidden_states: (B, S, hidden_dim)

        Returns:
            attention_weights: (B, S, num_groups) — 每个 token 对每个子群的注意力权重
        """
        # Query per token: (B, S, query_dim)
        query = self.query_proj(hidden_states)

        # Keys: (K, query_dim)
        keys = self.group_keys

        # Attention logits: (B, S, K)
        logits = torch.einsum('bsd,kd->bsk', query, keys) / self.scale

        # Softmax over groups: (B, S, K)
        attn = F.softmax(logits, dim=-1)

        return attn

    def get_num_groups(self) -> int:
        return self.group_keys.shape[0]

    def get_group_ids(self) -> List[str]:
        """返回子群 ID 列表（兼容接口）"""
        return [f"subgroup_{i}" for i in range(self.get_num_groups())]

    def extra_repr(self) -> str:
        return f'query_dim={self.query_dim}, num_groups={self.get_num_groups()}'
