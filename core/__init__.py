"""
GMGD 核心模块

包含群扩展 Transformer 的核心组件：
- MetaGroup: 元群（管理 SubGroup 索引 + 扩张）
- SubGroup: 子群（独立生成元 + 投影层）
- GroupAttention: 共享注意力聚合器
- GroupSmoothLayer: 群光滑层（含 Alpha Warm-up）
- GPTWithGroup: 群扩展 GPT 模型（支持单群/多群）
- GroupRegistry: 全局群注册表
- DynamicGroupExpander: 动态群扩张控制器
- PathIntegralAnalyzer: 路径积分分析器
"""

from .meta_group import MetaGroup, cayley_exp, stable_matrix_exp
from .subgroup import SubGroup
from .group_attention import GroupAttention
from .group_smooth_layer import GroupSmoothLayer
from .gpt_with_group import GPTWithGroup, CausalLMOutputWithCrossAttentions
from .group_registry import GroupRegistry
from .dynamic_expander import DynamicGroupExpander
from .path_integral import PathIntegralAnalyzer, PathIntegralAnalysis

__all__ = [
    'MetaGroup',
    'SubGroup',
    'GroupAttention',
    'GroupSmoothLayer',
    'GPTWithGroup',
    'CausalLMOutputWithCrossAttentions',
    'cayley_exp',
    'stable_matrix_exp',
    'GroupRegistry',
    'DynamicGroupExpander',
    'PathIntegralAnalyzer',
    'PathIntegralAnalysis',
]
