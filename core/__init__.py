"""
GMGD 核心模块

包含群扩展 Transformer 的核心组件：
- MetaGroup: 可学习元群表示
- GroupSmoothLayer: 群光滑层（含 Alpha Warm-up）
- GPTWithGroup: 群扩展 GPT 模型（支持单群/多群）
- SubGroupManager: 子群管理器
- DomainRouter: 多尺度上下文路由器
- GroupRegistry: 全局群注册表
- DynamicGroupExpander: 动态群扩张控制器
- PathIntegralAnalyzer: 路径积分分析器
"""

from .meta_group import MetaGroup, cayley_exp, stable_matrix_exp
from .group_smooth_layer import GroupSmoothLayer, GroupSmoothLayerConfig
from .gpt_with_group import GPTWithGroup, CausalLMOutputWithCrossAttentions
from .subgroup_manager import SubGroupManager
from .domain_router import DomainRouter
from .group_registry import GroupRegistry
from .dynamic_expander import DynamicGroupExpander
from .path_integral import PathIntegralAnalyzer, PathIntegralAnalysis

__all__ = [
    'MetaGroup',
    'GroupSmoothLayer',
    'GroupSmoothLayerConfig',
    'GPTWithGroup',
    'CausalLMOutputWithCrossAttentions',
    'cayley_exp',
    'stable_matrix_exp',
    'SubGroupManager',
    'DomainRouter',
    'GroupRegistry',
    'DynamicGroupExpander',
    'PathIntegralAnalyzer',
    'PathIntegralAnalysis',
]
