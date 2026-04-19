"""
GMGD 核心模块

包含群扩展 Transformer 的核心组件：
- MetaGroup: 可学习元群表示
- GroupSmoothLayer: 群光滑层
- GPTWithGroup: 群扩展 GPT 模型
"""

from .meta_group import MetaGroup, cayley_exp, stable_matrix_exp
from .group_smooth_layer import GroupSmoothLayer, GroupSmoothLayerConfig
from .gpt_with_group import GPTWithGroup, CausalLMOutputWithCrossAttentions

__all__ = [
    'MetaGroup',
    'GroupSmoothLayer',
    'GroupSmoothLayerConfig',
    'GPTWithGroup',
    'CausalLMOutputWithCrossAttentions',
    'cayley_exp',
    'stable_matrix_exp',
]
