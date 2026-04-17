"""
GMGD 群扩展 Transformer 核心模块

GMGD (Group Manifold Gradient Descent) - 群流形梯度下降
通过在 Transformer 架构中注入群结构先验，提升组合泛化能力。

核心组件：
- MetaGroup: 可学习元群表示
- GroupSmoothLayer: 群光滑层
- GPTWithGroup: 群扩展 GPT 模型

使用示例：
    >>> from gmgd_transformer import GPTWithGroup
    >>> model = GPTWithGroup(
    ...     base_model_name='gpt2',
    ...     group_d=32,
    ...     num_generators=12
    ... )
    >>> input_ids = torch.randint(0, 50257, (4, 100))
    >>> logits = model(input_ids)
"""

from meta_group import MetaGroup, cayley_exp, stable_matrix_exp
from group_smooth_layer import GroupSmoothLayer, GroupSmoothLayerConfig
from gpt_with_group import GPTWithGroup

__version__ = '1.0.0'
__author__ = 'GMGD Core Team'

__all__ = [
    'MetaGroup',
    'cayley_exp',
    'stable_matrix_exp',
    'GroupSmoothLayer',
    'GroupSmoothLayerConfig',
    'GPTWithGroup'
]
