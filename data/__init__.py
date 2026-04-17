"""
GMGD 群扩展 Transformer 数据生成器模块

包含：
- arithmetic: 算术表达式数据集
- synthetic_reasoning: 合成推理数据集
"""

from .arithmetic import ArithmeticDataset, create_length_split_datasets, generate_expression, evaluate_expression
from .synthetic_reasoning import TransitivityDataset, GroupRelationDataset

__all__ = [
    'ArithmeticDataset',
    'create_length_split_datasets',
    'generate_expression',
    'evaluate_expression',
    'TransitivityDataset',
    'GroupRelationDataset'
]
