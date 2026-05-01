"""
GMGD 核心模块

核心组件：
- Group: 纯群操作（生成元 + Cayley + 流形投影）
- TGroup: Transformer 群操作（Group + hidden_states 交互）
- MetaGroup: 管理群（Group 子类，管理 TGroup 列表）
- DomainManager: 全局域嵌入管理器（EMA 平滑 + 主导切换 + 渐进式恢复）
- GroupSmoothLayer: 光滑层（用域嵌入做群选择）
- GPTWithGroup: 完整模型
"""

from .group import Group, cayley_exp
from .tgroup import TGroup
from .meta_group import MetaGroup
from .gm_model import (
    DomainManager,
    GroupSmoothLayer,
    GPTWithGroup,
    DOMAIN_ARITHMETIC,
    DOMAIN_LANGUAGE,
    DOMAIN_MIXED,
    NUM_DOMAINS,
)

__all__ = [
    'Group',
    'cayley_exp',
    'TGroup',
    'MetaGroup',
    'DomainManager',
    'GroupSmoothLayer',
    'GPTWithGroup',
    'DOMAIN_ARITHMETIC',
    'DOMAIN_LANGUAGE',
    'DOMAIN_MIXED',
    'NUM_DOMAINS',
]
