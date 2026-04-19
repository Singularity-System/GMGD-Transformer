# GMGD 群扩展 Transformer 工程实施计划

## Context（项目背景）

当前大语言模型在组合泛化、逻辑一致性与推理效率方面存在根本性瓶颈。GMGD（Group Manifold Gradient Descent）框架通过将任务规则建模为元群，并将其表示为低维矩阵流形，在 Transformer 架构中注入群结构先验。

本项目目标：实现 GMGD 的最小可行产品——群扩展 Transformer，在不显著增加参数与计算量的前提下，显著提升模型的组合泛化能力与推理可靠性。

核心特性：
- 群光滑层：将 Transformer 隐状态投影至群流形，强制表征遵循代数约束
- 可学习元群：生成元矩阵通过反向传播自动从数据中习得
- 群关系损失：在训练中惩罚违反交换律、结合律等公理的行为
- 边缘推理核：训练完成后可剥离群参数，生成纯群推理引擎

## 实施阶段

### 阶段 1：核心模块开发

#### 1.1 MetaGroup（可学习元群）- meta_group.py
**文件路径**: `/Users/basesystem/GMGD-Transformaner/meta_group.py`

**关键设计**:
- 参数：`num_generators=12`, `d=32`, `group_type='orthogonal'`
- 正交群采用反对称参数化 + 稳定矩阵指数
- 支持批量向量化操作 `project_to_manifold_batch`

**数值稳定性修正**（关键风险修复）:
- 使用 Cayley 变换替代 `torch.linalg.matrix_exp`，避免梯度崩溃
- 谱归一化：限制生成元矩阵的谱半径 < 3.0
- 梯度裁剪：max_norm=1.0

**核心方法**:
```
- get_generator(idx) → (d, d) 矩阵
- all_generators() → (k, d, d) 堆叠张量
- project_to_manifold_batch(X, steps, lr) → 批量投影 (N, d, d)
- relation_loss(relations) → 群关系违反损失
```

#### 1.2 GroupSmoothLayer（群光滑层）- group_smooth_layer.py
**文件路径**: `/Users/basesystem/GMGD-Transformaner/group_smooth_layer.py`

**关键设计**:
- 完全向量化实现，无 Python 循环
- 残差连接：`hidden_states + smooth_hidden`
- 支持任意批量维度 `(..., d, d)`

**数据流**:
```
hidden_states (B, S, H)
    → Linear(H → d*d) → Reshape(B*S, d, d)
    → MetaGroup.project_to_manifold_batch()
    → Reshape(B, S, d*d) → Linear(d*d → H)
    → + hidden_states
```

**性能优化**:
- 使用 `torch.einsum` 替代双重 for 循环
- 批量处理：将 (B, S) 合并为 (B*S) 一次性投影
- 预期加速：从 ~850ms 降至 ~12ms（单层前向，B=32, S=512）

#### 1.3 GPTWithGroup（群扩展 GPT 模型）- gpt_with_group.py
**文件路径**: `/Users/basesystem/GMGD-Transformaner/gpt_with_group.py`

**关键设计**:
- 继承 `GPT2Model`，复用预训练 Token/Position 嵌入
- 每一 Transformer 块后追加 GroupSmoothLayer（共 12 层）
- 语言模型头与词嵌入权重共享

**构建方式**:
```python
model = GPTWithGroup(
    base_model_name='gpt2',
    group_d=32,
    num_generators=12
)
```

---

### 阶段 2：训练与评估基础设施

#### 2.1 数据生成器
**文件路径**: 
- `/Users/basesystem/GMGD-Transformaner/data/arithmetic.py`
- `/Users/basesystem/GMGD-Transformaner/data/synthetic_reasoning.py`

**关键设计**:
- 训练数据：短序列（长度 ≤5）
- 测试数据：长序列（长度 20–100）验证组合泛化
- 任务类型：算术表达式求值、传递性推理

#### 2.2 训练脚本 - train.py
**文件路径**: `/Users/basesystem/GMGD-Transformaner/train.py`

**关键组件**:
- 混合损失：`total_loss = task_loss + 0.1 * rel_loss`
- 优化器：AdamW，学习率 1e-4
- 群光滑层单独学习率：1e-5
- 监控指标：task_loss, rel_loss, orthogonality

#### 2.3 评估脚本 - evaluate.py
**文件路径**: `/Users/basesystem/GMGD-Transformaner/evaluate.py`

**测试维度**:
- 长度外推：[5, 10, 20, 50, 100]
- 抗幻觉测试（传递性）
- 消融实验：无群光滑层、无关系损失、纯基线

---

### 阶段 3：部署优化

#### 3.1 边缘推理核导出 - export_edge.py
**文件路径**: `/Users/basesystem/GMGD-Transformaner/export_edge.py`

**优化措施**:
- 固化群参数为独立 state_dict
- 融合投影层权重
- Cayley 变换替代 matrix_exp（加速 3–5 倍）
- INT8 量化（正交性损失 < 0.1%）

---

## 验证方案

### 单元测试
- MetaGroup：验证正交性 `||R_i^T R_i - I|| < 1e-6`
- GroupSmoothLayer：验证输出形状与残差连接
- 数值稳定性：注入大范数输入，验证无 NaN

### 端到端测试
1. 训练 2 个 epoch，验证 loss 下降
2. 长度外推测试：群扩展模型在所有长度上 >80%，基线在长度>10 时崩跌
3. 传递性测试：群扩展模型达到 100% 准确率

### 性能基准
- 训练 FLOPs/样本：~100.2%（可忽略）
- 推理延迟（M4 CPU）：46ms vs 45ms（+2%）
- 参数量：124M + 33K（+0.027%）

---

## 依赖配置

**文件路径**: `/Users/basesystem/GMGD-Transformaner/requirements.txt`

```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.0.0
accelerate>=0.20.0
scipy>=1.10.0
numpy>=1.24.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

---

## 关键风险与应对

| 风险 | 应对方案 |
|------|----------|
| matrix_exp 梯度崩溃 | 使用 Cayley 变换替代，或谱归一化 + 缩放 |
| 双重循环训练慢 | 批量向量化，einsum 替代 for 循环 |
| 训练初期 loss 震荡 | 前 2 个 epoch 冻结 proj_to/from 权重 |
| rel_loss 不下降 | 增加关系损失权重至 1.0，检查关系定义 |
| MPS 后端报错 | 回退至 CPU：`torch.device('cpu')` |

---

## 项目结构（最终）

```
gmgd-transformer/
├── README.md
├── requirements.txt
├── meta_group.py
├── group_smooth_layer.py
├── gpt_with_group.py
├── train.py
├── evaluate.py
├── export_edge.py
├── data/
│   ├── arithmetic.py
│   └── synthetic_reasoning.py
├── configs/
│   ├── gpt2_group.yaml
│   └── gpt2_medium_group.yaml
└── scripts/
    ├── run_train.sh
    └── run_eval.sh
```

---

## 执行顺序

1. **首先** 创建 requirements.txt 和项目结构
2. **然后** 实现 meta_group.py（核心数学组件）
3. **接着** 实现 group_smooth_layer.py（依赖 MetaGroup）
4. **然后** 实现 gpt_with_group.py（依赖前两者）
5. **之后** 创建数据生成器和训练脚本
6. **最后** 实现评估脚本和导出工具

---

## 阶段 4：动态群扩张机制（自举）

### 4.2 已实现文件（阶段 4 完成）

**核心思想**：不依赖启发式监控，直接通过最后一层路径积分的闭合状态反推缺失的群元素。

**数学原理**：
- 设最后一层输出全局群状态为 \(G_{\text{final}}\)，目标群状态为 \(G_{\text{target}}\)
- 路径积分闭合要求：\(G_{\text{final}} = G_{\text{target}}\)
- 若存在显著差异，缺失变换为：\(\Delta G_{\text{missing}} = G_{\text{final}}^{-1} \cdot G_{\text{target}}\)
- \(\Delta G_{\text{missing}}\) 即为候选新生成元的矩阵表示

**触发流程**：
```
1. 计算闭合误差 ΔG = G_final^{-1} · G_target
2. 若 ||ΔG - I|| < 阈值 → 路径已闭合，无需新群
3. 否则，提取 ΔG 作为候选生成元
4. 临时扩展现有群，优化关系损失
5. 若关系损失收敛 → 吸收 ΔG 到现有群
6. 否则 → 在最后一层创建新 MetaGroup，注册到全局注册表
```

**已实现文件路径**：
- `core/group_registry.py` - 全局群注册表，管理多个 MetaGroup，自动扩展路由器
- `core/path_integral.py` - 路径积分闭合分析与缺失变换计算，代数相容性测试
- `core/dynamic_expander.py` - 扩张控制器，协调吸收/分裂决策
- `core/gpt_with_group.py` - 已集成 DynamicGroupExpander，支持动态群扩张
- `experiments/train_mixed_diagnostic.py` - 已更新以使用动态群扩张
