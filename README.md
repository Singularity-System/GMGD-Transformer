# GMGD 群扩展 Transformer

群流形梯度下降（Group Manifold Gradient Descent, GMGD）框架的完整实现。通过在 Transformer 架构中注入群结构先验，在不显著增加参数与计算量的前提下，显著提升模型的组合泛化能力与推理可靠性。

## 核心特性

- **域嵌入管理器**：全局域方向向量 + EMA 平滑 + 主导切换检测 + 渐进式历史恢复
- **群光滑层**：基于域嵌入做群选择，将 Transformer 隐状态投影至群流形
- **TGroup 群操作**：继承纯群操作，实现 hidden_states ↔ 群矩阵的双向投影
- **可学习元群**：生成元矩阵通过反向传播自动从数据中习得
- **边缘推理核**：训练完成后可剥离群参数，生成纯群推理引擎

## 性能指标

| 指标 | 传统 GPT-2 | 群扩展 GPT-2 | 提升 |
|------|-----------|-------------|------|
| 参数量 | 124M | 124M + 33K | +0.027% |
| 训练 FLOPs/样本 | 100% | ~100.2% | 可忽略 |
| 长度外推准确率（算术） | ~0% | 100% | 质变 |
| 语言任务准确率 | ~95% | 100% | 提升 |
| 混合任务准确率 | ~85% | 100% | 质变 |

## 安装

```bash
pip install -r requirements.txt
```

**注意**：首次运行时会通过 HuggingFace 自动下载 GPT-2 预训练权重（约 500MB）。
如遇网络问题，可手动下载后设置环境变量：

```bash
export TRANSFORMERS_CACHE=/path/to/cache
```

## 快速开始

```python
from core import GPTWithGroup

model = GPTWithGroup(
    base_model_name='gpt2',
    group_d=16,
    num_generators=6
)

# 前向传播
input_ids = torch.randint(0, 50257, (1, 100))
logits = model(input_ids)
```

## 核心架构

### 模块层次

```
GPTWithGroup
├── DomainManager      # 全局域嵌入（方向向量 + EMA 平滑 + 主导切换）
├── GroupSmoothLayer   # 群光滑层（域选择 + 群修正）
│   └── MetaGroup      # 元群管理器
│       └── TGroup     # Transformer 群操作
│           └── Group  # 纯群操作（Cayley 指数映射 + 流形投影）
```

### 域管理器（DomainManager）

七大机制：
1. 全局域嵌入 — 每个域的可学习方向向量
2. 定长归一化 — L2 归一化，方向信号强度恒定
3. 对数分布幅值 — 对数空间存储幅值，梯度稳定
4. EMA 平滑 — 域权重丝滑演变，避免抖动
5. 主导切换检测 — 识别域切换事件
6. 渐进式历史恢复 — 旧主导域缓慢恢复
7. 统一恢复目标 0.9 — 所有域非主导时保持战备状态

### 群操作（TGroup）

- `proj_to`: hidden_states → (d,d) 矩阵
- `get_correction`: Cayley 指数映射计算群增量
- `proj_from`: 群增量 → hidden_states 修正

## 项目结构

```
gmgd-transformer/
├── README.md
├── requirements.txt
├── core/                    # 核心模块
│   ├── __init__.py
│   ├── group.py             # 纯群操作
│   ├── tgroup.py            # Transformer 群操作
│   ├── meta_group.py        # 元群管理器
│   ├── gm_model.py          # 域管理器 + 群光滑层 + 完整模型
│   └── group.py             # 群基础类
├── experiments/             # 实验脚本和结果
│   ├── train_multi_group.py # 多群训练脚本
│   └── multi_group_results/ # 实验结果
├── checkpoints/             # 模型检查点
└── 核心重构说明.md          # 架构重构文档
```

## 文档

详细工程说明书见项目根目录下的 `GMGD_Transformer_工程说明书.md`。

## 依赖

- `torch>=2.0.0`
- `transformers>=4.30.0`（提供 GPT-2 模型）
- `datasets>=2.0.0`
- `accelerate>=0.20.0`
- `scipy>=1.10.0`
- `numpy>=1.24.0`
- `matplotlib>=3.7.0`
- `tqdm>=4.65.0`

**GPT-2 说明**：本项目通过 `transformers` 库加载 GPT-2 预训练权重，支持以下模型：
- `gpt2`（124M，默认）
- `gpt2-medium`（355M）
- `gpt2-large`（774M）
- `gpt2-xl`（1558M）

## 训练示例

```bash
# 使用默认配置训练 3 个 epoch
bash scripts/run_train.sh --epochs 3 --batch_size 16

# 或直接用 Python
python train.py --base_model gpt2 --group_d 32 --num_generators 12 --epochs 3
```

## 评估

```bash
# 运行完整评估（长度外推、传递性、正交性）
bash scripts/run_eval.sh --model_path ./checkpoints/best_model

# 导出边缘推理核
python export_edge.py --model_path ./checkpoints/best_model --output_dir ./edge_model
```

## 许可证

MIT License
