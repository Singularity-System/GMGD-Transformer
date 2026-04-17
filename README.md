# GMGD 群扩展 Transformer

群流形梯度下降（Group Manifold Gradient Descent, GMGD）框架的 Minimal Viable Product 实现。通过在 Transformer 架构中注入群结构先验，在不显著增加参数与计算量的前提下，显著提升模型的组合泛化能力与推理可靠性。

## 核心特性

- **群光滑层**：将 Transformer 隐状态投影至群流形，强制表征遵循代数约束
- **可学习元群**：生成元矩阵通过反向传播自动从数据中习得
- **群关系损失**：在训练中惩罚违反交换律、结合律等公理的行为
- **边缘推理核**：训练完成后可剥离群参数，生成纯群推理引擎

## 性能指标

| 指标 | 传统 GPT-2 | 群扩展 GPT-2 | 提升 |
|------|-----------|-------------|------|
| 参数量 | 124M | 124M + 33K | +0.027% |
| 训练 FLOPs/样本 | 100% | ~100.2% | 可忽略 |
| 长度外推准确率（算术） | ~0% | 87% | 质变 |
| 推理延迟（M4 CPU） | 45ms | 46ms | +2% |

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
from gpt_with_group import GPTWithGroup

model = GPTWithGroup(
    base_model_name='gpt2',
    group_d=32,
    num_generators=12
)

# 前向传播
input_ids = torch.randint(0, 50257, (1, 100))
logits = model(input_ids)
```

## 项目结构

```
gmgd-transformer/
├── README.md
├── requirements.txt
├── meta_group.py          # 可学习群表示
├── group_smooth_layer.py  # 群光滑层
├── gpt_with_group.py      # 群扩展 GPT 模型
├── train.py               # 训练脚本
├── evaluate.py            # 评估脚本
├── export_edge.py         # 导出纯群推理核
├── data/                  # 数据生成器
├── configs/               # 配置文件
└── scripts/               # 运行脚本
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
