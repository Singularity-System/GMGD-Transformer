#!/usr/bin/env python3
"""
测试全局群状态累乘机制（路径积分）
验证：每一层在同一个群流形轨道上滑行
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch

from core import GPTWithGroup

# 创建模型
print("\n创建模型...")
model = GPTWithGroup(
    base_model_name='gpt2',
    group_d=16,
    num_generators=6,
    group_type='orthogonal',
    use_pretrained=False
)

print(f"模型层数：{model.num_layers}")
print(f"群表示维度：{model.group_d}")

# 前向传播
print("\n前向传播测试...")
batch_size = 2
seq_len = 32
vocab_size = 50257

input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
labels = input_ids.clone()

with torch.no_grad():
    outputs = model(input_ids=input_ids, labels=labels)

print(f"loss: {outputs.loss.item():.4f}")
print(f"manifold_distance: {outputs.manifold_distance.item():.6f}")

# 验证全局群状态
if outputs.global_group_state is not None:
    print(f"\n✓ global_group_state 存在")
    print(f"  形状：{outputs.global_group_state.shape}")
    print(f"  数据类型：{outputs.global_group_state.dtype}")

    # 计算全局群状态的统计特性
    ggs = outputs.global_group_state  # (B, d, d)

    # 1. 弗罗贝尼乌斯范数
    frob_norm = torch.norm(ggs, dim=(-2, -1)).mean().item()
    print(f"\n  弗罗贝尼乌斯范数：{frob_norm:.6f}")

    # 2. 与单位矩阵的偏差（衡量累积效应）
    identity = torch.eye(model.group_d).unsqueeze(0).expand(batch_size, -1, -1)
    deviation = torch.norm(ggs - identity).item()
    print(f"  与单位矩阵偏差：{deviation:.6f}")
    print(f"  （训练初期应较小，训练后应增大表示学到非平凡变换）")

    # 3. 正交性检验（对 orthogonal 群）
    ggs_T = ggs.transpose(-2, -1)
    ortho_check = torch.norm(ggs_T @ ggs - identity, dim=(-2, -1)).mean().item()
    print(f"\n  正交性偏差 ||G^T G - I||: {ortho_check:.6f}")
    print(f"  （应接近 0，表示保持正交性）")

    # 4. 行列式（正交群行列式应为 ±1）
    det = torch.linalg.det(ggs).abs().mean().item()
    print(f"  平均 |det(G)|: {det:.6f}")
    print(f"  （应接近 1）")

else:
    print("\n✗ global_group_state 缺失！")

# 验证层间增量
print("\n" + "=" * 60)
print("层间一致性分析")
print("=" * 60)

model.eval()

# 逐层分析群增量
batch_size = 1
seq_len = 16
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

with torch.no_grad():
    # 获取基础隐藏状态（GPT2Model 直接调用返回 last_hidden_state）
    transformer_outputs = model.base_model(input_ids=input_ids)
    hidden_states = transformer_outputs.last_hidden_state

    print(f"\n输入形状：{hidden_states.shape}")

    # 逐层通过群光滑层
    layer_stats = []
    for i, smooth_layer in enumerate(model.smooth_layers):
        hidden_states, manifold_dist, delta_group = smooth_layer(hidden_states)

        # 统计该层的群增量
        delta_norm = torch.norm(delta_group, dim=(-2, -1)).mean().item()
        delta_mean = delta_group.mean().item()
        delta_std = delta_group.std().item()

        layer_stats.append({
            'layer': i,
            'delta_norm': delta_norm,
            'delta_mean': delta_mean,
            'delta_std': delta_std
        })

        print(f"层 {i:2d}: ||δ||={delta_norm:.6f}, mean={delta_mean:8.6f}, std={delta_std:.6f}")

# 检查是否所有层都有非平凡贡献
delta_norms = [s['delta_norm'] for s in layer_stats]
min_delta = min(delta_norms)
max_delta = max(delta_norms)

print(f"\n群增量统计:")
print(f"  最小 ||δ||: {min_delta:.6f}")
print(f"  最大 ||δ||: {max_delta:.6f}")
print(f"  平均 ||δ||: {sum(delta_norms)/len(delta_norms):.6f}")

if min_delta > 1e-6:
    print("\n✓ 所有层都有非平凡群贡献（路径积分有效）")
else:
    print("\n✗ 某些层的群增量为零（可能存在短路）")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
