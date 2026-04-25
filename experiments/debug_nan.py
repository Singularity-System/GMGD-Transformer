#!/usr/bin/env python3
"""诊断 NaN 来源"""
import sys, os
sys.path.insert(0, os.path.dirname('..'))

import torch
from core import GPTWithGroup, MetaGroup
from transformers import GPT2Tokenizer

device = torch.device('cpu')
torch.manual_seed(42)

# 1. 测试 MetaGroup 初始化
print("=" * 60)
print("测试 1: MetaGroup 初始化")
print("=" * 60)

meta_group = MetaGroup(num_generators=6, d=16, group_type='orthogonal', init_noise=0.01)
print(f"generator_params 有 NaN: {torch.isnan(meta_group.generator_params).any().item()}")
print(f"generator_params 有 Inf: {torch.isinf(meta_group.generator_params).any().item()}")
print(f"generator_params shape: {meta_group.generator_params.shape}")
print(f"generator_params 范围: [{meta_group.generator_params.min().item():.6f}, {meta_group.generator_params.max().item():.6f}]")

# 测试 all_generators
try:
    gens = meta_group.all_generators()
    print(f"all_generators 有 NaN: {torch.isnan(gens).any().item()}")
    print(f"all_generators shape: {gens.shape}")
except Exception as e:
    print(f"all_generators 失败: {e}")

# 2. 测试 GPTWithGroup 初始化
print("\n" + "=" * 60)
print("测试 2: GPTWithGroup 初始化")
print("=" * 60)

model = GPTWithGroup(
    base_model_name='gpt2',
    group_d=16,
    num_generators=6,
    group_type='orthogonal',
    use_pretrained=True
).to(device)

print(f"meta_group.generator_params 有 NaN: {torch.isnan(model.meta_group.generator_params).any().item()}")
print(f"smooth_layers[0].proj_to.weight 有 NaN: {torch.isnan(model.smooth_layers[0].proj_to.weight).any().item()}")
print(f"smooth_layers[0].proj_from.weight 有 NaN: {torch.isnan(model.smooth_layers[0].proj_from.weight).any().item()}")

# 3. 测试前向传播
print("\n" * 60)
print("测试 3: 前向传播")
print("=" * 60)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text = "Expr: 3 + 5 ="
enc = tokenizer(text, return_tensors='pt')
input_ids = enc['input_ids'].to(device)

print(f"input_ids shape: {input_ids.shape}")

# 测试嵌入
with torch.no_grad():
    wte_out = model.base_model.wte(input_ids)
    wpe_out = model.base_model.wpe(torch.arange(input_ids.shape[1], device=device))
    hidden = wte_out + wpe_out

print(f"wte_out 有 NaN: {torch.isnan(wte_out).any().item()}")
print(f"hidden 有 NaN: {torch.isnan(hidden).any().item()}")
print(f"hidden 范围: [{hidden.min().item():.6f}, {hidden.max().item():.6f}]")

# 测试第一层
try:
    block_out = model.base_model.h[0](hidden)[0]
    print(f"block_out 有 NaN: {torch.isnan(block_out).any().item()}")

    smooth_out, dist, delta = model.smooth_layers[0](block_out)
    print(f"smooth_out 有 NaN: {torch.isnan(smooth_out).any().item()}")
    print(f"manifold_distance: {dist}")
except Exception as e:
    print(f"前向传播失败: {e}")
    import traceback
    traceback.print_exc()
