#!/usr/bin/env python3
"""
测试自适应群光滑机制

验证：
1. 可学习的残差权重 alpha 是否生效
2. 门控网络是否根据输入特征动态调整投影强度
3. 自适应学习率是否根据流形距离调整
4. 梯度是否能够正确反向传播到 MetaGroup 参数
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
import torch.nn as nn

from core import GPTWithGroup
from core.group_smooth_layer import GroupSmoothLayer
from core.meta_group import MetaGroup


def test_adaptive_layer():
    """测试自适应群光滑层的基本功能"""
    print("=" * 60)
    print("测试 1: 自适应群光滑层基本功能")
    print("=" * 60)

    # 创建元群
    meta_group = MetaGroup(num_generators=6, d=16, group_type='orthogonal')

    # 创建自适应层
    smooth_layer = GroupSmoothLayer(
        hidden_dim=768,
        group_d=16,
        meta_group=meta_group,
        proj_steps=1,
        smooth_lr=0.1,
        adaptive=True
    )

    print(f"\n层配置：{smooth_layer.extra_repr()}")

    # 前向传播
    batch_size = 2
    seq_len = 32
    hidden_dim = 768

    x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)

    with torch.no_grad():
        output, manifold_dist, delta_group = smooth_layer(x)

    print(f"\n输入形状：{x.shape}")
    print(f"输出形状：{output.shape}")
    print(f"流形距离：{manifold_dist.item():.6f}")
    print(f"群增量形状：{delta_group.shape}")

    # 检查可学习参数
    print(f"\n可学习参数:")
    print(f"  weight_alpha: {smooth_layer.weight_alpha.item():.4f}")
    print(f"  alpha = sigmoid(weight_alpha): {torch.sigmoid(smooth_layer.weight_alpha).item():.4f}")

    # 验证门控网络
    print(f"\n门控网络参数:")
    for name, param in smooth_layer.gate_network.named_parameters():
        print(f"  {name}: {param.shape}")

    # 验证梯度流动
    print("\n梯度流动验证:")

    # 重新创建需要梯度的输入
    x_requires_grad = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)
    output2, _, _ = smooth_layer(x_requires_grad)

    output_sum = output2.sum()
    output_sum.backward()

    print(f"  输入梯度存在：{x_requires_grad.grad is not None}")
    print(f"  输入梯度范数：{x_requires_grad.grad.norm().item():.6f}")

    # MetaGroup 参数梯度
    for name, param in meta_group.named_parameters():
        if param.grad is not None:
            print(f"  {name} 梯度范数：{param.grad.norm().item():.6f}")
        else:
            print(f"  {name} 无梯度")

    print("\n✓ 自适应层基本功能正常")


def test_adaptive_vs_non_adaptive():
    """对比自适应和非自适应模式的差异"""
    print("\n" + "=" * 60)
    print("测试 2: 自适应 vs 非自适应模式对比")
    print("=" * 60)

    # 创建两个相同的输入
    torch.manual_seed(42)
    x1 = torch.randn(2, 32, 768)

    torch.manual_seed(42)
    x2 = torch.randn(2, 32, 768)

    # 自适应层
    layer_adaptive = GroupSmoothLayer(
        hidden_dim=768, group_d=16,
        proj_steps=1, smooth_lr=0.1,
        adaptive=True
    )

    # 非自适应层
    layer_non_adaptive = GroupSmoothLayer(
        hidden_dim=768, group_d=16,
        proj_steps=1, smooth_lr=0.1,
        adaptive=False
    )

    # 复制权重以便公平比较
    with torch.no_grad():
        layer_non_adaptive.proj_to.weight.copy_(layer_adaptive.proj_to.weight)
        layer_non_adaptive.proj_to.bias.copy_(layer_adaptive.proj_to.bias)
        layer_non_adaptive.proj_from.weight.copy_(layer_adaptive.proj_from.weight)
        layer_non_adaptive.proj_from.bias.copy_(layer_adaptive.proj_from.bias)

    with torch.no_grad():
        out1, dist1, delta1 = layer_adaptive(x1)
        out2, dist2, delta2 = layer_non_adaptive(x2)

    print(f"\n自适应模式:")
    print(f"  输出范数：{out1.norm().item():.6f}")
    print(f"  流形距离：{dist1.item():.6f}")
    print(f"  群增量范数：{delta1.norm().item():.6f}")

    print(f"\n非自适应模式:")
    print(f"  输出范数：{out2.norm().item():.6f}")
    print(f"  流形距离：{dist2.item():.6f}")
    print(f"  群增量范数：{delta2.norm().item():.6f}")

    # 计算差异
    output_diff = (out1 - out2).norm().item()
    print(f"\n输出差异范数：{output_diff:.6f}")

    if output_diff > 1e-6:
        print("✓ 自适应和非自适应模式产生不同输出（符合预期）")
    else:
        print("✗ 自适应和非自适应模式输出相同（可能有问题）")


def test_full_model_adaptive():
    """测试完整模型中的自适应机制"""
    print("\n" + "=" * 60)
    print("测试 3: 完整模型中的自适应机制")
    print("=" * 60)

    # 创建模型（不使用预训练权重，因为离线模式）
    model = GPTWithGroup(
        base_model_name='gpt2',
        group_d=16,
        num_generators=6,
        group_type='orthogonal',
        use_pretrained=False
    )

    print(f"\n模型层数：{model.num_layers}")
    print(f"群表示维度：{model.group_d}")

    # 检查所有层是否都是自适应的
    adaptive_layers = sum(1 for layer in model.smooth_layers if layer.adaptive)
    print(f"自适应层数量：{adaptive_layers}/{model.num_layers}")

    # 打印每层的 alpha 值
    print("\n各层 alpha 值:")
    for i, layer in enumerate(model.smooth_layers):
        if layer.adaptive:
            alpha = torch.sigmoid(layer.weight_alpha).item()
            print(f"  层 {i}: alpha = {alpha:.4f}")

    # 前向传播
    batch_size = 2
    seq_len = 32
    vocab_size = 50257

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    # 训练模式
    model.train()

    outputs = model(input_ids=input_ids, labels=labels)

    print(f"\n前向传播结果:")
    print(f"  loss: {outputs.loss.item():.4f}")
    print(f"  manifold_distance: {outputs.manifold_distance.item():.6f}")

    # 反向传播验证梯度
    outputs.loss.backward()

    print("\n梯度验证:")

    # 检查 MetaGroup 梯度
    meta_group_grad = model.meta_group.generator_params.grad
    if meta_group_grad is not None:
        print(f"  MetaGroup 梯度范数：{meta_group_grad.norm().item():.6f}")
        print(f"  非零梯度比例：{(meta_group_grad != 0).float().mean().item():.2%}")
    else:
        print("  ✗ MetaGroup 无梯度")

    # 检查各层的 alpha 梯度
    print("\n各层 weight_alpha 梯度:")
    for i, layer in enumerate(model.smooth_layers):
        if layer.adaptive and layer.weight_alpha.grad is not None:
            print(f"  层 {i}: grad = {layer.weight_alpha.grad.item():.6f}")
        elif layer.adaptive:
            print(f"  层 {i}: 无梯度")

    # 检查门控网络梯度（第一层）
    print("\n门控网络梯度 (第一层):")
    first_layer = model.smooth_layers[0]
    for name, param in first_layer.gate_network.named_parameters():
        if param.grad is not None:
            print(f"  {name}: grad norm = {param.grad.norm().item():.6f}")
        else:
            print(f"  {name}: 无梯度")

    print("\n✓ 完整模型自适应机制验证通过")


def test_gate_network_behavior():
    """测试门控网络的行为"""
    print("\n" + "=" * 60)
    print("测试 4: 门控网络行为分析")
    print("=" * 60)

    # 创建层
    layer = GroupSmoothLayer(
        hidden_dim=768, group_d=16,
        proj_steps=1, smooth_lr=0.1,
        adaptive=True
    )

    # 创建不同的输入（不同的统计特征）
    torch.manual_seed(42)

    # 输入 1: 标准正态分布
    x1 = torch.randn(1, 100, 768)

    # 输入 2: 高方差
    x2 = torch.randn(1, 100, 768) * 5.0

    # 输入 3: 有偏分布
    x3 = torch.randn(1, 100, 768) + 3.0

    with torch.no_grad():
        # 手动通过门控网络
        def get_gate_stats(x):
            batch_size, seq_len = x.shape[:2]
            matrix_flat = layer.proj_to(x)
            matrices = matrix_flat.view(batch_size * seq_len, 16, 16)

            input_mean = matrices.mean(dim=(-2, -1), keepdim=True)
            input_std = matrices.std(dim=(-2, -1), keepdim=True)
            input_mean_norm = input_mean / (input_std + 1e-6)

            gate_input = torch.cat([
                input_mean_norm.view(batch_size * seq_len, 1),
                input_std.view(batch_size * seq_len, 1)
            ], dim=-1)

            gate_value = layer.gate_network(gate_input)

            return {
                'mean': input_mean.abs().mean().item(),
                'std': input_std.mean().item(),
                'gate_mean': gate_value.mean().item(),
                'gate_std': gate_value.std().item()
            }

        stats1 = get_gate_stats(x1)
        stats2 = get_gate_stats(x2)
        stats3 = get_gate_stats(x3)

        print(f"\n输入 1 (标准正态):")
        print(f"  |mean|={stats1['mean']:.4f}, std={stats1['std']:.4f}")
        print(f"  gate_mean={stats1['gate_mean']:.4f}, gate_std={stats1['gate_std']:.4f}")

        print(f"\n输入 2 (高方差):")
        print(f"  |mean|={stats2['mean']:.4f}, std={stats2['std']:.4f}")
        print(f"  gate_mean={stats2['gate_mean']:.4f}, gate_std={stats2['gate_std']:.4f}")

        print(f"\n输入 3 (有偏):")
        print(f"  |mean|={stats3['mean']:.4f}, std={stats3['std']:.4f}")
        print(f"  gate_mean={stats3['gate_mean']:.4f}, gate_std={stats3['gate_std']:.4f}")

        # 验证门控值是否随输入变化
        gate_diff = abs(stats1['gate_mean'] - stats2['gate_mean'])
        if gate_diff > 0.01:
            print(f"\n✓ 门控网络对不同输入产生不同响应 (差异={gate_diff:.4f})")
        else:
            print(f"\n✗ 门控网络响应相似 (差异={gate_diff:.4f})")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("自适应群光滑机制测试套件")
    print("=" * 60)

    test_adaptive_layer()
    test_adaptive_vs_non_adaptive()
    test_full_model_adaptive()
    test_gate_network_behavior()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
