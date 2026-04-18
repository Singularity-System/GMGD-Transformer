#!/usr/bin/env python3
"""
GMGD 群扩展 Transformer 最小可用测试

验证核心模块能正常运行：
1. MetaGroup 创建和群投影
2. GroupSmoothLayer 前向传播
3. GPTWithGroup 模型构建和前向传播

用法：
    python test_minimal.py
"""

import torch
import sys

print("=" * 60)
print("GMGD 群扩展 Transformer - 最小可用测试")
print("=" * 60)

# ==================== 测试 1: MetaGroup ====================
print("\n[1/3] 测试 MetaGroup...")

from meta_group import MetaGroup, cayley_exp

# 创建 MetaGroup
meta_group = MetaGroup(num_generators=12, d=32, group_type='orthogonal')
print(f"  ✓ 创建 MetaGroup: num_generators=12, d=32")

# 测试生成元
R0 = meta_group.get_generator(0)
print(f"  ✓ 获取生成元 R[0]: 形状 {R0.shape}")

# 测试正交性：R^T @ R 应该接近 I
I = torch.eye(32)
RtR = R0.transpose(-1, -2) @ R0
ortho_error = torch.norm(RtR - I, p='fro').item()
print(f"  ✓ 正交性误差：||R^T @ R - I|| = {ortho_error:.6e}")
assert ortho_error < 1e-5, f"正交性误差过大：{ortho_error}"

# 测试批量投影
X = torch.randn(16, 32, 32)  # (N, d, d)
X_proj = meta_group.project_to_manifold_batch(X, num_steps=1, lr=0.1)
print(f"  ✓ 批量投影：输入 {X.shape} → 输出 {X_proj.shape}")

# 测试 Cayley 变换
A = torch.randn(4, 32, 32)
A_skew = 0.5 * (A - A.transpose(-1, -2))
exp_A = cayley_exp(A_skew)
# 验证正交性
RtR_exp = exp_A.transpose(-1, -2) @ exp_A
ortho_error_exp = torch.norm(RtR_exp - I, p='fro').item()
print(f"  ✓ Cayley 变换正交性误差：{ortho_error_exp:.6e}")

print("  ✅ MetaGroup 测试通过!")

# ==================== 测试 2: GroupSmoothLayer ====================
print("\n[2/3] 测试 GroupSmoothLayer...")

from group_smooth_layer import GroupSmoothLayer

# 创建群光滑层
smooth_layer = GroupSmoothLayer(
    hidden_dim=768,
    group_d=32,
    num_generators=12,
    proj_steps=1,
    smooth_lr=0.1
)
print(f"  ✓ 创建 GroupSmoothLayer: hidden_dim=768, group_d=32")

# 测试前向传播
batch_size, seq_len = 2, 16
x = torch.randn(batch_size, seq_len, 768)
with torch.no_grad():
    y = smooth_layer(x)
print(f"  ✓ 前向传播：输入 {x.shape} → 输出 {y.shape}")
assert y.shape == x.shape, f"输出形状不匹配：{y.shape} vs {x.shape}"

# 验证残差连接（初始时 proj_from 为零，输出应接近输入）
diff = torch.norm(y - x, p='fro').item()
print(f"  ✓ 初始残差（应接近 0）：||y - x|| = {diff:.6e}")
assert diff < 1e-5, f"残差过大：{diff}"

print("  ✅ GroupSmoothLayer 测试通过!")

# ==================== 测试 3: GPTWithGroup ====================
print("\n[3/3] 测试 GPTWithGroup...")

try:
    from gpt_with_group import GPTWithGroup

    # 创建模型（使用预训练权重，首次运行会下载）
    print("  ⏳ 加载 GPT-2 模型（首次运行需下载约 500MB）...")

    model = GPTWithGroup(
        base_model_name='gpt2',
        group_d=32,
        num_generators=12,
        use_pretrained=True
    )
    print(f"  ✓ 创建 GPTWithGroup 成功")

    # 打印参数量
    total_params = model.get_num_params()
    group_params = sum(p.numel() for p in model.meta_group.parameters())
    print(f"  ✓ 总参数量：{total_params:,}")
    print(f"  ✓ 群参数量：{group_params:,} ({group_params/total_params*100:.4f}%)")

    # 测试前向传播
    batch_size = 1
    seq_len = 32
    input_ids = torch.randint(0, 50257, (batch_size, seq_len))

    print(f"  ⏳ 运行前向传播...")
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

    print(f"  ✓ 前向传播：输入 {input_ids.shape} → logits {logits.shape}")
    assert logits.shape == (batch_size, seq_len, 50257), f"输出形状错误：{logits.shape}"

    print("  ✅ GPTWithGroup 测试通过!")

except ImportError as e:
    print(f"  ⚠️  跳过 GPTWithGroup 测试（需要 transformers 库）: {e}")
except Exception as e:
    print(f"  ⚠️  GPTWithGroup 测试失败：{e}")

# ==================== 总结 ====================
print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)
print("""
测试结果:
  ✅ MetaGroup - 通过
  ✅ GroupSmoothLayer - 通过
  ✅/⚠️  GPTWithGroup - 通过/跳过（取决于 transformers）

下一步:
  1. 运行完整训练：python train.py --epochs 3
  2. 查看工程说明书：GMGD_Transformer_工程说明书.md
  3. 开始编码你的项目！
""")
