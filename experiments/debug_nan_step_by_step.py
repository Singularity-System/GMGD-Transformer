#!/usr/bin/env python3
"""精细追踪 NaN"""
import sys, os
sys.path.insert(0, os.path.dirname('..'))

import torch
from transformers import GPT2Tokenizer
from core import GPTWithGroup
import random

device = torch.device('cpu')
torch.manual_seed(42)
random.seed(42)

model = GPTWithGroup(
    base_model_name='gpt2',
    group_d=16,
    num_generators=6,
    group_type='orthogonal',
    use_pretrained=True
).to(device)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

# 生成批量数据
def generate_expr(max_length=10, seed=None):
    if seed is not None:
        random.seed(seed)
    numbers = [random.randint(0, 7) for _ in range(max_length // 2 + 1)]
    operators = [random.choice(['+', '-']) for _ in range(len(numbers) - 1)]
    result = numbers[0]
    expr_parts = [str(numbers[0])]
    for i, (op, num) in enumerate(zip(operators, numbers[1:])):
        if op == '+': result += num
        else:
            if result < num: op = '+'; result += num
            else: result -= num
        expr_parts.append(op); expr_parts.append(str(num))
    expr = ' '.join(expr_parts)
    try: result = eval(expr.replace(' ', ''))
    except: result = 0
    return expr, result

samples = []
for i in range(16):
    expr, result = generate_expr(10, seed=42+i)
    input_text = f"Expr: {expr} ="
    target_text = f" {result}"
    full_text = input_text + target_text
    enc = tokenizer(full_text, truncation=True, return_tensors='pt')
    input_ids = enc['input_ids'].squeeze(0)
    labels = input_ids.clone()
    input_len = len(tokenizer.encode(input_text))
    labels[:input_len] = -100
    samples.append({'input_ids': input_ids, 'labels': labels})

max_len = max(len(x['input_ids']) for x in samples)
batch_input_ids = torch.stack([
    torch.cat([torch.full((max_len - len(s['input_ids']),), tokenizer.pad_token_id), s['input_ids']])
    for s in samples
])
batch_labels = torch.stack([
    torch.cat([torch.full((max_len - len(s['labels']),), -100), s['labels']])
    for s in samples
])

# 逐层追踪
print("逐层追踪 NaN")
model.train()

with torch.no_grad():
    input_ids = batch_input_ids
    input_shape = input_ids.size()
    batch_size, seq_len = input_shape
    device = input_ids.device

    position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

    inputs_embeds = model.base_model.wte(input_ids)
    position_embeds = model.base_model.wpe(position_ids)
    hidden_states = inputs_embeds + position_embeds

    print(f"嵌入后 NaN: {torch.isnan(hidden_states).any().item()}")

    for i, (block, smooth_layer) in enumerate(zip(model.base_model.h, model.smooth_layers)):
        hidden_states = block(hidden_states)
        if torch.isnan(hidden_states).any():
            print(f"Transformer block {i} 产生 NaN!")
            break

        # 群光滑层——手动追踪
        original_shape = hidden_states.shape
        batch_size_s, seq_len_s, _ = hidden_states.shape
        matrix_flat = smooth_layer.proj_to(hidden_states)
        matrices = matrix_flat.view(batch_size_s * seq_len_s, model.group_d, model.group_d)

        print(f"Layer {i}: proj_to 后 NaN={torch.isnan(matrix_flat).any().item()}, "
              f"matrices NaN={torch.isnan(matrices).any().item()}, "
              f"matrices max={matrices.abs().max().item():.4f}")

        if torch.isnan(matrices).any():
            print(f"  → proj_to 产生 NaN!")
            break

        # 门控
        input_mean = matrices.mean(dim=(-2, -1), keepdim=True)
        input_std = matrices.std(dim=(-2, -1), keepdim=True)
        input_mean_norm = input_mean / (input_std + 1e-6)
        gate_input = torch.cat([
            input_mean_norm.view(batch_size_s * seq_len_s, 1),
            input_std.view(batch_size_s * seq_len_s, 1)
        ], dim=-1)
        gate_value = smooth_layer.gate_network(gate_input).view(batch_size_s * seq_len_s, 1, 1)
        matrices_gated = gate_value * matrices

        print(f"Layer {i}: gate NaN={torch.isnan(gate_value).any().item()}, "
              f"matrices_gated NaN={torch.isnan(matrices_gated).any().item()}")

        if torch.isnan(matrices_gated).any():
            print(f"  → gate_network 产生 NaN!")
            break

        # project_to_manifold_batch
        gens = model.meta_group.all_generators()
        print(f"Layer {i}: gens NaN={torch.isnan(gens).any().item()}, gens max={gens.abs().max().item():.4f}")

        if torch.isnan(gens).any():
            print(f"  → all_generators 产生 NaN!")
            break

        X_proj = matrices_gated.clone()
        coeffs = torch.einsum('nij,kij->nk', X_proj, gens)
        print(f"Layer {i}: coeffs NaN={torch.isnan(coeffs).any().item()}, coeffs max={coeffs.abs().max().item():.4f}")

        if torch.isnan(coeffs).any():
            print(f"  → einsum coeffs 产生 NaN!")
            break

        approx = torch.einsum('nk,kij->nij', coeffs, gens)
        print(f"Layer {i}: approx NaN={torch.isnan(approx).any().item()}, approx max={approx.abs().max().item():.4f}")

        if torch.isnan(approx).any():
            print(f"  → einsum approx 产生 NaN!")
            break

        grad = X_proj - approx
        lr = 0.1
        X_proj = X_proj - lr * grad

        print(f"Layer {i}: X_proj after update NaN={torch.isnan(X_proj).any().item()}")

        if torch.isnan(X_proj).any():
            print(f"  → project_to_manifold 产生 NaN!")
            break

        # 继续光滑层剩余部分
        matrices_smooth = X_proj
        smooth_flat = matrices_smooth.view(batch_size_s, seq_len_s, -1)
        smooth_hidden = smooth_layer.proj_from(smooth_flat)
        alpha = torch.sigmoid(smooth_layer.weight_alpha)
        hidden_states = hidden_states + alpha * smooth_hidden

        if torch.isnan(hidden_states).any():
            print(f"Layer {i}: 残差连接后 NaN!")
            break
        else:
            print(f"Layer {i}: 通过 ✓")
