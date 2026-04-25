#!/usr/bin/env python3
"""追踪带梯度时前向传播中 NaN 的具体位置"""
import sys, os
sys.path.insert(0, os.path.dirname('..'))

import torch
from core import GPTWithGroup
from transformers import GPT2Tokenizer
import random
import torch.nn as nn

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

print(f"batch_input_ids shape: {batch_input_ids.shape}")
print(f"batch_labels shape: {batch_labels.shape}")
print(f"Label distribution: unique={torch.unique(batch_labels)}")
print(f"Non -100 labels: {(batch_labels != -100).sum().item()}")

model.train()

# 手动追踪 forward 流程
input_ids = batch_input_ids
input_shape = input_ids.size()
batch_size, seq_len = input_shape
device = input_ids.device

position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

inputs_embeds = model.base_model.wte(input_ids)
position_embeds = model.base_model.wpe(position_ids)
hidden_states = inputs_embeds + position_embeds

print(f"\n嵌入后: NaN={torch.isnan(hidden_states).any().item()}, Inf={torch.isinf(hidden_states).any().item()}, max={hidden_states.abs().max().item():.4f}")

for i, (block, smooth_layer) in enumerate(zip(model.base_model.h, model.smooth_layers)):
    hidden_states = block(hidden_states)
    hidden_states, m_dist, delta = smooth_layer(hidden_states)
    nan_flag = torch.isnan(hidden_states).any().item()
    if i % 3 == 0 or nan_flag:
        print(f"Layer {i}: NaN={nan_flag}, Inf={torch.isinf(hidden_states).any().item()}, max={hidden_states.abs().max().item():.4f}, m_dist={m_dist:.4f}")

print(f"\nln_f 前: NaN={torch.isnan(hidden_states).any().item()}, max={hidden_states.abs().max().item():.4f}")

hidden_states = model.base_model.ln_f(hidden_states)
print(f"ln_f 后: NaN={torch.isnan(hidden_states).any().item()}, Inf={torch.isinf(hidden_states).any().item()}, max={hidden_states.abs().max().item():.4f}")

logits = model.lm_head(hidden_states)
print(f"lm_head 后: NaN={torch.isnan(logits).any().item()}, Inf={torch.isinf(logits).any().item()}, max={logits.abs().max().item():.4f}")

# 检查 logits 中的极端值
print(f"logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
print(f"logits > 100: {(logits > 100).sum().item()}")
print(f"logits < -100: {(logits < -100).sum().item()}")

# 计算 loss
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous() if 'labels' in dir() else batch_labels[..., 1:].contiguous()

print(f"\nshift_logits shape: {shift_logits.shape}, NaN={torch.isnan(shift_logits).any().item()}")
print(f"shift_labels shape: {shift_labels.shape}, unique={torch.unique(shift_labels)[:10]}")

loss_fct = nn.CrossEntropyLoss()

# 检查是否有有效标签
valid_labels = (shift_labels != -100).sum().item()
print(f"Valid (non -100) labels: {valid_labels}/{shift_labels.numel()}")

# 检查哪些位置的 logits 是 NaN
nan_logits = torch.isnan(shift_logits).any(dim=-1)  # (B, S-1)
print(f"Positions with NaN logits: {nan_logits.sum().item()}")

# 在有效标签位置检查
valid_positions = shift_labels != -100
nan_at_valid = nan_logits & valid_positions
print(f"NaN at valid label positions: {nan_at_valid.sum().item()}")

try:
    loss = loss_fct(shift_logits.view(-1, model.vocab_size), shift_labels.view(-1))
    print(f"loss: {loss}")
except Exception as e:
    print(f"Loss computation error: {e}")
