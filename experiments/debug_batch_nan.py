#!/usr/bin/env python3
"""诊断批量训练时的 NaN"""
import sys, os
sys.path.insert(0, os.path.dirname('..'))

import torch
from core import GPTWithGroup
from transformers import GPT2Tokenizer
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

# 模拟训练数据生成
def generate_no_carry_expression(max_length=10, seed=None):
    if seed is not None:
        random.seed(seed)
    numbers = [random.randint(0, 7) for _ in range(max_length // 2 + 1)]
    operators = [random.choice(['+', '-']) for _ in range(len(numbers) - 1)]
    result = numbers[0]
    expr_parts = [str(numbers[0])]
    for i, (op, num) in enumerate(zip(operators, numbers[1:])):
        if op == '+':
            result += num
        else:
            if result < num:
                op = '+'
                result += num
            else:
                result -= num
        expr_parts.append(op)
        expr_parts.append(str(num))
    expr = ' '.join(expr_parts)
    tokens = expr.split()
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
        if tokens[-1] in ['+', '-']:
            tokens = tokens[:-1]
        expr = ' '.join(tokens)
    try:
        result = eval(expr.replace(' ', ''))
    except:
        result = 0
    return expr, result

# 生成训练样本
print("=" * 60)
print("生成训练数据")
print("=" * 60)

samples = []
for i in range(16):
    expr, result = generate_no_carry_expression(10, seed=42+i)
    input_text = f"Expr: {expr} ="
    target_text = f" {result}"
    full_text = input_text + target_text

    enc = tokenizer(full_text, truncation=True, return_tensors='pt')
    input_ids = enc['input_ids'].squeeze(0)
    labels = input_ids.clone()
    input_len = len(tokenizer.encode(input_text))
    labels[:input_len] = -100

    samples.append({'input_ids': input_ids, 'labels': labels})
    print(f"  样本 {i}: {full_text!r} (len={input_ids.shape[0]})")

# 手动批处理（左 padding）
print("\n" + "=" * 60)
print("左 padding 批处理")
print("=" * 60)

max_len = max(len(x['input_ids']) for x in samples)
print(f"max_len: {max_len}")

padded_input_ids = []
padded_labels = []
for item in samples:
    pad_len = max_len - len(item['input_ids'])
    input_ids = torch.cat([
        torch.full((pad_len,), tokenizer.pad_token_id),
        item['input_ids']
    ])
    labels = torch.cat([
        torch.full((pad_len,), -100),
        item['labels']
    ])
    padded_input_ids.append(input_ids)
    padded_labels.append(labels)

batch_input_ids = torch.stack(padded_input_ids)
batch_labels = torch.stack(padded_labels)

print(f"batch_input_ids shape: {batch_input_ids.shape}")
print(f"batch_labels shape: {batch_labels.shape}")
print(f"PAD token ID: {tokenizer.pad_token_id}")
print(f"EOS token ID: {tokenizer.eos_token_id}")
print(f"batch 中 PAD 比例: {(batch_input_ids == tokenizer.pad_token_id).float().mean().item():.2%}")

# 测试前向传播
print("\n" + "=" * 60)
print("批量前向传播")
print("=" * 60)

model.train()
try:
    outputs = model(input_ids=batch_input_ids, labels=batch_labels)
    loss = outputs.loss
    print(f"loss: {loss}")
    print(f"loss is finite: {torch.isfinite(loss).item()}")

    # 检查反向传播
    loss.backward()

    # 检查梯度
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"  NaN gradient in {name}")
            if torch.isinf(param.grad).any():
                print(f"  Inf gradient in {name}")

    print("反向传播完成，所有梯度正常")

    # 测试优化器步骤
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    outputs = model(input_ids=batch_input_ids, labels=batch_labels)
    outputs.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    print("优化器步骤完成")

    # 再跑一步
    optimizer.zero_grad()
    outputs = model(input_ids=batch_input_ids, labels=batch_labels)
    print(f"第二步 loss: {outputs.loss}")

except Exception as e:
    print(f"前向传播失败: {e}")
    import traceback
    traceback.print_exc()
