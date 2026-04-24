#!/usr/bin/env python3
"""测试干净 GMGD 版本的生成"""
import sys, os
sys.path.insert(0, os.path.dirname('..'))

import torch
from transformers import GPT2Tokenizer
from core import GPTWithGroup, MetaGroup

device = torch.device('cpu')

# 创建 GMGD 模型（从零初始化，不使用预训练）
model = GPTWithGroup(
    base_model_name='gpt2',
    group_d=16,
    num_generators=6,
    group_type='orthogonal',
    use_pretrained=False
).to(device)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

test_cases = [
    ("Expr: 3 + 5 =", 8),
    ("Expr: 7 + 4 + 1 =", 12),
    ("Expr: 3 + 5 - 2 =", 6),
]

print("=" * 60)
print("测试 GMGD 模型生成（未训练，验证 generate 修复）")
print("=" * 60)

import re

for text, expected in test_cases:
    enc = tokenizer(text, return_tensors='pt')
    input_ids = enc['input_ids'].to(device)
    input_len = input_ids.shape[1]

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = generated[0][input_len:]
    gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    nums = re.findall(r'-?\d+', gen_text)
    pred = int(nums[0]) if nums else None

    print(f"  {text} → {gen_text!r} → pred={pred}, expected={expected}")

# 也测试 GPT-2 预训练基线
print("\n" + "=" * 60)
print("测试 GPT-2 预训练基线")
print("=" * 60)

from transformers import GPT2LMHeadModel
gpt2 = GPT2LMHeadModel.from_pretrained('gpt2').to(device).eval()

for text, expected in test_cases:
    enc = tokenizer(text, return_tensors='pt')
    input_ids = enc['input_ids'].to(device)
    input_len = input_ids.shape[1]

    with torch.no_grad():
        generated = gpt2.generate(
            input_ids=input_ids,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = generated[0][input_len:]
    gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    nums = re.findall(r'-?\d+', gen_text)
    pred = int(nums[0]) if nums else None

    print(f"  {text} → {gen_text!r} → pred={pred}, expected={expected}")
