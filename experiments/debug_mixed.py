#!/usr/bin/env python3
"""调试混合诊断"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
import re
from transformers import GPT2Tokenizer
from core import GPTWithGroup
from safetensors.torch import load_file

device = torch.device('cpu')

# 加载模型
model_path = './checkpoints/gmgd_20epoch/checkpoint_epoch_3'
print(f"Loading model from {model_path}...")

model = GPTWithGroup(
    base_model_name='gpt2',
    group_d=16,
    num_generators=6,
    group_type='orthogonal',
    use_pretrained=False
).to(device)

state_dict = load_file(os.path.join(model_path, 'model.safetensors'))
new_state = {k.replace('transformer.', ''): v for k, v in state_dict.items()}
model.base_model.load_state_dict(new_state, strict=False)

group_state = torch.load(os.path.join(model_path, 'group_state.pt'), map_location=device, weights_only=False)
model.meta_group.load_state_dict(group_state['meta_group'])
for i, smooth_layer in enumerate(model.smooth_layers):
    layer_state = group_state.get(f'smooth_layer_{i}', {})
    if layer_state:
        smooth_layer.load_state_dict(layer_state)

model.eval()
print("Model loaded.")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# 测试样本
test_cases = [
    "计算：3+5-2，注意不要算错",
    "5+9-3 的结果是",
    "表达式：10-2+1，求值",
    "3+2 = ?",
]

print("\n" + "=" * 60)
print("调试输出")
print("=" * 60)

for text in test_cases:
    print(f"\n输入：{text}")

    # 期望答案
    try:
        expr = re.search(r'[\d\+\-\s]+', text).group()
        expected = eval(expr.replace(' ', ''))
        print(f"期望答案：{expected}")
    except:
        print("无法解析期望答案")

    # Tokenize
    enc = tokenizer(text, return_tensors='pt', padding=True)
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)

    print(f"Input shape: {input_ids.shape}")
    print(f"Input tokens: {tokenizer.convert_ids_to_tokens(input_ids[0])}")

    # 生成
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            max_length=input_ids.shape[1] + 8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask
        )

    pred_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"生成：{pred_text}")

    # 提取数字
    nums = re.findall(r'-?\d+', pred_text)
    if nums:
        print(f"提取的数字：{nums[-1]}")
    else:
        print("未提取到数字")
