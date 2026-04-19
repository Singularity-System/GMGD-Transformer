#!/usr/bin/env python3
"""
快速长度外推评估
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
import random
import re
import json

from core import GPTWithGroup
from safetensors.torch import load_file


def generate_no_carry_expression(max_length: int = 10, seed: int = None) -> tuple:
    """生成不进位的加减法表达式"""
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


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples: int, expr_length: int, tokenizer, seed: int = 0):
        random.seed(seed)
        self.samples = []
        for i in range(num_samples):
            expr, result = generate_no_carry_expression(expr_length, seed=seed+i)
            text = f"{expr} = {result}"
            enc = tokenizer(text, padding='max_length', max_length=expr_length+16, truncation=True, return_tensors='pt')
            self.samples.append({
                'input_ids': enc['input_ids'][0],
                'answer': result,
                'expr': expr
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    answers = torch.tensor([item['answer'] for item in batch])
    exprs = [item['expr'] for item in batch]
    return {'input_ids': input_ids, 'answer': answers, 'expr': exprs}


def evaluate(model, tokenizer, device, expr_length: int, num_samples: int = 100):
    """评估模型在指定长度上的准确率"""
    dataset = SimpleDataset(num_samples, expr_length, tokenizer, seed=42+expr_length)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Length {expr_length}", leave=False):
            input_ids = batch['input_ids'].to(device)
            answers = batch['answer']

            # 生成预测
            preds = []
            for ids in input_ids:
                pred_ids = model.generate(
                    input_ids=ids.unsqueeze(0),
                    max_length=ids.shape[0] + 8,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                pred_text = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
                preds.append(pred_text)

            # 解析预测结果
            for pred_text, true_val in zip(preds, answers):
                try:
                    nums = re.findall(r'-?\d+', pred_text)
                    if nums:
                        pred_val = int(nums[-1])
                        if pred_val == true_val:
                            correct += 1
                except:
                    pass
                total += 1

    return correct / max(total, 1)


def load_model_from_checkpoint(model_path: str, device: torch.device) -> GPTWithGroup:
    """从检查点加载模型"""
    # 读取配置
    config_path = os.path.join(model_path, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    print(f"Config: group_d={config['group_d']}, num_generators={config['num_generators']}")

    # 创建模型
    model = GPTWithGroup(
        base_model_name='gpt2',
        group_d=config['group_d'],
        num_generators=config['num_generators'],
        group_type=config['group_type'],
        use_pretrained=False
    ).to(device)

    # 加载 base_model 权重
    base_state = load_file(os.path.join(model_path, 'model.safetensors'))
    # 移除 transformer 前缀
    new_state = {k.replace('transformer.', ''): v for k, v in base_state.items()}
    model.base_model.load_state_dict(new_state)

    # 加载群状态
    group_state = torch.load(os.path.join(model_path, 'group_state.pt'), map_location=device, weights_only=False)
    model.meta_group.load_state_dict(group_state['meta_group'])
    for i, smooth_layer in enumerate(model.smooth_layers):
        layer_state = group_state.get(f'smooth_layer_{i}', {})
        if layer_state:
            smooth_layer.load_state_dict(layer_state)

    model.to(device)
    model.eval()
    return model


if __name__ == '__main__':
    print("=" * 60)
    print("GMGD 长度外推快速评估")
    print("=" * 60)

    device = torch.device('cpu')
    print(f"\nUsing device: {device}")

    # 加载模型
    model_path = './checkpoints/gmgd_20epoch/checkpoint_epoch_3'
    print(f"\nLoading model from {model_path}...")

    model = load_model_from_checkpoint(model_path, device)
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # 评估不同长度
    test_lengths = [5, 10, 20, 40, 60, 80]
    results = {}

    print("\n" + "=" * 60)
    print("长度外推测试结果")
    print("=" * 60)

    for length in test_lengths:
        acc = evaluate(model, tokenizer, device, length, num_samples=50)
        results[length] = acc
        print(f"长度 {length:3d}: 准确率 = {acc:.2%}")

    print("\n" + "=" * 60)
    print("摘要")
    print("=" * 60)
    for length, acc in results.items():
        print(f"  长度 {length:3d}: {acc:.2%}")

    # 与 GPT-2 基线对比
    print("\n" + "=" * 60)
    print("对比 GPT-2 基线（历史数据，长度 80 约 1%）")
    print("=" * 60)
    gpt2_baseline = {5: 0.95, 10: 0.90, 20: 0.50, 40: 0.10, 60: 0.03, 80: 0.01}
    print("长度\tGMGD\tGPT-2\t提升")
    for length in test_lengths:
        gmgd_acc = results[length]
        gpt2_acc = gpt2_baseline.get(length, 0)
        improvement = gmgd_acc - gpt2_acc
        print(f"{length}\t{gmgd_acc:.2%}\t{gpt2_acc:.2%}\t{improvement:+.2%}")
