#!/usr/bin/env python3
"""
混合诊断任务训练脚本：算术 + 语言干扰

用于训练模型在语言干扰下仍能正确执行算术运算。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
import torch.nn as nn
import re
import random
import json
from datetime import datetime
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from core import GPTWithGroup
from core.dynamic_expander import DynamicGroupExpander
from data.arithmetic import generate_expression, evaluate_expression


# ==================== 模板 ====================
TRAIN_TEMPLATES = [
    "计算：{}，注意不要算错",
    "{}的结果是",
    "请帮我算一下{}等于多少",
]

TEST_TEMPLATES = [
    "表达式：{}，求值",
    "{} = ?",
]


# ==================== 数据生成 ====================
def generate_mixed_sample(expr_depth: int, template: str, seed: int = None) -> tuple:
    random.seed(seed)
    expr = generate_expression(max_depth=expr_depth, operators=['+', '-'], max_number=7)
    expr = expr.replace('(', '').replace(')', '')
    try:
        value = evaluate_expression(expr)
    except:
        value = 0
    input_text = template.format(expr)
    return input_text, value, expr


class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples: int, expr_depth: int, tokenizer, templates: list, seed: int = 42, max_len: int = 128):
        random.seed(seed)
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        for i in range(num_samples):
            input_text, value, expr = generate_mixed_sample(expr_depth, random.choice(templates), seed=seed+i)
            # 构建目标文本：只包含答案
            target_text = f" {value}"  # 前面加空格，因为 tokenizer 会这样分隔

            enc = tokenizer(input_text, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt')
            target_enc = tokenizer(target_text, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt')

            input_ids = enc['input_ids'][0]
            attention_mask = enc['attention_mask'][0]
            labels = target_enc['input_ids'][0]
            # 将 padding 部分的 labels 设为 -100
            labels[labels == tokenizer.pad_token_id] = -100

            self.samples.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'answer': value,
                'expr': expr,
                'input_text': input_text
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
        'answer': torch.tensor([b['answer'] for b in batch]),
        'expr': [b['expr'] for b in batch],
        'input_text': [b['input_text'] for b in batch]
    }


def evaluate(model, tokenizer, device, eval_dataset, batch_size=8):
    model.eval()
    loader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn)
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            answers = batch['answer']

            generated = model.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + 8,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=attention_mask
            )

            for gen_ids, true_val in zip(generated, answers):
                pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                nums = re.findall(r'-?\d+', pred_text)
                if nums:
                    pred_val = int(nums[-1])
                    if pred_val == true_val:
                        correct += 1
                total += 1

    return correct / max(total, 1)


def main():
    print("=" * 60)
    print("混合诊断任务训练")
    print("=" * 60)

    device = torch.device('cpu')
    print(f"Device: {device}")

    # 创建模型 - 启用动态群扩张
    print("\n创建模型（启用动态群扩张）...")
    model = GPTWithGroup(
        base_model_name='gpt2',
        group_d=16,
        num_generators=6,
        group_type='orthogonal',
        use_pretrained=False,
        enable_dynamic_expansion=True,  # 启用动态群扩张
        expansion_threshold=0.3,  # 较敏感的触发阈值
        max_generators_per_group=8  # 单群最大生成元数
    ).to(device)

    print(f"参数字数：{sum(p.numel() for p in model.parameters()):,}")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # 训练配置
    train_depth = 2  # 短表达式训练
    num_train = 500
    num_eval = 100
    epochs = 5
    batch_size = 8
    learning_rate = 1e-4
    expansion_warmup_steps = 50  # 前 50 步不触发扩张，让模型先稳定

    # 创建数据集
    print(f"\n创建数据集...")
    train_dataset = MixedDataset(num_train, train_depth, tokenizer, TRAIN_TEMPLATES, seed=42)
    eval_dataset = MixedDataset(num_eval, train_depth, tokenizer, TEST_TEMPLATES, seed=100)

    print(f"训练样本：{len(train_dataset)}")
    print(f"评估样本：{len(eval_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 优化器 - 群参数和 Transformer 一起训练
    # 群光滑层使用较小的学习率
    base_params = [p for n, p in model.base_model.named_parameters()]
    group_params = [p for n, p in model.named_parameters() if 'smooth_layers' in n or 'expander' in n]
    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': learning_rate},
        {'params': group_params, 'lr': learning_rate * 0.1}  # 群参数学习率更低
    ])
    print("训练策略：群参数和 Transformer 一起训练（群参数学习率 x0.1）")
    print("动态群扩张：启用")

    # 训练
    print(f"\n开始训练 {epochs} 个 epochs...")
    best_acc = 0
    global_step = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            answers = batch['answer']

            optimizer.zero_grad()

            # 动态群扩张：提供 target_group_state 和 global_step
            # 简化实现：使用单位阵作为目标（后续可根据任务类型提供更精确的目标）
            batch_size = input_ids.shape[0]
            target_group_state = torch.eye(
                model.group_d, device=device
            ).unsqueeze(0).expand(batch_size, -1, -1)

            outputs = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                target_group_state=target_group_state,
                global_step=global_step
            )
            loss = outputs.loss

            # 检查 NaN
            if not torch.isfinite(loss):
                print(f"Warning: Non-finite loss detected: {loss.item()}")
                optimizer.zero_grad()
                continue

            loss.backward()

            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                print(f"Warning: Gradient explosion detected, skipping step")
                optimizer.zero_grad()
                continue

            optimizer.step()

            # 检查扩张事件
            if outputs.expansion_info and outputs.expansion_info.get('triggered'):
                print(f"  [Step {global_step}] 群扩张：{outputs.expansion_info}")

            global_step += 1

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # 评估
        acc = evaluate(model, tokenizer, device, eval_dataset, batch_size=8)

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Eval Acc={acc:.2%}")

        if acc > best_acc:
            best_acc = acc
            # 保存最佳模型
            save_dir = './checkpoints/mixed_diagnostic/best_model'
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"  新最佳模型保存至 {save_dir}, acc={best_acc:.2%}")

    # 保存最终检查点
    save_dir = './checkpoints/mixed_diagnostic/final'
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # 保存训练配置
    config = {
        'train_depth': train_depth,
        'num_train': num_train,
        'num_eval': num_eval,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'best_acc': best_acc,
        'templates': TRAIN_TEMPLATES,
        'test_templates': TEST_TEMPLATES
    }
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n训练完成！最佳评估准确率：{best_acc:.2%}")
    print(f"最终检查点：{save_dir}")


if __name__ == '__main__':
    main()
