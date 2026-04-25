#!/usr/bin/env python3
"""
只训练 GPT-2 baseline 30 轮，用于公平对比 GMGD-1（30 轮训练）
更新 length_extrapolation_results/results.json 中的 GPT-2 数据
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
import random
import re

from experiments.length_extrapolation import (
    LengthExtrapolationDataset,
    evaluate_length_extrapolation
)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# === 加载 tokenizer ===
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_EVALUATE_OFFLINE'] = '1'
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

# === 创建数据集（与 GMGD 训练相同的 seed=42） ===
dataset = LengthExtrapolationDataset(
    train_length=10,
    test_lengths=[10, 15, 20, 30, 50, 80, 100],
    samples_per_split=500,
    tokenizer=tokenizer,
    seed=42
)
train_loader = dataset.get_train_loader(batch_size=16)

# === GPT-2 模型 ===
class GPT2Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, labels=None, **kwargs):
        outputs = self.model(input_ids=input_ids, labels=labels, **kwargs)
        return {'loss': outputs.loss, 'logits': outputs.logits}

    def generate(self, input_ids, max_new_tokens=20, do_sample=False, **kwargs):
        return self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **kwargs
        )

print("Loading GPT-2...")
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_wrapper = GPT2Wrapper(gpt2_model).to(device)

# === 训练 ===
optimizer = torch.optim.AdamW(gpt2_wrapper.parameters(), lr=1e-4)
train_losses = []

for epoch in range(30):
    gpt2_wrapper.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/30")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = gpt2_wrapper(input_ids=input_ids, labels=labels)
        loss = outputs['loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches
    train_losses.append(avg_loss)
    print(f"Epoch {epoch + 1}/30 — avg_loss = {avg_loss:.4f}")

# === 保存 ===
save_path = os.path.join(os.path.dirname(__file__), '..', 'length_extrapolation_results', 'gpt2_baseline_30ep')
os.makedirs(save_path, exist_ok=True)
torch.save(gpt2_wrapper.state_dict(), os.path.join(save_path, 'gpt2_30ep.pt'))
print(f"Saved GPT-2 30-epoch model to {save_path}")

# === 评估 ===
print("\n=== Evaluation ===")
results = {
    'train_losses': train_losses,
}

for length in [10, 15, 20, 30, 50, 80, 100]:
    test_loader = dataset.get_test_loader(length, batch_size=16)
    acc = evaluate_length_extrapolation(
        gpt2_wrapper, test_loader, tokenizer, device=device, debug=(length == 10)
    )
    results[length] = acc
    print(f"  Length {length}: accuracy = {acc:.2%}")

# === 更新 results.json ===
results_json_path = os.path.join(os.path.dirname(__file__), '..', 'length_extrapolation_results', 'results.json')
if os.path.exists(results_json_path):
    with open(results_json_path, 'r') as f:
        existing = json.load(f)
else:
    existing = {}

existing['GPT-2'] = results

with open(results_json_path, 'w') as f:
    json.dump(existing, f, indent=2)
print(f"\nUpdated {results_json_path}")
print("Done!")
