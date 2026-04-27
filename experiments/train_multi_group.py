#!/usr/bin/env python3
"""
多群 GroupAlgebra 四阶段训练协议

阶段 1: 分域预训练 — 单群，纯算术数据，群学习代数规则
阶段 2: 投影层预热 — 冻结生成元，训练投影层，学习注意力映射
阶段 3: 交替冻结 — 交替冻结群/投影层，避免耦合震荡
阶段 4: 髓鞘化 — 降低所有 lr，全参数微调，稳定收敛

用法：
    python experiments/train_multi_group.py --epochs 30 --samples 500
    python experiments/train_multi_group.py --epochs 1 --samples 50 --quick-test  # 快速测试
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import re
import random
import argparse
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm

from core import GPTWithGroup
from data.multi_domain_dataset import (
    MultiDomainDataset,
    create_multi_domain_datasets,
    get_domain_loader,
    collate_fn
)

device = torch.device('cpu')  # GPT-2 generate 在 MPS 上有 bug
print(f"Using device: {device}")


# ==================== 数据准备 ====================

def prepare_data(train_length: int = 10, samples: int = 500, seed: int = 42):
    """准备多域数据集"""
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    datasets = create_multi_domain_datasets(
        tokenizer=tokenizer,
        train_samples=samples,
        eval_samples=max(samples // 5, 20),
        train_length=train_length,
        seed=seed
    )

    return datasets, tokenizer


# ==================== 参数分组 ====================

def get_param_groups(model, group_lr=2e-3, proj_lr=5e-4, other_lr=1e-4):
    """三层学习率参数分组"""
    mg_params = set()
    proj_params = set()
    attn_params = set()

    for n, p in model.named_parameters():
        if 'meta_group' in n and 'generator_params' in n:
            mg_params.add(id(p))
        elif 'proj_to' in n or 'proj_from' in n or 'logit_alpha' in n:
            proj_params.add(id(p))
        elif 'attention' in n or 'query_proj' in n or 'group_keys' in n:
            attn_params.add(id(p))

    other_params = [
        p for p in model.parameters()
        if id(p) not in mg_params and id(p) not in proj_params and id(p) not in attn_params
    ]

    param_groups = []
    mg_list = [p for p in model.parameters() if id(p) in mg_params]
    if mg_list:
        param_groups.append({'params': mg_list, 'lr': group_lr})

    proj_list = [p for p in model.parameters() if id(p) in proj_params]
    if proj_list:
        param_groups.append({'params': proj_list, 'lr': proj_lr})

    attn_list = [p for p in model.parameters() if id(p) in attn_params]
    if attn_list:
        param_groups.append({'params': attn_list, 'lr': proj_lr})

    if other_params:
        param_groups.append({'params': other_params, 'lr': other_lr})

    return param_groups


def freeze_params(param_ids: set, model):
    """冻结指定参数"""
    for p in model.parameters():
        if id(p) in param_ids:
            p.requires_grad = False


def unfreeze_params(param_ids: set, model):
    """解冻指定参数"""
    for p in model.parameters():
        if id(p) in param_ids:
            p.requires_grad = True


# ==================== 训练循环 ====================

def train_epoch(
    model,
    loaders: List[DataLoader],
    optimizer,
    epoch: int,
    device,
    manifold_loss_weight: float = 0.5,
    global_step: int = 0,
    desc: str = "Training"
) -> dict:
    """
    训练一个 epoch

    Args:
        model: GPTWithGroup 模型
        loaders: 数据加载器列表（多域时多个）
        optimizer: 优化器
        epoch: 当前 epoch
        device: 设备
        manifold_loss_weight: 流形损失权重
        global_step: 全局步数
        desc: 进度条描述

    Returns:
        stats: 训练统计
    """
    model.train()
    total_loss = 0.0
    total_manifold = 0.0
    num_batches = 0

    # 循环所有加载器，取最长的那个
    max_batches = max(len(loader) for loader in loaders) if loaders else 0

    pbar = tqdm(range(max_batches), desc=f"Epoch {epoch} — {desc}")
    for step_idx in pbar:
        # 从不同域轮流采样
        loader_idx = step_idx % len(loaders)
        batch_idx = step_idx % len(loaders[loader_idx])
        batch = next(iter(DataLoader(
            loaders[loader_idx].dataset,
            batch_size=loaders[loader_idx].batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            sampler=torch.utils.data.RandomSampler(loaders[loader_idx].dataset, replacement=True, num_samples=loaders[loader_idx].batch_size)
        )))

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # 前向传播
        target_group_state = torch.eye(
            model.group_d, device=device
        ).unsqueeze(0).expand(input_ids.shape[0], -1, -1)

        outputs = model(
            input_ids=input_ids,
            labels=labels,
            target_group_state=target_group_state,
            global_step=global_step
        )

        loss = outputs.loss
        manifold_dist = outputs.manifold_distance

        # 总损失
        total_loss_value = loss + manifold_loss_weight * manifold_dist

        if torch.isfinite(total_loss_value):
            total_loss_value.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if torch.isfinite(grad_norm):
                optimizer.step()
            else:
                optimizer.zero_grad()
        else:
            optimizer.zero_grad()
            total_loss_value = loss  # 仍然记录原始 loss

        total_loss += loss.item()
        total_manifold += manifold_dist.item() if manifold_dist is not None else 0.0
        num_batches += 1
        global_step += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'manifold': f'{manifold_dist.item():.4f}' if manifold_dist is not None else '0.0'
        })

    stats = {
        'avg_loss': total_loss / max(num_batches, 1),
        'avg_manifold': total_manifold / max(num_batches, 1),
        'num_batches': num_batches,
        'global_step': global_step,
    }
    return stats


@torch.no_grad()
def evaluate(model, dataset, tokenizer, device, batch_size=16) -> float:
    """评估准确率"""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    correct = 0
    total = 0

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        answers = batch['answer']

        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
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


# ==================== 五阶段训练协议 ====================

def train_phase_1(
    model, train_datasets, optimizer, device, epochs: int, global_step: int
) -> dict:
    """阶段 1: 分域预训练 — 单群，纯算术数据"""
    print("\n" + "=" * 60)
    print("阶段 1: 分域预训练 — 群学习代数规则")
    print("=" * 60)

    # 仅使用算术数据
    arith_dataset = train_datasets['train_arithmetic']
    loader = get_domain_loader(arith_dataset, batch_size=16)

    stats = train_epoch(
        model, [loader], optimizer, 0, device,
        manifold_loss_weight=0.5,
        global_step=global_step,
        desc="Phase 1: Arithmetic pretraining"
    )
    return stats


def train_phase_2(
    model, train_datasets, optimizer, device, epochs: int, global_step: int
) -> dict:
    """阶段 2: 投影层预热 — 冻结生成元，训练投影层"""
    print("\n" + "=" * 60)
    print("阶段 2: 投影层预热 — 学习路由映射")
    print("=" * 60)

    # 冻结生成元
    mg_params = set()
    for n, p in model.named_parameters():
        if 'meta_group' in n and 'generator_params' in n:
            mg_params.add(id(p))
    freeze_params(mg_params, model)

    # 创建多域加载器
    loaders = [
        get_domain_loader(train_datasets['train_arithmetic'], batch_size=16),
        get_domain_loader(train_datasets['train_mixed'], batch_size=16),
    ]

    all_losses = []
    for epoch in range(1, epochs + 1):
        stats = train_epoch(
            model, loaders, optimizer, epoch, device,
            manifold_loss_weight=0.3,
            global_step=global_step,
            desc="Phase 2: Proj warmup"
        )
        all_losses.append(stats)
        global_step = stats['global_step']
        print(f"  Epoch {epoch}: loss={stats['avg_loss']:.4f}, manifold={stats['avg_manifold']:.4f}")

    # 解冻生成元
    unfreeze_params(mg_params, model)
    return all_losses[-1] if all_losses else {}


def train_phase_3(
    model, train_datasets, optimizer, device, epochs: int, global_step: int
) -> dict:
    """阶段 3: 交替冻结 — 避免耦合震荡"""
    print("\n" + "=" * 60)
    print("阶段 3: 交替冻结 — 解耦训练")
    print("=" * 60)

    mg_params = set()
    proj_params = set()
    for n, p in model.named_parameters():
        if 'meta_group' in n and 'generator_params' in n:
            mg_params.add(id(p))
        elif 'proj_to' in n or 'proj_from' in n or 'gate_network' in n or 'logit_alpha' in n:
            proj_params.add(id(p))

    loaders = [
        get_domain_loader(train_datasets['train_arithmetic'], batch_size=16),
        get_domain_loader(train_datasets['train_mixed'], batch_size=16),
    ]

    all_losses = []
    for epoch in range(1, epochs + 1):
        # 奇数 epoch: 冻结投影层，训练生成元
        # 偶数 epoch: 冻结生成元，训练投影层
        if epoch % 2 == 1:
            freeze_params(proj_params, model)
            unfreeze_params(mg_params, model)
            desc = "Phase 3: Freeze proj, train gen"
        else:
            freeze_params(mg_params, model)
            unfreeze_params(proj_params, model)
            desc = "Phase 3: Freeze gen, train proj"

        stats = train_epoch(
            model, loaders, optimizer, epoch, device,
            manifold_loss_weight=0.5,
            global_step=global_step,
            desc=desc
        )
        all_losses.append(stats)
        global_step = stats['global_step']
        print(f"  Epoch {epoch}: loss={stats['avg_loss']:.4f}, manifold={stats['avg_manifold']:.4f}")

    # 全部解冻
    unfreeze_params(mg_params, model)
    unfreeze_params(proj_params, model)
    return all_losses[-1] if all_losses else {}


def train_phase_4(
    model, train_datasets, optimizer, device, epochs: int, global_step: int
) -> dict:
    """阶段 4: 多域训练 — 加入语言+混合数据"""
    print("\n" + "=" * 60)
    print("阶段 4: 多域训练 — 全数据训练")
    print("=" * 60)

    loaders = [
        get_domain_loader(train_datasets['train_arithmetic'], batch_size=16),
        get_domain_loader(train_datasets['train_language'], batch_size=16),
        get_domain_loader(train_datasets['train_mixed'], batch_size=16),
    ]

    all_losses = []
    for epoch in range(1, epochs + 1):
        stats = train_epoch(
            model, loaders, optimizer, epoch, device,
            manifold_loss_weight=0.5,
            global_step=global_step,
            desc=f"Phase 4: Multi-domain epoch {epoch}"
        )
        all_losses.append(stats)
        global_step = stats['global_step']
        print(f"  Epoch {epoch}: loss={stats['avg_loss']:.4f}, "
              f"manifold={stats['avg_manifold']:.4f}")

    return all_losses[-1] if all_losses else {}


def train_phase_5(
    model, train_datasets, optimizer, device, epochs: int, global_step: int
) -> dict:
    """阶段 5: 髓鞘化 — 低学习率微调"""
    print("\n" + "=" * 60)
    print("阶段 5: 髓鞘化 — 微调稳定")
    print("=" * 60)

    # 降低所有学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1

    loaders = [
        get_domain_loader(train_datasets['train_arithmetic'], batch_size=16),
        get_domain_loader(train_datasets['train_mixed'], batch_size=16),
    ]

    all_losses = []
    for epoch in range(1, epochs + 1):
        stats = train_epoch(
            model, loaders, optimizer, epoch, device,
            manifold_loss_weight=0.5,
            global_step=global_step,
            desc="Phase 5: Myelination"
        )
        all_losses.append(stats)
        global_step = stats['global_step']
        print(f"  Epoch {epoch}: loss={stats['avg_loss']:.4f}, manifold={stats['avg_manifold']:.4f}")

    return all_losses[-1] if all_losses else {}


# ==================== 主函数 ====================

def run_multi_group_training(
    train_length: int = 10,
    epochs: int = 30,
    samples: int = 500,
    seed: int = 42,
    save_dir: str = './multi_group_results',
    quick_test: bool = False
):
    """运行完整的五阶段训练"""

    os.makedirs(save_dir, exist_ok=True)

    if quick_test:
        epochs = 1
        samples = 50
        print("=== 快速测试模式 ===")

    # 准备数据
    print("Preparing data...")
    datasets, tokenizer = prepare_data(train_length, samples, seed)

    # 创建模型
    print("\nCreating GPTWithGroup model (multi-group mode)...")
    model = GPTWithGroup(
        base_model_name='gpt2',
        group_d=16,
        num_generators=6,
        group_type='orthogonal',
        use_pretrained=True,
        enable_dynamic_expansion=True,
        expansion_threshold=0.5,
        max_generators_per_group=12
    ).to(device)

    print(f"总参数量：{model.get_num_params():,}")

    # 配置各阶段 epoch 数
    phase_epochs = {
        1: max(epochs // 5, 1),       # 分域预训练: 20%
        2: max(epochs // 5, 1),       # 投影预热: 20%
        3: max(epochs // 5, 1),       # 交替冻结: 20%
        4: max(epochs // 5, 1),       # 稀疏路由: 20%
        5: max(epochs // 5, 1),       # 髓鞘化: 20%
    }
    if quick_test:
        phase_epochs = {k: 1 for k in phase_epochs}

    total_phase_epochs = sum(phase_epochs.values())
    print(f"各阶段 epoch 数：{phase_epochs} (总计 {total_phase_epochs})")

    global_step = 0
    all_results = defaultdict(dict)

    # 阶段 1
    if not quick_test:
        phase1_stats = train_phase_1(
            model, datasets,
            torch.optim.AdamW(get_param_groups(model)), device,
            phase_epochs[1], global_step
        )
    else:
        phase1_stats = {'avg_loss': 0.0}
    global_step = phase1_stats.get('global_step', global_step)
    all_results['phase1'] = phase1_stats
    print(f"阶段 1 完成: loss={phase1_stats.get('avg_loss', 0):.4f}")

    # 阶段 2
    phase2_stats = train_phase_2(
        model, datasets,
        torch.optim.AdamW(get_param_groups(model)), device,
        phase_epochs[2], global_step
    )
    global_step = phase2_stats.get('global_step', global_step)
    all_results['phase2'] = phase2_stats
    print(f"阶段 2 完成: loss={phase2_stats.get('avg_loss', 0):.4f}")

    # 阶段 3
    phase3_stats = train_phase_3(
        model, datasets,
        torch.optim.AdamW(get_param_groups(model)), device,
        phase_epochs[3], global_step
    )
    global_step = phase3_stats.get('global_step', global_step)
    all_results['phase3'] = phase3_stats
    print(f"阶段 3 完成: loss={phase3_stats.get('avg_loss', 0):.4f}")

    # 阶段 4
    phase4_stats = train_phase_4(
        model, datasets,
        torch.optim.AdamW(get_param_groups(model)), device,
        phase_epochs[4], global_step
    )
    global_step = phase4_stats.get('global_step', global_step)
    all_results['phase4'] = phase4_stats
    print(f"阶段 4 完成: loss={phase4_stats.get('avg_loss', 0):.4f}")

    # 阶段 5
    phase5_stats = train_phase_5(
        model, datasets,
        torch.optim.AdamW(get_param_groups(model)), device,
        phase_epochs[5], global_step
    )
    all_results['phase5'] = phase5_stats
    print(f"阶段 5 完成: loss={phase5_stats.get('avg_loss', 0):.4f}")

    # 保存模型
    save_path = os.path.join(save_dir, 'model')
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n模型已保存至 {save_path}")

    # 评估
    print("\n=== 评估 ===")
    eval_results = {}
    for domain in ['arithmetic', 'language', 'mixed']:
        eval_dataset = datasets[f'eval_{domain}']
        acc = evaluate(model, eval_dataset, tokenizer, device, batch_size=16)
        eval_results[domain] = acc
        print(f"  {domain}: accuracy = {acc:.2%}")

    # 保存结果
    final_results = {
        'phase_results': {k: {kk: (vv.item() if hasattr(vv, 'item') else vv) for kk, vv in v.items()}
                          for k, v in all_results.items()},
        'eval_accuracy': eval_results,
        'config': {
            'train_length': train_length,
            'epochs': epochs,
            'samples': samples,
            'phase_epochs': phase_epochs,
            'group_d': 16,
            'num_generators': 6,
        }
    }

    results_path = os.path.join(save_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\n结果已保存至 {results_path}")

    return model, final_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Group GroupAlgebra Training')
    parser.add_argument('--train_length', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./multi_group_results')
    parser.add_argument('--quick-test', action='store_true', help='Quick test mode (1 epoch, 50 samples)')

    args = parser.parse_args()
    run_multi_group_training(
        train_length=args.train_length,
        epochs=args.epochs,
        samples=args.samples,
        seed=args.seed,
        save_dir=args.save_dir,
        quick_test=args.quick_test
    )
