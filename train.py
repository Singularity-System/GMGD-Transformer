"""
GMGD 群扩展 Transformer 训练脚本

关键组件：
- 混合损失：total_loss = task_loss + lambda * rel_loss
- 优化器：AdamW，学习率 1e-4
- 群光滑层单独学习率：1e-5
- 监控指标：task_loss, rel_loss, orthogonality

使用示例：
    python train.py --epochs 3 --batch_size 16 --output_dir ./checkpoints
"""

import argparse
import os
import json
from typing import Dict, List, Optional
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import GPT2Tokenizer
from tqdm import tqdm

from gpt_with_group import GPTWithGroup
from meta_group import MetaGroup
from data.arithmetic import ArithmeticDataset, create_length_split_datasets


def get_param_groups(model: GPTWithGroup, args) -> List[Dict]:
    """
    构建参数分组（为主网络和群参数设置不同学习率）

    Args:
        model: GPTWithGroup 模型
        args: 命令行参数

    Returns:
        参数分组列表
    """
    # 主网络参数（GPT-2）
    main_params = []
    # 群相关参数
    group_params = []
    # 投影层参数
    proj_params = []

    for name, param in model.named_parameters():
        if 'meta_group' in name:
            group_params.append(param)
        elif 'smooth_layers' in name:
            proj_params.append(param)
        else:
            main_params.append(param)

    param_groups = [
        {'params': main_params, 'lr': args.learning_rate},
        {'params': group_params, 'lr': args.group_lr},
        {'params': proj_params, 'lr': args.proj_lr}
    ]

    return param_groups


def compute_relation_loss(meta_group: MetaGroup, args) -> torch.Tensor:
    """
    计算群关系损失

    定义生成元之间的交换关系（可根据任务调整）

    Args:
        meta_group: MetaGroup 模块
        args: 命令行参数

    Returns:
        关系损失
    """
    relations = []

    # 定义交换关系：生成元 i 和 i+1 应该交换
    # 这只是一个示例，实际应用中应根据任务设计
    for i in range(0, meta_group.num_generators - 1, 2):
        relations.append(((i, i + 1), (i + 1, i)))

    # 添加一些结合律关系
    if meta_group.num_generators >= 3:
        relations.append(((0, 1, 2), (0, 2, 1)))

    return meta_group.relation_loss(relations)


def train_epoch(
    model: GPTWithGroup,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    args,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """
    训练一个 epoch

    Returns:
        包含各种 loss 统计的字典
    """
    model.train()

    total_task_loss = 0.0
    total_rel_loss = 0.0
    total_ortho_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            labels=labels
        )

        task_loss = outputs.loss

        # 计算群关系损失
        rel_loss = compute_relation_loss(model.meta_group, args)

        # 计算正交性损失（仅对 orthogonal 群）
        if args.group_type == 'orthogonal':
            ortho_loss = model.meta_group.orthogonality_loss()
        else:
            ortho_loss = torch.tensor(0.0, device=device)

        # 总损失
        total_loss = (
            task_loss +
            args.rel_loss_weight * rel_loss +
            args.ortho_loss_weight * ortho_loss
        )

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()

        # 统计
        total_task_loss += task_loss.item()
        total_rel_loss += rel_loss.item()
        total_ortho_loss += ortho_loss.item()
        num_batches += 1

        # 更新进度条
        pbar.set_postfix({
            'task_loss': f"{task_loss.item():.4f}",
            'rel_loss': f"{rel_loss.item():.6f}",
            'ortho_loss': f"{ortho_loss.item():.6f}"
        })

    return {
        'task_loss': total_task_loss / num_batches,
        'rel_loss': total_rel_loss / num_batches,
        'ortho_loss': total_ortho_loss / num_batches
    }


def evaluate(
    model: GPTWithGroup,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    评估模型

    Returns:
        评估指标字典
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            # 支持 dict 或 object 返回
            if isinstance(outputs, dict):
                loss = outputs.get('loss', None)
            else:
                loss = outputs.loss
            if loss is not None:
                total_loss += loss.item()

            num_batches += 1

    return {
        'eval_loss': total_loss / max(num_batches, 1),
    }


def save_checkpoint(model: GPTWithGroup, optimizer: torch.optim.Optimizer, args, epoch: int, step: int):
    """保存检查点"""
    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 保存模型
    model.save_pretrained(checkpoint_dir)

    # 保存优化器状态
    torch.save({
        'epoch': epoch,
        'step': step,
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args)
    }, os.path.join(checkpoint_dir, 'training_state.pt'))

    # 保存配置
    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"Checkpoint saved to {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train GMGD Group-extended Transformer')

    # 模型配置
    parser.add_argument('--base_model', type=str, default='gpt2',
                        help='Base GPT-2 model name')
    parser.add_argument('--group_d', type=int, default=32,
                        help='Group representation dimension')
    parser.add_argument('--num_generators', type=int, default=12,
                        help='Number of generators')
    parser.add_argument('--group_type', type=str, default='orthogonal',
                        choices=['orthogonal', 'general_linear'],
                        help='Group type')

    # 训练配置
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for main network')
    parser.add_argument('--group_lr', type=float, default=1e-5,
                        help='Learning rate for meta_group')
    parser.add_argument('--proj_lr', type=float, default=1e-5,
                        help='Learning rate for projection layers')
    parser.add_argument('--rel_loss_weight', type=float, default=0.1,
                        help='Weight for relation loss')
    parser.add_argument('--ortho_loss_weight', type=float, default=0.01,
                        help='Weight for orthogonality loss')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')

    # 数据配置
    parser.add_argument('--train_samples', type=int, default=5000,
                        help='Number of training samples')
    parser.add_argument('--eval_samples', type=int, default=500,
                        help='Number of evaluation samples')
    parser.add_argument('--max_expr_depth', type=int, default=3,
                        help='Max expression depth for training')

    # 输出配置
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N batches')
    parser.add_argument('--save_interval', type=int, default=1,
                        help='Save checkpoint every N epochs')

    # 设备配置
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu', 'mps'],
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 确定设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存配置
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_path = os.path.join(args.output_dir, f'config_{timestamp}.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to {config_path}")

    # 加载 tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # 创建数据集
    print("Creating datasets...")
    train_dataset = ArithmeticDataset(
        num_samples=args.train_samples,
        max_depth=args.max_expr_depth,
        tokenizer=tokenizer
    )

    eval_dataset = ArithmeticDataset(
        num_samples=args.eval_samples,
        max_depth=args.max_expr_depth + 1,
        tokenizer=tokenizer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # 创建模型
    print("Creating model...")
    model = GPTWithGroup(
        base_model_name=args.base_model,
        group_d=args.group_d,
        num_generators=args.num_generators,
        group_type=args.group_type
    ).to(device)

    print(f"Model parameters: {model.get_num_params():,}")
    print(f"MetaGroup parameters: {sum(p.numel() for p in model.meta_group.parameters()):,}")

    # 构建优化器
    param_groups = get_param_groups(model, args)
    optimizer = AdamW(param_groups, weight_decay=0.01)

    # 训练循环
    print("\nStarting training...")
    best_eval_loss = float('inf')

    for epoch in range(args.epochs):
        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, args, device, epoch
        )

        print(f"\nEpoch {epoch + 1}/{args.epochs} - Training Metrics:")
        print(f"  Task Loss: {train_metrics['task_loss']:.4f}")
        print(f"  Relation Loss: {train_metrics['rel_loss']:.6f}")
        print(f"  Orthogonality Loss: {train_metrics['ortho_loss']:.6f}")

        # 评估
        eval_metrics = evaluate(model, eval_loader, device)
        print(f"  Eval Loss: {eval_metrics['eval_loss']:.4f}")

        # 保存最佳模型
        if eval_metrics['eval_loss'] < best_eval_loss:
            best_eval_loss = eval_metrics['eval_loss']
            best_model_dir = os.path.join(args.output_dir, 'best_model')
            os.makedirs(best_model_dir, exist_ok=True)
            model.save_pretrained(best_model_dir)
            print(f"  New best model saved to {best_model_dir}")

        # 定期保存检查点
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, args, epoch, len(train_loader))

    print("\nTraining completed!")
    print(f"Best eval loss: {best_eval_loss:.4f}")


if __name__ == '__main__':
    main()
