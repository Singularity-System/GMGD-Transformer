#!/usr/bin/env python3
"""
长度外推实验

训练：长度为 10 的算术表达式（加减法，不进位）
测试：不同长度（10, 15, 20, 30, 50, 80, 100）的表达式
模型：GPT-2 基线 vs GMGD-1（群扩展）

用法：
    python length_extrapolation.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
import random
import re

from core import GPTWithGroup, MetaGroup


# ==================== 数据生成 ====================

def generate_no_carry_expression(max_length: int = 10, seed: int = None) -> tuple:
    """
    生成不进位的加减法表达式

    约束：
    - 所有数字为 0-7（确保加减不进位/借位）
    - 左对齐：表达式靠左
    - 结果不产生负数

    返回：(表达式字符串，结果数字)
    """
    if seed is not None:
        random.seed(seed)

    # 生成数字列表（0-7 之间）
    numbers = [random.randint(0, 7) for _ in range(max_length // 2 + 1)]
    operators = [random.choice(['+', '-']) for _ in range(len(numbers) - 1)]

    # 构建表达式，确保结果非负
    result = numbers[0]
    expr_parts = [str(numbers[0])]

    for i, (op, num) in enumerate(zip(operators, numbers[1:])):
        if op == '+':
            result += num
        else:
            # 如果是减法，确保不会导致负数
            if result < num:
                # 调整：改为加法或调整数字
                op = '+'
                result += num
            else:
                result -= num
        expr_parts.append(op)
        expr_parts.append(str(num))

    expr = ' '.join(expr_parts)

    # 截断到最大长度
    tokens = expr.split()
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
        expr = ' '.join(tokens)
        # 重新计算结果（去掉的部分）
        # 简化处理：截断到完整的操作
        if tokens[-1] in ['+', '-']:
            tokens = tokens[:-1]
            expr = ' '.join(tokens)

    # 重新计算结果
    try:
        result = eval(expr.replace(' ', ''))
    except:
        result = 0

    return expr, result


class LengthExtrapolationDataset(torch.utils.data.Dataset):
    """
    长度外推数据集

    Args:
        train_length: 训练时表达式长度
        test_lengths: 测试时的长度列表
        samples_per_split: 每个 split 的样本数
        tokenizer: HuggingFace tokenizer
    """

    def __init__(
        self,
        train_length: int = 10,
        test_lengths: list = None,
        samples_per_split: int = 500,
        tokenizer=None,
        seed: int = 42
    ):
        self.train_length = train_length
        self.test_lengths = test_lengths or [10, 15, 20, 30, 50, 80, 100]
        self.tokenizer = tokenizer
        self.seed = seed

        random.seed(seed)

        # 生成训练集
        self.train_data = []
        for i in range(samples_per_split):
            expr, result = generate_no_carry_expression(train_length, seed + i)
            self.train_data.append((expr, result))

        # 生成测试集
        self.test_data = {}
        for length in self.test_lengths:
            self.test_data[length] = []
            for i in range(samples_per_split // 5):  # 每个测试长度少一些样本
                expr, result = generate_no_carry_expression(length, seed + i + 1000)
                self.test_data[length].append((expr, result))

    def get_train_loader(self, batch_size: int = 16) -> DataLoader:
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required")

        encoded_data = []
        for expr, result in self.train_data:
            input_text = f"Expr: {expr} ="
            target_text = f" {result}"
            full_text = input_text + target_text

            enc = self.tokenizer(
                full_text,
                truncation=True,
                return_tensors='pt'
            )

            input_ids = enc['input_ids'].squeeze(0)
            labels = input_ids.clone()

            # 只计算答案部分的 loss
            input_len = len(self.tokenizer.encode(input_text))
            labels[:input_len] = -100

            encoded_data.append({
                'input_ids': input_ids,
                'labels': labels,
                'expr': expr,
                'result': result
            })

        def collate_fn(batch):
            # 使用左 padding 动态批处理
            max_len = max(len(x['input_ids']) for x in batch)
            padded_input_ids = []
            padded_labels = []
            exprs = []
            results = []

            for item in batch:
                pad_len = max_len - len(item['input_ids'])
                # 左 padding
                input_ids = torch.cat([
                    torch.full((pad_len,), self.tokenizer.pad_token_id),
                    item['input_ids']
                ])
                labels = torch.cat([
                    torch.full((pad_len,), -100),
                    item['labels']
                ])
                padded_input_ids.append(input_ids)
                padded_labels.append(labels)
                exprs.append(item['expr'])
                results.append(item['result'])

            return {
                'input_ids': torch.stack(padded_input_ids),
                'labels': torch.stack(padded_labels),
                'expr': exprs,
                'result': results
            }

        return DataLoader(encoded_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    def get_test_loader(self, length: int, batch_size: int = 16) -> DataLoader:
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required")

        if length not in self.test_data:
            raise ValueError(f"No test data for length {length}")

        encoded_data = []
        for expr, result in self.test_data[length]:
            input_text = f"Expr: {expr} ="

            enc = self.tokenizer(
                input_text,
                truncation=True,
                return_tensors='pt'
            )

            input_ids = enc['input_ids'].squeeze(0)

            encoded_data.append({
                'input_ids': input_ids,
                'expr': expr,
                'result': result
            })

        def collate_fn(batch):
            # 使用左 padding 动态批处理
            max_len = max(len(x['input_ids']) for x in batch)
            padded_input_ids = []
            exprs = []
            results = []

            for item in batch:
                pad_len = max_len - len(item['input_ids'])
                # 左 padding
                input_ids = torch.cat([
                    torch.full((pad_len,), self.tokenizer.pad_token_id),
                    item['input_ids']
                ])
                padded_input_ids.append(input_ids)
                exprs.append(item['expr'])
                results.append(item['result'])

            return {
                'input_ids': torch.stack(padded_input_ids),
                'expr': exprs,
                'result': results
            }

        return DataLoader(encoded_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# ==================== 训练函数 ====================

def train_model(
    model,
    train_loader,
    epochs: int = 3,
    learning_rate: float = 1e-4,
    group_lr: float = None,  # 群参数学习率（默认与主网络相同）
    manifold_loss_weight: float = 0.1,  # 流形对齐损失权重
    device: torch.device = None,
    save_path: str = None
) -> list:
    """训练模型"""

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    # 参数分组：群参数用更高学习率
    if group_lr is not None and hasattr(model, 'meta_group'):
        group_params = list(model.meta_group.parameters())
        other_params = [p for n, p in model.named_parameters() if 'meta_group' not in n]
        param_groups = [
            {'params': group_params, 'lr': group_lr},
            {'params': other_params, 'lr': learning_rate}
        ]
        optimizer = torch.optim.AdamW(param_groups)
        print(f"优化器：群参数 lr={group_lr}, 其他 lr={learning_rate}, 流形损失权重={manifold_loss_weight}")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_manifold_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, labels=labels)

            if isinstance(outputs, dict):
                loss = outputs.get('loss', None)
                manifold_loss = outputs.get('manifold_distance', torch.tensor(0.0, device=device))
            else:
                loss = outputs.loss
                manifold_loss = outputs.manifold_distance if hasattr(outputs, 'manifold_distance') else torch.tensor(0.0, device=device)

            # 总损失 = 任务损失 + 流形对齐损失
            total_loss_value = loss + manifold_loss_weight * manifold_loss

            if loss is not None:
                optimizer.zero_grad()
                total_loss_value.backward()
                optimizer.step()

                total_loss += loss.item()
                total_manifold_loss += manifold_loss.item()
                num_batches += 1

                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'manifold': f"{manifold_loss.item():.6f}"
                })

        avg_loss = total_loss / max(num_batches, 1)
        avg_manifold = total_manifold_loss / max(num_batches, 1)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}: avg_loss = {avg_loss:.4f}, avg_manifold_distance = {avg_manifold:.6f}")

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(save_path)
        else:
            torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))
        print(f"Model saved to {save_path}")

    return train_losses


# ==================== 评估函数 ====================

def evaluate_length_extrapolation(
    model,
    test_loader,
    tokenizer,
    device: torch.device = None,
    max_generate: int = 20,
    debug: bool = False
) -> float:
    """评估特定长度下的准确率"""

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    correct = 0
    total = 0
    debug_count = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            results = batch['result']

            # 生成答案（使用 left padding）
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=20,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

            # 解析生成的答案
            for i, gen_ids in enumerate(generated):
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                true_result = results[i]
                expr_input = batch['expr'][i]

                # 提取答案
                pred_result = None
                try:
                    # 策略 1: 寻找 "= 数字" 模式
                    if '=' in gen_text:
                        # 提取等号后的内容
                        after_eq = gen_text.split('=')[-1].strip()
                        # 提取第一个有效数字（跳过空格和重复符号）
                        matches = re.findall(r'-?\d+', after_eq)
                        for match in matches:
                            # 跳过单独的 '-' 或空字符串
                            if match and match != '-':
                                pred_result = int(match)
                                break

                    # 策略 2: 如果没有等号，找最后一个数字
                    if pred_result is None:
                        # 从输入表达式后的内容找答案
                        if expr_input in gen_text:
                            after_expr = gen_text.split(expr_input)[-1]
                            # 提取所有数字
                            matches = re.findall(r'-?\d+', after_expr)
                            if matches:
                                # 取第一个非符号数字
                                for match in matches:
                                    if match.lstrip('-').isdigit():
                                        pred_result = int(match)
                                        break
                        else:
                            # 直接提取最后一个数字
                            matches = re.findall(r'\d+', gen_text)
                            if matches:
                                pred_result = int(matches[-1])
                except Exception as e:
                    pass

                # 调试输出：显示前 3 个样本
                if debug and debug_count < 3:
                    print(f"  [{debug_count}] Input: {batch['expr'][i]}")
                    print(f"      True: {true_result}, Pred: {pred_result}")
                    print(f"      Gen: {repr(gen_text[-50:])}")  # 显示最后 50 字符
                    debug_count += 1

                if pred_result is not None and pred_result == true_result:
                    correct += 1

                total += 1

    return correct / max(total, 1)


# ==================== 主实验 ====================

def run_extrapolation_experiment(
    train_length: int = 10,
    test_lengths: list = None,
    epochs: int = 3,
    batch_size: int = 16,
    samples_per_split: int = 500,
    save_dir: str = './length_extrapolation_results',
    skip_gpt2: bool = True  # 新增参数：跳过 GPT-2 基线
):
    """
    运行完整的长度外推实验

    比较：
    1. GPT-2 基线
    2. GMGD-1（群扩展）
    """

    if test_lengths is None:
        test_lengths = [10, 15, 20, 30, 50, 80, 100]

    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("长度外推实验")
    print("=" * 60)
    print(f"训练长度：{train_length}")
    print(f"测试长度：{test_lengths}")
    print(f"保存目录：{save_dir}")
    print()

    # 加载 tokenizer（使用本地缓存）
    print("Loading tokenizer...")
    # 在导入 transformers 之前设置环境变量
    import sys
    # 强制离线模式
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_EVALUATE_OFFLINE'] = '1'

    # 使用本地路径加载（避免任何网络请求）
    try:
        # 尝试从缓存加载
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    except Exception:
        # 如果缓存没有，尝试从默认缓存路径加载
        import subprocess
        cache_path = subprocess.check_output(
            ['python3', '-c', 'from transformers import TRANSFORMERS_CACHE; print(TRANSFORMERS_CACHE)'],
            text=True
        ).strip()
        print(f"Trying cache path: {cache_path}")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Decoder 模型需要左 padding

    # 创建数据集
    print("Creating datasets...")
    dataset = LengthExtrapolationDataset(
        train_length=train_length,
        test_lengths=test_lengths,
        samples_per_split=samples_per_split,
        tokenizer=tokenizer
    )
    train_loader = dataset.get_train_loader(batch_size=batch_size)

    # 实验结果存储
    results = {
        'GPT-2': {},
        'GMGD-1': {}
    }

    # ========== 实验 1: GPT-2 基线 ==========
    if not skip_gpt2:
        print("\n" + "=" * 60)
        print("实验 1: GPT-2 基线模型")
        print("=" * 60)

        gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        # 修改 forward 使其返回兼容格式
        gpt2_model.config.return_dict = True

        # 包装一下，使其有 forward 和 generate 方法
        class GPT2Wrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids, labels=None, **kwargs):
                outputs = self.model(input_ids=input_ids, labels=labels, **kwargs)
                return {'loss': outputs.loss, 'logits': outputs.logits}

            def generate(self, input_ids, max_new_tokens=20, do_sample=False, **kwargs):
                # 使用 max_new_tokens 而不是 max_length
                return self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    **kwargs
                )

        gpt2_wrapper = GPT2Wrapper(gpt2_model)

        # 训练
        gpt2_losses = train_model(
            gpt2_wrapper,
            train_loader,
            epochs=epochs,
            learning_rate=1e-4,
            save_path=os.path.join(save_dir, 'gpt2_baseline')
        )
        results['GPT-2']['train_losses'] = gpt2_losses

        # 评估各长度（首次运行时开启 debug 查看生成内容）
        debug_mode = True  # 第一次运行时开启调试
        for length in test_lengths:
            print(f"\n评估 GPT-2 在长度 {length}...")
            test_loader = dataset.get_test_loader(length, batch_size=batch_size)
            acc = evaluate_length_extrapolation(
                gpt2_wrapper, test_loader, tokenizer, debug=debug_mode
            )
            results['GPT-2'][length] = acc
            print(f"  长度 {length}: 准确率 = {acc:.2%}")
            debug_mode = False  # 只显示第一批调试信息

        # 清理内存
        del gpt2_wrapper
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        print("\n" + "=" * 60)
        print("跳过 GPT-2 基线实验（已使用先前结果）")
        print("=" * 60)
        # 从已有结果文件加载 GPT-2 结果
        results_path = os.path.join(save_dir, 'results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                existing_results = json.load(f)
            results['GPT-2'] = existing_results.get('GPT-2', {})
            print(f"已从 {results_path} 加载 GPT-2 结果")
        else:
            print("警告：未找到已有结果文件，GPT-2 结果将为空")

    # ========== 实验 2: GMGD-1 ==========
    print("\n" + "=" * 60)
    print("实验 2: GMGD-1（群扩展）模型")
    print("=" * 60)

    gmgd_model = GPTWithGroup(
        base_model_name='gpt2',
        group_d=16,  # 16x16 = 256, 需要调整 proj_to 输出
        num_generators=6,
        group_type='orthogonal',
        use_pretrained=True
    )
    # 重新创建 smooth_layers 以匹配正确的维度
    # group_d=16 => d*d=256，但 GPT-2 hidden_dim=768
    # 这里我们改为使用 group_d=28 (28*28=784 接近 768) 或者修改 smooth layer
    # 简单方案：使用 group_d 使得 d*d = hidden_dim
    # 但 768 不是完全平方数，所以我们需要修改 GroupSmoothLayer 支持非平方维度

    # 训练（群参数用更高学习率，流形损失强制表征对齐）
    gmgd_losses = train_model(
        gmgd_model,
        train_loader,
        epochs=epochs,
        learning_rate=1e-4,
        group_lr=5e-4,  # 群参数 5x 学习率
        manifold_loss_weight=0.1,  # 流形损失权重 0.1（温和约束，避免损害语言能力）
        save_path=os.path.join(save_dir, 'gmgd_model')
    )
    results['GMGD-1']['train_losses'] = gmgd_losses

    # 评估各长度
    debug_mode = True  # 第一次运行时开启调试
    for length in test_lengths:
        print(f"\n评估 GMGD-1 在长度 {length}...")
        test_loader = dataset.get_test_loader(length, batch_size=batch_size)
        acc = evaluate_length_extrapolation(
            gmgd_model, test_loader, tokenizer, debug=debug_mode
        )
        results['GMGD-1'][length] = acc
        print(f"  长度 {length}: 准确率 = {acc:.2%}")
        debug_mode = False  # 只显示第一批调试信息

    # ========== 保存结果 ==========
    results_path = os.path.join(save_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n结果已保存到：{results_path}")

    # ========== 绘图 ==========
    print("\n生成图表...")
    plot_extrapolation_results(results, test_lengths, save_dir)

    return results


def plot_extrapolation_results(results: dict, test_lengths: list, save_dir: str):
    """绘制长度外推曲线"""

    plt.figure(figsize=(12, 8))

    # 准备数据
    gpt2_accs = [results['GPT-2'].get(l, 0) for l in test_lengths]
    gmgd_accs = [results['GMGD-1'].get(l, 0) for l in test_lengths]

    # 绘制主图
    plt.subplot(2, 1, 1)
    plt.plot(test_lengths, gpt2_accs, 'o-', label='GPT-2 Baseline', linewidth=2, markersize=8)
    plt.plot(test_lengths, gmgd_accs, 's-', label='GMGD-1 (Ours)', linewidth=2, markersize=8)

    plt.xlabel('Expression Length (tokens)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Length Extrapolation: Arithmetic Expression Evaluation', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)

    # 绘制差异图
    plt.subplot(2, 1, 2)
    diff = np.array(gmgd_accs) - np.array(gpt2_accs)
    plt.bar(test_lengths, diff, color='steelblue', alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
    plt.xlabel('Expression Length (tokens)', fontsize=12)
    plt.ylabel('Accuracy Improvement (GMGD - GPT-2)', fontsize=12)
    plt.title('Performance Gain from Group Extension', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'length_extrapolation.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'length_extrapolation.pdf'), bbox_inches='tight')
    plt.close()

    print(f"图表已保存到：{save_dir}/length_extrapolation.png")

    # 打印关键数据
    print("\n" + "=" * 50)
    print("关键结果摘要")
    print("=" * 50)
    print(f"{'长度':<8} {'GPT-2':<12} {'GMGD-1':<12} {'提升':<10}")
    print("-" * 50)
    for length in test_lengths:
        gpt2 = results['GPT-2'].get(length, 0)
        gmgd = results['GMGD-1'].get(length, 0)
        gain = gmgd - gpt2
        print(f"{length:<8} {gpt2:<12.2%} {gmgd:<12.2%} {gain:+10.2%}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Length Extrapolation Experiment')
    parser.add_argument('--train_length', type=int, default=10,
                        help='Training expression length')
    parser.add_argument('--test_lengths', type=int, nargs='+',
                        default=[10, 15, 20, 30, 50, 80, 100],
                        help='Test expression lengths')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--samples', type=int, default=500,
                        help='Samples per split')
    parser.add_argument('--output_dir', type=str,
                        default='./length_extrapolation_results',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu', 'mps'],
                        help='Device to use')
    parser.add_argument('--skip_gpt2', action='store_true', default=True,
                        help='Skip GPT-2 baseline experiment (use existing results)')

    args = parser.parse_args()

    # 默认参数调整（更充分的训练）
    if args.samples == 500 and args.epochs == 3:
        print("提示：建议使用 --epochs 10 以获得更好的收敛效果")

    # 设置设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # 运行实验
    run_extrapolation_experiment(
        train_length=args.train_length,
        test_lengths=args.test_lengths,
        epochs=args.epochs,
        batch_size=args.batch_size,
        samples_per_split=args.samples,
        save_dir=args.output_dir,
        skip_gpt2=args.skip_gpt2
    )
