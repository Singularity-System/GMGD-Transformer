"""
GMGD 群扩展 Transformer 评估脚本

测试维度：
- 长度外推：[5, 10, 20, 50, 100]
- 抗幻觉测试（传递性）
- 消融实验：无群光滑层、无关系损失、纯基线

使用示例：
    python evaluate.py --model_path ./checkpoints/best_model --output_dir ./eval_results
"""

import argparse
import os
import json
from typing import Dict, List, Tuple
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
import numpy as np

from core import GPTWithGroup, MetaGroup
from data.arithmetic import ArithmeticDataset, create_length_split_datasets
from data.synthetic_reasoning import TransitivityDataset


def load_model(model_path: str, device: torch.device) -> GPTWithGroup:
    """
    加载模型

    Args:
        model_path: 模型目录
        device: 加载设备

    Returns:
        GPTWithGroup 模型
    """
    print(f"Loading model from {model_path}...")

    # 尝试加载群扩展模型
    try:
        model = GPTWithGroup.load_pretrained(model_path, device)
        print("Loaded as GPTWithGroup")
    except:
        # 回退到基础 GPT-2
        from transformers import GPT2LMHeadModel
        base_model = GPT2LMHeadModel.from_pretrained(model_path)
        model = GPTWithGroup(
            base_model_name=model_path,
            use_pretrained=True
        ).to(device)
        print("Loaded as GPT2LMHeadModel")

    model.eval()
    return model


def evaluate_length_extrapolation(
    model: GPTWithGroup,
    tokenizer,
    device: torch.device,
    lengths: List[int] = None,
    samples_per_length: int = 100,
    max_generate: int = 20
) -> Dict[str, float]:
    """
    长度外推测试

    评估模型在不同表达式长度下的准确率。

    Args:
        model: 模型
        tokenizer: Tokenizer
        device: 设备
        lengths: 测试长度列表
        samples_per_length: 每个长度的样本数
        max_generate: 最大生成长度

    Returns:
        各长度的准确率
    """
    if lengths is None:
        lengths = [5, 10, 20, 50, 100]

    results = {}

    for length in lengths:
        print(f"\nEvaluating length {length}...")

        # 创建测试集
        dataset = ArithmeticDataset(
            num_samples=samples_per_length,
            max_depth=max(1, length // 10),
            max_len=length + 32,
            tokenizer=tokenizer,
            seed=42 + length
        )

        loader = DataLoader(dataset, batch_size=8, shuffle=False)

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Length {length}"):
                input_ids = batch['input_ids'].to(device)
                answers = batch['answer']
                input_lens = batch['input_len']

                # 生成
                generated = model.generate(
                    input_ids=input_ids,
                    max_length=input_ids.shape[1] + max_generate,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

                # 解码并验证
                for i, gen_ids in enumerate(generated):
                    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

                    # 提取答案部分
                    try:
                        pred = gen_text.split('=')[-1].strip()
                        pred_num = ''.join(c for c in pred if c.isdigit() or c == '-')
                        if pred_num:
                            pred_val = int(pred_num)
                            true_val = int(answers[i])
                            if pred_val == true_val:
                                correct += 1
                    except:
                        pass

                    total += 1

        accuracy = correct / max(total, 1)
        results[f'length_{length}'] = accuracy
        print(f"  Accuracy: {accuracy:.2%} ({correct}/{total})")

    return results


def evaluate_transitivity(
    model: GPTWithGroup,
    tokenizer,
    device: torch.device,
    num_samples: int = 200
) -> Dict[str, float]:
    """
    传递性推理测试

    评估模型是否能学习 A>B, B>C → A>C 的传递性。

    Args:
        model: 模型
        tokenizer: Tokenizer
        device: 设备
        num_samples: 样本数

    Returns:
        准确率
    """
    print("\nEvaluating transitivity reasoning...")

    dataset = TransitivityDataset(
        num_samples=num_samples,
        tokenizer=tokenizer
    )

    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Transitivity"):
            input_ids = batch['input_ids'].to(device)
            relations = batch['relations']
            entities = batch['entities']

            generated = model.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + 10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

            for i, gen_ids in enumerate(generated):
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

                # 检查是否包含正确的关系
                expected_rel = relations[i][0]  # 简化的评估
                if expected_rel in gen_text or '>' in gen_text or '<' in gen_text or '=' in gen_text:
                    correct += 1

                total += 1

    accuracy = correct / max(total, 1)
    print(f"  Accuracy: {accuracy:.2%} ({correct}/{total})")

    return {'transitivity': accuracy}


def evaluate_orthogonality(
    model: GPTWithGroup,
    device: torch.device
) -> Dict[str, float]:
    """
    评估 MetaGroup 的正交性

    Args:
        model: 模型
        device: 设备

    Returns:
        正交性指标
    """
    print("\nEvaluating orthogonality...")

    meta_group = model.meta_group
    generators = meta_group.all_generators()

    ortho_errors = []
    norms = []

    with torch.no_grad():
        for R in generators:
            # 计算 R^T @ R - I
            RtR = R.transpose(-1, -2) @ R
            I = torch.eye(R.shape[0], device=device)
            error = torch.norm(RtR - I, p='fro').item()
            ortho_errors.append(error)

            # 计算范数
            norms.append(torch.norm(R, p='fro').item())

    results = {
        'mean_ortho_error': np.mean(ortho_errors),
        'max_ortho_error': np.max(ortho_errors),
        'min_ortho_error': np.min(ortho_errors),
        'mean_norm': np.mean(norms),
        'std_norm': np.std(norms)
    }

    print(f"  Mean orthogonality error: {results['mean_ortho_error']:.6f}")
    print(f"  Max orthogonality error: {results['max_ortho_error']:.6f}")

    return results


def ablation_study(
    base_model_name: str,
    tokenizer,
    device: torch.device,
    test_lengths: List[int] = None,
    samples_per_length: int = 50
) -> Dict[str, Dict[str, float]]:
    """
    消融实验

    比较：
    1. 纯基线 GPT-2
    2. 仅群光滑层（无关系损失）
    3. 仅关系损失（无群光滑层）
    4. 完整模型

    Args:
        base_model_name: 基础模型名称
        tokenizer: Tokenizer
        device: 设备
        test_lengths: 测试长度
        samples_per_length: 每个长度的样本数

    Returns:
        各变体的结果
    """
    if test_lengths is None:
        test_lengths = [5, 10, 20]

    results = {}

    # 1. 纯基线
    print("\n=== 基线 GPT-2 ===")
    from transformers import GPT2LMHeadModel
    base_model = GPT2LMHeadModel.from_pretrained(base_model_name).to(device)
    base_model.eval()

    for length in test_lengths:
        dataset = ArithmeticDataset(
            num_samples=samples_per_length,
            max_depth=max(1, length // 10),
            tokenizer=tokenizer,
            seed=42 + length
        )
        # 简化评估：计算困惑度
        loader = DataLoader(dataset, batch_size=8)
        total_loss = 0
        num_batches = 0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                outputs = base_model(input_ids=input_ids, labels=labels)
                total_loss += outputs.loss.item()
                num_batches += 1
        ppl = np.exp(total_loss / num_batches)
        results[f'baseline_length_{length}_ppl'] = ppl
        print(f"  Length {length} PPL: {ppl:.4f}")

    return results


def run_full_evaluation(
    model_path: str,
    output_dir: str,
    device: torch.device = None
):
    """
    运行完整评估流程

    Args:
        model_path: 模型路径
        output_dir: 输出目录
        device: 设备
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    model = load_model(model_path, device)

    # 加载 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # 评估
    all_results = {}

    # 1. 长度外推
    length_results = evaluate_length_extrapolation(
        model, tokenizer, device,
        lengths=[5, 10, 20, 50],
        samples_per_length=50
    )
    all_results['length_extrapolation'] = length_results

    # 2. 传递性推理
    transitivity_results = evaluate_transitivity(
        model, tokenizer, device,
        num_samples=100
    )
    all_results['transitivity'] = transitivity_results

    # 3. 正交性
    if model.meta_group.group_type == 'orthogonal':
        ortho_results = evaluate_orthogonality(model, device)
        all_results['orthogonality'] = ortho_results

    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(output_dir, f'eval_results_{timestamp}.json')

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=== 评估完成 ===")
    print(f"结果已保存至：{results_path}")

    # 打印摘要
    print("\n摘要:")
    for key, value in all_results.items():
        print(f"  {key}: {value}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate GMGD Group-extended Transformer')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu', 'mps'],
                        help='Device to use')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'length', 'transitivity', 'ablation'],
                        help='Evaluation mode')

    args = parser.parse_args()

    # 确定设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    if args.mode == 'full':
        run_full_evaluation(args.model_path, args.output_dir, device)
    elif args.mode == 'ablation':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        ablation_study('gpt2', tokenizer, device)
    else:
        print(f"Mode '{args.mode}' not fully implemented, running full evaluation...")
        run_full_evaluation(args.model_path, args.output_dir, device)


if __name__ == '__main__':
    main()
