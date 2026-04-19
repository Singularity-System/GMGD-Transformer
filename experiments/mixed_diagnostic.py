#!/usr/bin/env python3
"""
混合诊断数据集：算术 + 语言干扰

目的：区分"群模块合理辅助"与"Transformer 被短路"

任务：给定包含算术表达式和自然语言干扰的字符串，输出算术结果。
- 算术部分：需要代数规则（群模块擅长）
- 语言干扰：需要统计知识（Transformer 擅长）

诊断逻辑：
- 若 Transformer 被短路 → 语言干扰部分性能下降
- 若自适应机制正常 → 两部分性能均不低于各自擅长者
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 设置离线模式，避免连接 HuggingFace
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import random
import re
import torch

from data.arithmetic import generate_expression, evaluate_expression


# ==================== 模板定义 ====================

# 训练集模板（前 3 个）
TRAIN_TEMPLATES = [
    "计算：{}，注意不要算错",
    "{}的结果是",
    "请帮我算一下{}等于多少",
]

# 测试集模板（后 2 个，未见过的）
TEST_TEMPLATES = [
    "表达式：{}，求值",
    "{} = ?",
    "求解：{}，给出答案",
    "Evaluate: {}",  # 英文干扰
    "Compute the value of: {}",
]

# 所有模板（用于基线测试）
ALL_TEMPLATES = TRAIN_TEMPLATES + TEST_TEMPLATES


# ==================== 数据生成 ====================

def generate_mixed_sample(expr_depth: int, template: str, seed: int = None) -> tuple:
    """
    生成混合样本（算术 + 语言干扰）

    Args:
        expr_depth: 表达式深度（控制长度）
        template: 语言模板
        seed: 随机种子

    Returns:
        (输入文本，数值答案，表达式字符串)
    """
    if seed is not None:
        random.seed(seed)

    expr = generate_expression(max_depth=expr_depth, operators=['+', '-'], max_number=7)
    # 移除括号，简化表达式
    expr = expr.replace('(', '').replace(')', '')
    try:
        value = evaluate_expression(expr.replace(' ', ''))
    except:
        value = 0
    input_text = template.format(expr)

    return input_text, value, expr


class MixedDiagnosticDataset(torch.utils.data.Dataset):
    """
    混合诊断数据集
    """
    def __init__(
        self,
        num_samples: int,
        expr_depth: int,
        tokenizer,
        templates: list = None,
        seed: int = 42,
        max_len: int = 128
    ):
        random.seed(seed)
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        if templates is None:
            templates = ALL_TEMPLATES

        for i in range(num_samples):
            input_text, value, expr = generate_mixed_sample(
                expr_depth,
                random.choice(templates),
                seed=seed + i
            )

            # Tokenize
            enc = tokenizer(
                input_text,
                padding='max_length',
                max_length=max_len,
                truncation=True,
                return_tensors='pt'
            )

            self.samples.append({
                'input_ids': enc['input_ids'][0],
                'attention_mask': enc['attention_mask'][0],
                'answer': value,
                'expression': expr,
                'input_text': input_text
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    answers = torch.tensor([item['answer'] for item in batch])
    expressions = [item['expression'] for item in batch]
    input_texts = [item['input_text'] for item in batch]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'answer': answers,
        'expression': expressions,
        'input_text': input_texts
    }


# ==================== 门控诊断分析 ====================

def analyze_gate_behavior(model, batch, tokenizer, device):
    """
    分析门控网络行为：检查哪些 token 上门控值较高

    返回：
    - 算术 token 上的平均门控值
    - 语言干扰 token 上的平均门控值
    """
    model.eval()

    # 需要访问模型内部的门控输出
    # 这里假设 GroupSmoothLayer 有记录门控值的能力
    gate_values = {}

    with torch.no_grad():
        # 前向传播，收集门控值
        pass

    return gate_values


# ==================== 评估函数 ====================

def evaluate_mixed_diagnostic(
    model,
    tokenizer,
    device,
    expr_depths: list = [2, 4, 6],
    num_samples_per_length: int = 50,
    train_templates: list = None,
    test_templates: list = None
):
    """
    运行混合诊断评估

    返回：
    - 算术准确率（按深度）
    - 模板泛化准确率（训练 vs 测试模板）
    """
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    model.eval()

    results = {
        'accuracy_by_depth': {},
        'accuracy_by_template': {
            'train_templates': [],
            'test_templates': []
        },
        'gate_analysis': {}
    }

    for depth in expr_depths:
        print(f"\n评估深度 {depth}...")

        # 使用训练模板测试
        train_dataset = MixedDiagnosticDataset(
            num_samples_per_length,
            depth,
            tokenizer,
            templates=train_templates or TRAIN_TEMPLATES,
            seed=42 + depth,
            max_len=128
        )
        train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn)

        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f"Depth {depth} (train tmpl)", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            answers = batch['answer']

            # 生成预测
            generated = model.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + 8,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=attention_mask
            )

            # 解析答案
            for gen_ids, true_val in zip(generated, answers):
                pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                nums = re.findall(r'-?\d+', pred_text)
                if nums:
                    pred_val = int(nums[-1])
                    if pred_val == true_val:
                        train_correct += 1
                train_total += 1

        results['accuracy_by_depth'][f'depth_{depth}_train_tmpl'] = train_correct / train_total
        results['accuracy_by_template']['train_templates'].append(train_correct / train_total)
        print(f"  训练模板准确率：{train_correct / train_total:.2%}")

        # 使用测试模板测试（未见过的）
        test_dataset = MixedDiagnosticDataset(
            num_samples_per_length,
            depth,
            tokenizer,
            templates=test_templates or TEST_TEMPLATES,
            seed=42 + depth + 1000,
            max_len=128
        )
        test_loader = DataLoader(test_dataset, batch_size=8, collate_fn=collate_fn)

        test_correct = 0
        test_total = 0

        for batch in tqdm(test_loader, desc=f"Depth {depth} (test tmpl)", leave=False):
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
                        test_correct += 1
                test_total += 1

        results['accuracy_by_depth'][f'depth_{depth}_test_tmpl'] = test_correct / test_total
        results['accuracy_by_template']['test_templates'].append(test_correct / test_total)
        print(f"  测试模板准确率：{test_correct / test_total:.2%}")

    # 汇总
    results['summary'] = {
        'mean_train_template_acc': sum(results['accuracy_by_template']['train_templates']) / len(results['accuracy_by_template']['train_templates']),
        'mean_test_template_acc': sum(results['accuracy_by_template']['test_templates']) / len(results['accuracy_by_template']['test_templates']),
    }

    return results


# ==================== 主函数 ====================

if __name__ == '__main__':
    import json
    import torch
    from transformers import GPT2Tokenizer
    from core import GPTWithGroup

    print("=" * 60)
    print("混合诊断实验：算术 + 语言干扰")
    print("=" * 60)

    device = torch.device('cpu')

    # 加载模型
    model_path = './checkpoints/gmgd_20epoch/checkpoint_epoch_3'
    print(f"\nLoading model from {model_path}...")

    from core.gpt_with_group import GPTWithGroup
    from transformers import GPT2LMHeadModel
    from safetensors.torch import load_file
    import json

    # 读取配置
    with open(os.path.join(model_path, 'config.json')) as f:
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

    print("Loading base_model weights...")
    # 从 safetensors 加载 base_model
    state_dict = load_file(os.path.join(model_path, 'model.safetensors'))
    # 移除 transformer 前缀
    new_state = {k.replace('transformer.', ''): v for k, v in state_dict.items()}
    model.base_model.load_state_dict(new_state, strict=False)
    print("Base model loaded.")

    print("Loading group state...")
    group_state = torch.load(os.path.join(model_path, 'group_state.pt'), map_location=device, weights_only=False)
    model.meta_group.load_state_dict(group_state['meta_group'])
    for i, smooth_layer in enumerate(model.smooth_layers):
        layer_state = group_state.get(f'smooth_layer_{i}', {})
        if layer_state:
            smooth_layer.load_state_dict(layer_state)
    print("Group state loaded.")

    model.to(device)
    model.eval()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # 运行诊断评估
    results = evaluate_mixed_diagnostic(
        model,
        tokenizer,
        device,
        expr_depths=[2, 4, 6],
        num_samples_per_length=30,
        train_templates=TRAIN_TEMPLATES,
        test_templates=TEST_TEMPLATES
    )

    # 打印结果
    print("\n" + "=" * 60)
    print("诊断结果")
    print("=" * 60)

    print("\n按长度和模板类型的准确率:")
    for key, value in results['accuracy_by_length'].items():
        print(f"  {key}: {value:.2%}")

    print(f"\n平均训练模板准确率：{results['summary']['mean_train_template_acc']:.2%}")
    print(f"平均测试模板准确率：{results['summary']['mean_test_template_acc']:.2%}")

    # 诊断结论
    print("\n" + "=" * 60)
    print("诊断结论")
    print("=" * 60)

    train_acc = results['summary']['mean_train_template_acc']
    test_acc = results['summary']['mean_test_template_acc']

    # GPT-2 基线（估计）
    gpt2_train_baseline = 0.85  # 短序列
    gpt2_test_baseline = 0.70   # 未见模板会下降

    if test_acc >= gpt2_test_baseline:
        print("✓ 自适应机制正常工作")
        print("  - 群模块处理算术，Transformer 处理语言干扰")
        print("  - 未见模板上性能未下降，说明语言理解未被短路")
    elif test_acc < gpt2_test_baseline * 0.8:
        print("✗ Transformer 可能被短路")
        print("  - 语言干扰部分性能显著低于基线")
        print("  - 需检查门控网络是否在语言 token 上异常激活")
    else:
        print("? 需要进一步分析")
        print(f"  - 测试模板准确率 {test_acc:.2%} 介于基线之间")

    # 保存结果
    output_path = './experiments/mixed_diagnostic_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存至：{output_path}")
