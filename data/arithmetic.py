"""
算术表达式数据集生成器

生成用于训练和测试组合泛化能力的算术表达式求值数据。
关键设计：
- 训练数据：短序列（长度 ≤5）
- 测试数据：长序列（长度 20–100）验证组合泛化

支持操作：+, -, *
支持数字：0-9 的整数
"""

import random
import re
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader


def generate_expression(
    max_depth: int = 3,
    operators: List[str] = None,
    max_number: int = 20
) -> str:
    """
    递归生成随机算术表达式

    Args:
        max_depth: 最大嵌套深度
        operators: 支持的操作符列表
        max_number: 最大数字

    Returns:
        表达式字符串
    """
    if operators is None:
        operators = ['+', '-', '*']

    if max_depth == 0:
        # 基础情况：返回随机数
        return str(random.randint(0, max_number))

    # 递归情况：生成二元表达式
    op = random.choice(operators)
    left = generate_expression(max_depth - 1, operators, max_number)
    right = generate_expression(max_depth - 1, operators, max_number)

    # 避免除以零（如果支持除法）
    if op == '/' and right == '0':
        right = str(random.randint(1, max_number))

    return f"({left}{op}{right})"


def evaluate_expression(expr: str) -> int:
    """
    安全地计算算术表达式

    Args:
        expr: 表达式字符串

    Returns:
        计算结果
    """
    # 只允许安全字符
    if not re.match(r'^[\d\+\-\*\(\)\s]+$', expr):
        raise ValueError(f"Invalid expression: {expr}")

    # 使用 eval 计算（已过滤危险字符）
    return int(eval(expr))


def expression_to_tokens(
    expr: str,
    tokenizer,
    max_len: int = 128
) -> Dict[str, torch.Tensor]:
    """
    将表达式转换为 token

    Args:
        expr: 表达式字符串
        tokenizer: HuggingFace tokenizer
        max_len: 最大长度

    Returns:
        包含 input_ids 和 labels 的字典
    """
    # 格式化输入： "Expression: (1+2)*3 = "
    input_text = f"Expression: {expr} = "

    # Tokenize
    encoded = tokenizer(
        input_text,
        return_tensors='pt',
        max_length=max_len,
        padding='max_length',
        truncation=True
    )

    input_ids = encoded['input_ids'].squeeze(0)

    # 创建 labels（用于语言模型训练）
    labels = input_ids.clone()
    # 将 input 部分的 label 设为 -100（忽略）
    input_len = len(tokenizer.encode(input_text))
    labels[:input_len] = -100

    return {
        'input_ids': input_ids,
        'labels': labels,
        'expression': expr,
        'answer': str(evaluate_expression(expr))
    }


class ArithmeticDataset(Dataset):
    """
    算术表达式求值数据集

    参数：
        num_samples: 样本数量
        max_depth: 表达式最大深度（控制难度）
        max_len: 最大序列长度
        operators: 支持的操作符
        tokenizer: HuggingFace tokenizer
        seed: 随机种子

    示例：
        >>> from transformers import GPT2Tokenizer
        >>> tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        >>> dataset = ArithmeticDataset(
        ...     num_samples=1000,
        ...     max_depth=3,
        ...     tokenizer=tokenizer
        ... )
        >>> sample = dataset[0]
        >>> print(f"{sample['expression']} = {sample['answer']}")
    """

    def __init__(
        self,
        num_samples: int = 1000,
        max_depth: int = 3,
        max_len: int = 128,
        operators: List[str] = None,
        tokenizer=None,
        seed: int = 42
    ):
        super().__init__()

        self.num_samples = num_samples
        self.max_depth = max_depth
        self.max_len = max_len
        self.operators = operators or ['+', '-', '*']
        self.tokenizer = tokenizer
        self.seed = seed

        random.seed(seed)

        # 生成所有表达式
        self.expressions = []
        self.answers = []

        for _ in range(num_samples):
            expr = generate_expression(
                max_depth=max_depth,
                operators=self.operators
            )
            self.expressions.append(expr)
            self.answers.append(evaluate_expression(expr))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        expr = self.expressions[idx]
        answer = self.answers[idx]

        if self.tokenizer is None:
            # 返回原始数据
            return {
                'expression': expr,
                'answer': str(answer),
                'idx': idx
            }

        # 转换为 token
        input_text = f"Expression: {expr} = "
        target_text = str(answer)

        # 拼接完整序列
        full_text = input_text + target_text

        encoded = self.tokenizer(
            full_text,
            return_tensors='pt',
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )

        input_ids = encoded['input_ids'].squeeze(0)
        labels = input_ids.clone()

        # 只计算答案部分的损失
        input_len = len(self.tokenizer.encode(input_text))
        labels[:input_len] = -100

        return {
            'input_ids': input_ids,
            'labels': labels,
            'expression': expr,
            'answer': str(answer),
            'input_len': input_len
        }

    def get_accuracy(self, predictions: List[str], ground_truth: List[str]) -> float:
        """
        计算准确率

        Args:
            predictions: 预测答案列表
            ground_truth: 真实答案列表

        Returns:
            准确率
        """
        correct = 0
        for pred, truth in zip(predictions, ground_truth):
            # 提取数字（去除空格和特殊字符）
            pred_num = re.findall(r'-?\d+', pred.strip())
            if pred_num and int(pred_num[0]) == int(truth):
                correct += 1
        return correct / len(ground_truth)


def create_length_split_datasets(
    tokenizer,
    train_max_len: int = 5,
    test_lengths: List[int] = None,
    samples_per_split: int = 500,
    seed: int = 42
) -> Dict[str, ArithmeticDataset]:
    """
    创建按长度分割的训练/测试数据集

    用于验证组合泛化能力：训练短序列，测试长序列。

    Args:
        tokenizer: HuggingFace tokenizer
        train_max_len: 训练集最大表达式长度
        test_lengths: 测试集长度列表
        samples_per_split: 每个 split 的样本数
        seed: 随机种子

    Returns:
        数据集字典，包含 'train' 和 'test_{length}'
    """
    if test_lengths is None:
        test_lengths = [5, 10, 20, 50, 100]

    datasets = {}

    # 训练集：短表达式（深度 1-2）
    datasets['train'] = ArithmeticDataset(
        num_samples=samples_per_split,
        max_depth=2,
        tokenizer=tokenizer,
        seed=seed
    )

    # 测试集：不同长度的表达式
    for length in test_lengths:
        # 根据目标长度估算深度
        # 深度与长度大致成正比
        depth = max(1, length // 10)
        datasets[f'test_{length}'] = ArithmeticDataset(
            num_samples=samples_per_split,
            max_depth=depth,
            max_len=length + 32,  # 留有余量
            tokenizer=tokenizer,
            seed=seed + length  # 不同长度不同种子
        )

    return datasets


if __name__ == '__main__':
    # 测试数据生成
    from transformers import GPT2Tokenizer

    print("测试算术表达式数据集生成...")

    # 加载 tokenizer
    try:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    except:
        tokenizer = None
        print("无法加载 GPT2Tokenizer，使用原始格式")

    # 创建数据集
    if tokenizer:
        datasets = create_length_split_datasets(
            tokenizer,
            train_max_len=5,
            test_lengths=[5, 10, 20],
            samples_per_split=10
        )

        print(f"\n训练集大小：{len(datasets['train'])}")
        for name, dataset in datasets.items():
            if name != 'train':
                print(f"{name} 大小：{len(dataset)}")

        # 打印样本
        print("\n=== 训练集样本 ===")
        for i in range(3):
            sample = datasets['train'][i]
            print(f"{sample['expression']} = {sample['answer']}")
    else:
        # 无 tokenizer 测试
        dataset = ArithmeticDataset(num_samples=5, max_depth=2)
        print("\n=== 原始格式样本 ===")
        for i in range(5):
            sample = dataset[i]
            print(f"{sample['expression']} = {sample['answer']}")
