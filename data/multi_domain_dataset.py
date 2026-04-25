"""
多域数据集：算术 + 语言 + 混合

用于多群 GroupAlgebra 训练，提供三种域的数据：
- arithmetic: 不进位加减法表达式（需要代数规则）
- language: 自然语言文本（需要统计知识）
- mixed: 算术表达式 + 语言干扰（需要两种能力）

每批数据带有 domain label，用于监督路由训练。
"""

import random
import re
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader

from data.arithmetic import generate_expression, evaluate_expression


# === 语言干扰模板 ===

LANGUAGE_TEMPLATES = [
    "The total number of points that a player can earn in a game of {sport} is approximately {number}.",
    "There are about {number} different species of {animal} in the world.",
    "The capital city of {country} has a population of around {number} million people.",
    "A typical {animal} can live for about {number} years in the wild.",
    "The distance from Earth to the {celestial_body} is roughly {number} kilometers.",
    "In {country}, it takes about {number} hours to drive across the country.",
    "The average temperature in {city} during summer is around {number} degrees Celsius.",
    "There are approximately {number} stars visible to the naked eye from Earth.",
    "A human heart beats about {number} times per minute during exercise.",
    "The library has a collection of over {number} books on various subjects.",
]

SPORTS = ['basketball', 'football', 'tennis', 'baseball', 'hockey', 'soccer']
ANIMALS = ['bird', 'fish', 'mammal', 'reptile', 'insect', 'amphibian']
COUNTRIES = ['France', 'Japan', 'Brazil', 'Canada', 'Australia', 'Germany', 'Italy', 'China']
CITIES = ['Tokyo', 'Paris', 'London', 'New York', 'Sydney', 'Berlin', 'Rome']
CELESTIAL_BODIES = ['Moon', 'Sun', 'Mars', 'Venus', 'Jupiter']


def _fill_template(template: str, seed: int = None) -> str:
    """用随机内容填充语言模板"""
    if seed is not None:
        random.seed(seed)
    text = template
    text = text.replace('{sport}', random.choice(SPORTS))
    text = text.replace('{animal}', random.choice(ANIMALS))
    text = text.replace('{country}', random.choice(COUNTRIES))
    text = text.replace('{city}', random.choice(CITIES))
    text = text.replace('{celestial_body}', random.choice(CELESTIAL_BODIES))
    text = text.replace('{number}', str(random.randint(1, 99)))
    return text


def generate_arithmetic_sample(max_length: int = 10, seed: int = None) -> tuple:
    """生成算术域样本"""
    if seed is not None:
        random.seed(seed)
    numbers = [random.randint(0, 7) for _ in range(max_length // 2 + 1)]
    operators = [random.choice(['+', '-']) for _ in range(len(numbers) - 1)]
    result = numbers[0]
    expr_parts = [str(numbers[0])]
    for i, (op, num) in enumerate(zip(operators, numbers[1:])):
        if op == '-' and result < num:
            op = '+'
            result += num
        else:
            result += num if op == '+' else -num
        expr_parts.append(op)
        expr_parts.append(str(num))
    expr = ' '.join(expr_parts[:max_length])
    tokens = expr.split()
    if tokens[-1] in ['+', '-']:
        tokens = tokens[:-1]
        expr = ' '.join(tokens)
    try:
        result = eval(expr.replace(' ', ''))
    except:
        result = 0
    return f"Expr: {expr} =", result, expr


def generate_language_sample(seed: int = None) -> tuple:
    """生成语言域样本（分类任务：从文本中提取数字）"""
    if seed is not None:
        random.seed(seed)
    template = random.choice(LANGUAGE_TEMPLATES)
    # 生成具体文本并记录答案
    text = template
    number = random.randint(1, 99)
    text = text.replace('{number}', str(number))
    text = text.replace('{sport}', random.choice(SPORTS))
    text = text.replace('{animal}', random.choice(ANIMALS))
    text = text.replace('{country}', random.choice(COUNTRIES))
    text = text.replace('{city}', random.choice(CITIES))
    text = text.replace('{celestial_body}', random.choice(CELESTIAL_BODIES))
    return f"Extract: {text}", number, text


def generate_mixed_sample(max_length: int = 10, seed: int = None) -> tuple:
    """生成混合域样本（算术 + 语言干扰）"""
    if seed is not None:
        random.seed(seed)
    # 生成算术表达式
    numbers = [random.randint(0, 7) for _ in range(max_length // 2 + 1)]
    operators = [random.choice(['+', '-']) for _ in range(len(numbers) - 1)]
    result = numbers[0]
    expr_parts = [str(numbers[0])]
    for i, (op, num) in enumerate(zip(operators, numbers[1:])):
        if op == '-' and result < num:
            op = '+'
            result += num
        else:
            result += num if op == '+' else -num
        expr_parts.append(op)
        expr_parts.append(str(num))
    expr = ' '.join(expr_parts[:max_length])
    tokens = expr.split()
    if tokens[-1] in ['+', '-']:
        tokens = tokens[:-1]
        expr = ' '.join(tokens)
    try:
        result = eval(expr.replace(' ', ''))
    except:
        result = 0
    # 添加语言干扰
    lang_part = _fill_template(random.choice(LANGUAGE_TEMPLATES), seed + 100)
    input_text = f"{lang_part} Compute: {expr} ="
    return input_text, result, expr


class MultiDomainDataset(Dataset):
    """
    多域数据集

    Args:
        domain: 域类型 ('arithmetic', 'language', 'mixed', 'all')
        num_samples: 样本数量
        max_length: 算术表达式最大长度
        tokenizer: HuggingFace tokenizer
        seed: 随机种子
        max_len: 最大 token 长度
    """

    def __init__(
        self,
        domain: str = 'all',
        num_samples: int = 500,
        max_length: int = 10,
        tokenizer=None,
        seed: int = 42,
        max_len: int = 128
    ):
        super().__init__()

        self.domain = domain
        self.num_samples = num_samples
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.seed = seed
        self.max_len = max_len
        self.samples = []

        random.seed(seed)

        if domain == 'arithmetic':
            self._generate_arithmetic(num_samples, seed)
        elif domain == 'language':
            self._generate_language(num_samples, seed)
        elif domain == 'mixed':
            self._generate_mixed(num_samples, seed)
        elif domain == 'all':
            # 三等分
            per_domain = num_samples // 3
            self._generate_arithmetic(per_domain, seed)
            self._generate_language(per_domain, seed + 1000)
            self._generate_mixed(num_samples - 2 * per_domain, seed + 2000)

    def _generate_arithmetic(self, n: int, seed: int):
        for i in range(n):
            input_text, answer, expr = generate_arithmetic_sample(self.max_length, seed + i)
            self.samples.append({
                'input_text': input_text,
                'answer': answer,
                'expr': expr,
                'domain': 'arithmetic'
            })

    def _generate_language(self, n: int, seed: int):
        for i in range(n):
            input_text, answer, text = generate_language_sample(seed + i)
            self.samples.append({
                'input_text': input_text,
                'answer': answer,
                'expr': text,
                'domain': 'language'
            })

    def _generate_mixed(self, n: int, seed: int):
        for i in range(n):
            input_text, answer, expr = generate_mixed_sample(self.max_length, seed + i)
            self.samples.append({
                'input_text': input_text,
                'answer': answer,
                'expr': expr,
                'domain': 'mixed'
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.tokenizer is None:
            return sample

        input_text = sample['input_text']
        target_text = f" {sample['answer']}"

        # Tokenize full sequence
        full_text = input_text + target_text
        enc = self.tokenizer(
            full_text,
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = enc['input_ids'].squeeze(0)
        labels = input_ids.clone()

        # 只计算答案部分的损失
        input_enc = self.tokenizer(
            input_text,
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            return_tensors='pt'
        )
        input_len = (input_enc['input_ids'] != self.tokenizer.pad_token_id).sum().item()
        labels[:input_len] = -100

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': (input_ids != self.tokenizer.pad_token_id).long(),
            'answer': sample['answer'],
            'expr': sample['expr'],
            'domain': sample['domain']
        }


def create_multi_domain_datasets(
    tokenizer,
    train_samples: int = 500,
    eval_samples: int = 100,
    train_length: int = 10,
    seed: int = 42
) -> Dict[str, MultiDomainDataset]:
    """
    创建多域训练/评估数据集

    Returns:
        {'train_arithmetic': ..., 'train_language': ..., 'train_mixed': ...,
         'eval_arithmetic': ..., 'eval_language': ..., 'eval_mixed': ...}
    """
    datasets = {}

    for split, samples, length in [('train', train_samples, train_length),
                                     ('eval', eval_samples, train_length)]:
        for domain in ['arithmetic', 'language', 'mixed']:
            key = f'{split}_{domain}'
            datasets[key] = MultiDomainDataset(
                domain=domain,
                num_samples=samples,
                max_length=length,
                tokenizer=tokenizer,
                seed=seed + hash(key) % 10000
            )

    return datasets


def collate_fn(batch):
    """DataLoader collate function"""
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'answer': [b['answer'] for b in batch],
        'expr': [b['expr'] for b in batch],
        'domain': [b['domain'] for b in batch]
    }


def get_domain_loader(
    dataset: MultiDomainDataset,
    batch_size: int = 16,
    shuffle: bool = True
) -> DataLoader:
    """创建指定域的数据加载器"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
