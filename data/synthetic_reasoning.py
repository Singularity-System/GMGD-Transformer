"""
合成推理数据集生成器

生成用于测试逻辑推理能力的数据集：
- 传递性推理：A>B, B>C → A>C
- 等价推理：A=B, B=C → A=C
- 否定推理：A≠B, B=C → A≠C
"""

import random
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset


class TransitivityDataset(Dataset):
    """
    传递性推理数据集

    生成形如 "A > B, B > C, 因此 A ? C" 的推理题
    用于测试模型是否能学习群的传递性公理

    参数：
        num_samples: 样本数量
        num_entities: 实体数量（字母 A-Z）
        relation_types: 关系类型 ['>', '<', '=', '!=']
        tokenizer: HuggingFace tokenizer
        seed: 随机种子

    示例：
        >>> dataset = TransitivityDataset(num_samples=100)
        >>> sample = dataset[0]
        >>> print(sample['question'])  # "If A > B and B > C, what is the relation between A and C?"
        >>> print(sample['answer'])    # "A > C"
    """

    def __init__(
        self,
        num_samples: int = 1000,
        num_entities: int = 6,
        relation_types: List[str] = None,
        tokenizer=None,
        seed: int = 42
    ):
        super().__init__()

        self.num_samples = num_samples
        self.num_entities = num_entities
        self.relation_types = relation_types or ['>', '<', '=']
        self.tokenizer = tokenizer
        self.seed = seed

        random.seed(seed)

        # 生成实体（A, B, C, ...）
        self.entities = [chr(ord('A') + i) for i in range(num_entities)]

        # 生成样本
        self.samples = []
        for _ in range(num_samples):
            self.samples.append(self._generate_sample())

    def _generate_sample(self) -> Dict:
        """生成单个推理样本"""
        # 随机选择 3 个不同实体
        entities = random.sample(self.entities, 3)
        A, B, C = entities[0], entities[1], entities[2]

        # 随机选择关系
        rel1 = random.choice(self.relation_types)
        rel2 = random.choice(self.relation_types)

        # 计算传递结果
        if rel1 == rel2 == '>':
            result_rel = '>'
        elif rel1 == rel2 == '<':
            result_rel = '<'
        elif rel1 == '>' and rel2 == '>':
            result_rel = '>'
        elif rel1 == '<' and rel2 == '<':
            result_rel = '<'
        elif rel1 == '=' or rel2 == '=':
            # 等价的传递性
            if rel1 == '=' and rel2 == '=':
                result_rel = '='
            elif rel1 == '=':
                result_rel = rel2
            else:
                result_rel = rel1
        else:
            # 复杂情况：随机选择一个合理答案
            result_rel = random.choice(self.relation_types)

        # 构建问题
        question = f"If {A} {rel1} {B} and {B} {rel2} {C}, what is the relation between {A} and {C}?"
        answer = f"{A} {result_rel} {C}"

        return {
            'entities': entities,
            'relations': [rel1, rel2],
            'question': question,
            'answer': answer,
            'result_relation': result_rel
        }

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        if self.tokenizer is None:
            return sample

        # 转换为 token
        input_text = f"Question: {sample['question']} Answer:"
        target_text = sample['answer']

        full_text = input_text + target_text

        encoded = self.tokenizer(
            full_text,
            return_tensors='pt',
            max_length=128,
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
            'question': sample['question'],
            'answer': sample['answer'],
            'entities': sample['entities'],
            'relations': sample['relations']
        }

    def evaluate_accuracy(self, predictions: List[str]) -> float:
        """
        评估准确率

        Args:
            predictions: 预测答案列表

        Returns:
            准确率
        """
        correct = 0
        for pred, sample in zip(predictions, self.samples):
            # 检查是否包含正确的关系
            if sample['result_relation'] in pred:
                correct += 1
        return correct / len(predictions)


class GroupRelationDataset(Dataset):
    """
    群关系验证数据集

    生成用于测试群公理理解的数据：
    - 交换律：R_a @ R_b = R_b @ R_a
    - 结合律：(R_a @ R_b) @ R_c = R_a @ (R_b @ R_c)
    - 单位元：R_a @ I = R_a
    - 逆元：R_a @ R_a^{-1} = I

    参数：
        num_samples: 样本数量
        num_generators: 生成元数量
        tokenizer: HuggingFace tokenizer
    """

    def __init__(
        self,
        num_samples: int = 1000,
        num_generators: int = 4,
        tokenizer=None,
        seed: int = 42
    ):
        super().__init__()

        self.num_samples = num_samples
        self.num_generators = num_generators
        self.tokenizer = tokenizer
        self.seed = seed

        random.seed(seed)

        self.samples = []
        for _ in range(num_samples):
            self.samples.append(self._generate_sample())

    def _generate_sample(self) -> Dict:
        """生成单个群关系样本"""
        # 随机选择关系类型
        relation_type = random.choice(['commutativity', 'associativity', 'identity', 'inverse'])

        if relation_type == 'commutativity':
            # 交换律测试
            a, b = random.sample(range(self.num_generators), 2)
            question = f"Does generator {a} commute with generator {b}?"
            # 假设某些生成元对是可交换的
            should_commute = random.choice([True, False])
            answer = "Yes" if should_commute else "No"
            relation = 'commutativity'

        elif relation_type == 'associativity':
            # 结合律测试
            a, b, c = random.sample(range(self.num_generators), 3)
            question = f"Is (({a} * {b}) * {c}) = ({a} * ({b} * {c}))?"
            answer = "Yes"  # 群操作总是满足结合律
            relation = 'associativity'

        elif relation_type == 'identity':
            # 单位元测试
            a = random.randint(0, self.num_generators - 1)
            question = f"What is generator {a} * identity?"
            answer = f"Generator {a}"
            relation = 'identity'

        else:  # inverse
            # 逆元测试
            a = random.randint(0, self.num_generators - 1)
            question = f"What is generator {a} * inverse({a})?"
            answer = "identity"
            relation = 'inverse'

        return {
            'relation_type': relation_type,
            'question': question,
            'answer': answer,
            'relation': relation
        }

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        if self.tokenizer is None:
            return sample

        input_text = f"Question: {sample['question']} Answer:"
        target_text = sample['answer']

        full_text = input_text + target_text

        encoded = self.tokenizer(
            full_text,
            return_tensors='pt',
            max_length=128,
            padding='max_length',
            truncation=True
        )

        input_ids = encoded['input_ids'].squeeze(0)
        labels = input_ids.clone()
        input_len = len(self.tokenizer.encode(input_text))
        labels[:input_len] = -100

        return {
            'input_ids': input_ids,
            'labels': labels,
            'question': sample['question'],
            'answer': sample['answer'],
            'relation_type': sample['relation_type']
        }


if __name__ == '__main__':
    # 测试
    from transformers import GPT2Tokenizer

    print("测试合成推理数据集...")

    try:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    except:
        tokenizer = None

    # 传递性数据集
    print("\n=== 传递性推理样本 ===")
    trans_dataset = TransitivityDataset(num_samples=5, tokenizer=tokenizer)
    for i in range(3):
        sample = trans_dataset.samples[i]
        print(f"Q: {sample['question']}")
        print(f"A: {sample['answer']}")
        print()

    # 群关系数据集
    print("=== 群关系样本 ===")
    group_dataset = GroupRelationDataset(num_samples=5, tokenizer=tokenizer)
    for i in range(3):
        sample = group_dataset.samples[i]
        print(f"Q: {sample['question']}")
        print(f"A: {sample['answer']}")
        print()
