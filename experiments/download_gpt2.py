#!/usr/bin/env python3
"""
使用镜像站下载 GPT-2 模型和 tokenizer
"""

import os

# 设置镜像站（必须在导入 transformers 之前设置）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import GPT2Tokenizer, GPT2LMHeadModel

print("=" * 60)
print("使用镜像站下载 GPT-2 模型")
print("=" * 60)
print(f"镜像地址：{os.environ['HF_ENDPOINT']}")
print()

# 下载 tokenizer
print("步骤 1/2: 下载 tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
print("✓ Tokenizer 下载完成")
print()

# 下载模型
print("步骤 2/2: 下载 GPT2LMHeadModel...")
model = GPT2LMHeadModel.from_pretrained('gpt2')
print("✓ 模型下载完成")
print()

print("=" * 60)
print("下载完成！")
print("=" * 60)
print(f"模型参数：{sum(p.numel() for p in model.parameters()):,}")
print(f"词表大小：{tokenizer.vocab_size:,}")
print()
print("模型已缓存到本地，后续运行无需重复下载")
