"""
GPTWithGroup: 群扩展 GPT 模型

在 GPT-2 架构中注入群结构先验：
- 继承 GPT2Model，复用预训练 Token/Position 嵌入
- 每一 Transformer 块后追加 GroupSmoothLayer（共 N 层）
- 语言模型头与词嵌入权重共享
- 路径积分机制：全局群状态累乘

构建方式：
    model = GPTWithGroup(
        base_model_name='gpt2',
        group_d=32,
        num_generators=12
    )

前向传播流程：
    x = wte(input_ids) + wpe(positions)
    global_group_state = I
    for block, smooth in zip(blocks, smooth_layers):
        x = block(x, attention_mask)
        x, manifold_dist, delta_group = smooth(x)
        global_group_state = global_group_state @ exp(delta_group)
    x = ln_f(x)
    return lm_head(x)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
from transformers import GPT2Model, GPT2Config

from .meta_group import MetaGroup
from .group_smooth_layer import GroupSmoothLayer
from dataclasses import dataclass


class GPTWithGroup(nn.Module):
    """
    群扩展 GPT-2 模型

    在标准 GPT-2 的每一层后添加 GroupSmoothLayer，将隐状态投影到群流形。

    参数：
        base_model_name: 基础 GPT-2 模型名称 ('gpt2', 'gpt2-medium', 'gpt2-large')
        group_d: 群表示维数 d（矩阵大小 d×d）
        num_generators: 生成元个数
        group_type: 群类型 ('orthogonal' 或 'general_linear')
        proj_steps: 投影梯度步数
        smooth_lr: 群光滑层学习率
        use_pretrained: 是否使用预训练权重

    示例：
        >>> model = GPTWithGroup(
        ...     base_model_name='gpt2',
        ...     group_d=32,
        ...     num_generators=12
        ... )
        >>> input_ids = torch.randint(0, 50257, (4, 100))
        >>> logits = model(input_ids)  # (4, 100, 50257)
    """

    def __init__(
        self,
        base_model_name: str = 'gpt2',
        group_d: int = 32,
        num_generators: int = 12,
        group_type: str = 'orthogonal',
        proj_steps: int = 1,
        smooth_lr: float = 0.1,
        use_pretrained: bool = True
    ):
        super().__init__()

        self.base_model_name = base_model_name
        self.group_d = group_d
        self.num_generators = num_generators
        self.group_type = group_type

        # 加载基础 GPT-2 模型（使用 GPT2Model 以便在每一层后插入群光滑层）
        if use_pretrained:
            self.base_model = GPT2Model.from_pretrained(base_model_name)
        else:
            config = GPT2Config.from_pretrained(base_model_name)
            self.base_model = GPT2Model(config)

        # 获取模型配置
        self.config = self.base_model.config
        self.hidden_dim = self.config.n_embd  # 768 for gpt2, 1024 for gpt2-medium
        self.num_layers = self.config.n_layer  # 12 for gpt2, 24 for gpt2-medium
        self.vocab_size = self.config.vocab_size  # 50257

        # 创建共享的 MetaGroup
        self.meta_group = MetaGroup(
            num_generators=num_generators,
            d=group_d,
            group_type=group_type
        )

        # 为每一层创建 GroupSmoothLayer
        self.smooth_layers = nn.ModuleList([
            GroupSmoothLayer(
                hidden_dim=self.hidden_dim,
                group_d=group_d,
                meta_group=self.meta_group,  # 共享同一元群
                proj_steps=proj_steps,
                smooth_lr=smooth_lr
            )
            for _ in range(self.num_layers)
        ])

        # 语言模型头（与词嵌入共享权重）
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
        # 绑定权重（GPT2Model 的 wte）
        self.lm_head.weight = self.base_model.wte.weight

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,  # 默认返回 dict 格式
    ) -> Union[Tuple[torch.Tensor], dict]:
        """
        前向传播

        Args:
            input_ids: 输入 token ID (B, S)
            attention_mask: 注意力掩码 (B, S)
            labels: 标签（用于计算损失）
            其他参数与 GPT2Model 相同

        Returns:
            logits: 语言模型输出 (B, S, vocab_size)
            loss: 若提供 labels，返回交叉熵损失
            manifold_distance: 平均流形距离（用于正则化）
            global_group_state: 全局群状态 (B, d, d)
        """
        # 关键设计：每一层 Transformer 块后立即应用群光滑层
        # 流程：for block, smooth in zip(blocks, smooth_layers):
        #           x = block(x, attention_mask)[0]
        #           x = smooth(x)
        #       x = ln_f(x)
        #       return lm_head(x)

        # 获取词嵌入和位置嵌入
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_len = input_shape
        else:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_len = input_shape

        # 生成 position_ids（GPT2Model 没有 position_ids 属性，需要手动创建）
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # 初始嵌入
        if inputs_embeds is None:
            inputs_embeds = self.base_model.wte(input_ids)
        position_embeds = self.base_model.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # 注意力掩码
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None

        # 初始化全局群状态
        global_group_state = torch.eye(
            self.group_d, dtype=hidden_states.dtype, device=device
        ).unsqueeze(0).expand(batch_size, -1, -1)
        manifold_distance = 0.0

        # 逐层处理：Transformer 块 + 群光滑层
        for i, (block, smooth_layer) in enumerate(zip(self.base_model.h, self.smooth_layers)):
            # Transformer 块（GPT2Block 返回张量，不是 tuple）
            hidden_states = block(hidden_states, attention_mask=extended_attention_mask)

            # 群光滑层
            hidden_states, layer_manifold_dist, delta_group = smooth_layer(hidden_states)
            manifold_distance += layer_manifold_dist

            # 路径积分：累乘群增量（使用 Cayley 变换）
            # delta_group: (B, S, d, d) → mean(dim=1) → (B, d, d)
            delta_group_mean = delta_group.mean(dim=1)
            delta_skew = 0.5 * (delta_group_mean - delta_group_mean.transpose(-2, -1))
            half_delta = 0.5 * delta_skew

            # Cayley 变换：exp(delta) ≈ (I + delta/2) @ (I - delta/2)^{-1}
            I = torch.eye(self.group_d, dtype=hidden_states.dtype, device=device)
            I_batch = I.unsqueeze(0).expand(batch_size, -1, -1)
            numerator = I_batch + half_delta
            denominator = I_batch - half_delta

            try:
                denominator_inv = torch.linalg.inv(denominator)
                exp_delta = numerator @ denominator_inv
                global_group_state = global_group_state @ exp_delta
            except RuntimeError:
                # 数值不稳定时回退到一阶近似
                global_group_state = global_group_state @ (I_batch + delta_group_mean)

        # 最终层归一化
        hidden_states = self.base_model.ln_f(hidden_states)

        # 平均流形距离
        manifold_distance = manifold_distance / len(self.smooth_layers)

        # 计算语言模型 logits
        logits = self.lm_head(hidden_states)

        # 计算损失
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))

        # 返回
        if not return_dict:
            return (logits, manifold_distance, global_group_state) + (loss,) if loss is not None else (logits, manifold_distance, global_group_state)

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            manifold_distance=manifold_distance,
            global_group_state=global_group_state,
        )

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: int = 100,
        max_new_tokens: int = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.LongTensor:
        """
        文本生成（自回归贪婪/采样解码）

        Args:
            input_ids: 输入 prompt 的 token ID (B, S)
            max_length: 最大生成长度
            max_new_tokens: 最大新生成 token 数（优先于 max_length）
            do_sample: 是否采样
            temperature: 采样温度
            top_k: Top-k 采样
            pad_token_id: PAD token ID
            eos_token_id: EOS token ID

        Returns:
            生成的 token ID 序列 (B, L)
        """
        if max_new_tokens is not None:
            max_length = input_ids.shape[1] + max_new_tokens

        self.eval()
        generated = input_ids.clone()

        # 获取 EOS token ID
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.config.eos_token_id  # GPT-2 默认 PAD = EOS

        # 跟踪每个样本是否已结束
        batch_size = generated.shape[0]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=generated.device)

        with torch.no_grad():
            while generated.shape[1] < max_length:
                outputs = self(generated[:, -1024:])  # GPT-2 上下文窗口限制
                logits = outputs.logits[:, -1, :]

                if do_sample:
                    logits = logits / temperature
                    if top_k > 0:
                        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                        logits[indices_to_remove] = float('-inf')
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                # 关键修复：在生成阶段屏蔽 PAD token（当 pad_token_id == eos_token_id 时）
                # 训练数据中 PAD 占主导导致模型偏向预测 EOS
                if pad_token_id == eos_token_id:
                    logits_for_selection = logits.clone()
                    logits_for_selection[:, pad_token_id] = float('-inf')
                    if not do_sample:
                        next_token = torch.argmax(logits_for_selection, dim=-1, keepdim=True)
                    else:
                        probs = torch.softmax(logits_for_selection, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)

                # 检查哪些样本生成了 EOS
                is_eos = (next_token.squeeze(-1) == eos_token_id)
                is_eos = (next_token.squeeze(-1) == eos_token_id)
                finished = finished | is_eos

                # 已结束的样本保持 PAD token
                next_token = next_token.masked_fill(
                    finished.unsqueeze(-1),
                    pad_token_id
                )

                generated = torch.cat([generated, next_token], dim=1)

                # 如果所有样本都结束了，停止生成
                if finished.all():
                    break

        return generated

    def get_num_params(self) -> int:
        """获取模型总参数数量"""
        return sum(p.numel() for p in self.parameters())

    def get_meta_group(self) -> MetaGroup:
        """获取关联的 MetaGroup 模块"""
        return self.meta_group

    def get_smooth_layers(self) -> nn.ModuleList:
        """获取所有群光滑层"""
        return self.smooth_layers

    @classmethod
    def from_config(cls, config: dict) -> 'GPTWithGroup':
        """
        从配置字典创建模型

        Args:
            config: 配置字典，包含 base_model_name, group_d 等

        Returns:
            GPTWithGroup 实例
        """
        return cls(
            base_model_name=config.get('base_model_name', 'gpt2'),
            group_d=config.get('group_d', 32),
            num_generators=config.get('num_generators', 12),
            group_type=config.get('group_type', 'orthogonal'),
            proj_steps=config.get('proj_steps', 1),
            smooth_lr=config.get('smooth_lr', 0.1),
            use_pretrained=config.get('use_pretrained', True)
        )

    def save_pretrained(self, save_dir: str) -> None:
        """
        保存模型（包括 MetaGroup 状态）

        Args:
            save_dir: 保存目录
        """
        # 保存基础模型
        self.base_model.save_pretrained(save_dir)

        # 保存群相关参数
        state_dict = {
            'meta_group': self.meta_group.state_dict(),
            'smooth_layers': self.smooth_layers.state_dict(),
            'lm_head': self.lm_head.state_dict(),
        }
        torch.save(state_dict, f'{save_dir}/group_state.pt')

    @classmethod
    def load_pretrained(cls, load_dir: str, device: torch.device = None) -> 'GPTWithGroup':
        """
        加载预训练模型

        Args:
            load_dir: 模型目录
            device: 加载设备

        Returns:
            GPTWithGroup 实例
        """
        if device is None:
            device = torch.device('cpu')

        # 创建模型
        model = cls(
            base_model_name=load_dir,
            use_pretrained=True
        )

        # 加载群相关参数
        group_state = torch.load(f'{load_dir}/group_state.pt', map_location=device)
        model.meta_group.load_state_dict(group_state['meta_group'])
        model.smooth_layers.load_state_dict(group_state['smooth_layers'])
        model.lm_head.load_state_dict(group_state['lm_head'])

        return model.to(device)


# 兼容 transformers 的输出类型
@dataclass
class CausalLMOutputWithCrossAttentions:
    """
    简化的 CausalLMOutputWithCrossAttentions 实现
    （避免直接依赖 transformers 的内部类型）

    属性：
        loss: 语言模型损失（若提供 labels）
        logits: 语言模型输出 (B, S, vocab_size)
        manifold_distance: 平均流形距离（标量，用于正则化）
        global_group_state: 全局群状态 (B, d, d)
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    manifold_distance: Optional[torch.FloatTensor] = None
    global_group_state: Optional[torch.FloatTensor] = None
