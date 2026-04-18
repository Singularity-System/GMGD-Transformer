"""
GPTWithGroup: 群扩展 GPT 模型

在 GPT-2 架构中注入群结构先验：
- 继承 GPT2Model，复用预训练 Token/Position 嵌入
- 每一 Transformer 块后追加 GroupSmoothLayer（共 N 层）
- 语言模型头与词嵌入权重共享

构建方式：
    model = GPTWithGroup(
        base_model_name='gpt2',
        group_d=32,
        num_generators=12
    )

前向传播流程：
    x = wte(input_ids) + wpe(positions)
    for block, smooth in zip(blocks, smooth_layers):
        x = block(x, attention_mask)[0]
        x = smooth(x)
    x = ln_f(x)
    return lm_head(x)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Config

from meta_group import MetaGroup
from group_smooth_layer import GroupSmoothLayer


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

        # 加载基础 GPT-2 模型（使用 GPT2LMHeadModel 以便支持 generate）
        if use_pretrained:
            self.base_model = GPT2LMHeadModel.from_pretrained(base_model_name)
        else:
            config = GPT2Config.from_pretrained(base_model_name)
            self.base_model = GPT2LMHeadModel(config)

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
        # 绑定权重（GPT2LMHeadModel 的结构）
        self.lm_head.weight = self.base_model.transformer.wte.weight

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
        return_dict: Optional[bool] = True,  # 默认为 True
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
        """
        # 通过基础 GPT-2 模型获取隐藏状态
        # 使用 transformer 层而不是 LMHeadModel 来获取 hidden states
        transformer_outputs = self.base_model.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取最后一层隐藏状态
        hidden_states = transformer_outputs[0]  # (B, S, H)

        # 依次通过每一层的群光滑层
        # 关键改进：全局群状态累乘（路径积分机制）
        # 每一层在同一个群流形轨道上继续滑行，而不是独立投影

        # 初始化全局群状态为单位矩阵 (B, d, d)
        batch_size = hidden_states.shape[0]
        global_group_state = torch.eye(
            self.group_d,
            dtype=hidden_states.dtype,
            device=hidden_states.device
        ).unsqueeze(0).expand(batch_size, -1, -1)  # (B, d, d)

        # 累积流形距离和群增量
        manifold_distance = 0.0
        layer_delta_groups = []  # 记录每层贡献用于分析

        for i, smooth_layer in enumerate(self.smooth_layers):
            hidden_states, layer_manifold_dist, delta_group = smooth_layer(hidden_states)
            manifold_distance = manifold_distance + layer_manifold_dist
            layer_delta_groups.append(delta_group)

            # 累乘群增量：使用指数映射保证结果仍在群流形上
            # global_state = global_state @ exp(mean(delta_group))
            # 对序列维度取平均，得到 (B, d, d) 的群增量
            delta_group_mean = delta_group.mean(dim=1)  # (B, d, d)

            # 使用 Cayley 变换将增量映射到群流形（保持正交性）
            # 对于正交群，exp(δ) ≈ (I + δ/2) @ (I - δ/2)^{-1}
            d = self.group_d
            I = torch.eye(d, dtype=hidden_states.dtype, device=hidden_states.device)
            I_batch = I.unsqueeze(0).expand(batch_size, -1, -1)

            # 取反对称部分（保证指数映射到正交群）
            delta_skew = 0.5 * (delta_group_mean - delta_group_mean.transpose(-2, -1))

            # Cayley 变换：exp(δ) ≈ (I + δ/2) @ (I - δ/2)^{-1}
            half_delta = 0.5 * delta_skew
            numerator = I_batch + half_delta
            denominator = I_batch - half_delta

            # 矩阵求逆（批量）
            try:
                denominator_inv = torch.linalg.inv(denominator)
                exp_delta = numerator @ denominator_inv

                # 累乘：global_state = global_state @ exp(δ)
                global_group_state = global_group_state @ exp_delta
            except RuntimeError:
                # 求逆失败时回退到简单加法
                global_group_state = global_group_state @ (I_batch + delta_group_mean)

        # 平均流形距离（跨层数）
        manifold_distance = manifold_distance / len(self.smooth_layers)

        # 计算语言模型 logits
        logits = self.lm_head(hidden_states)

        # 计算损失（若提供 labels）
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1)
            )

        # 返回
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            manifold_distance=manifold_distance,
            global_group_state=global_group_state,  # 全局群状态（路径积分结果）
        )

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: int = 100,
        max_new_tokens: int = None,
        num_beams: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.LongTensor:
        """
        文本生成（简化版本，避免与 transformers 新版本冲突）

        Args:
            input_ids: 输入 prompt 的 token ID
            max_length: 最大生成长度
            max_new_tokens: 最大新生成 token 数（优先于 max_length）
            num_beams: Beam search 的 beam 数
            do_sample: 是否采样
            temperature: 采样温度
            top_k: Top-k 采样
            top_p: Nucleus 采样
            pad_token_id: Padding token ID
            eos_token_id: EOS token ID

        Returns:
            生成的 token ID 序列
        """
        from transformers import GenerationConfig

        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        # 计算实际的 max_length
        if max_new_tokens is not None:
            actual_max_length = input_ids.shape[1] + max_new_tokens
        else:
            actual_max_length = max_length

        generation_config = GenerationConfig(
            max_length=actual_max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

        # 使用 base_model 的 generate 方法（它继承自 GenerationMixin）
        return self.base_model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            **kwargs
        )

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
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class CausalLMOutputWithCrossAttentions:
    """
    简化的 CausalLMOutputWithCrossAttentions 实现
    （避免直接依赖 transformers 的内部类型）
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    manifold_distance: Optional[torch.FloatTensor] = None
    global_group_state: Optional[torch.FloatTensor] = None  # 全局群状态 (B, d, d)
