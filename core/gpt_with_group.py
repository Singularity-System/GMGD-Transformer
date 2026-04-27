"""
GPTWithGroup: 群扩展 GPT 模型

在 GPT-2 架构中注入群结构先验：
- 继承 GPT2Model，复用预训练 Token/Position 嵌入
- 每一 Transformer 块后追加 GroupSmoothLayer（共 N 层）
- 语言模型头与词嵌入权重共享
- 路径积分机制：全局群状态累乘
- 多群架构：DynamicGroupExpander 自动管理 MetaGroup + GroupAttention

构建方式：
    # 单群模式
    model = GPTWithGroup(base_model_name='gpt2', group_d=32, num_generators=12)

    # 多群模式
    model = GPTWithGroup(
        base_model_name='gpt2',
        group_d=16,
        num_generators=6,
        enable_dynamic_expansion=True
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

import json
import os

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
from transformers import GPT2Model, GPT2Config

from .meta_group import MetaGroup
from .group_attention import GroupAttention
from .group_smooth_layer import GroupSmoothLayer
from .dynamic_expander import DynamicGroupExpander
from dataclasses import dataclass


class GPTWithGroup(nn.Module):
    """
    群扩展 GPT-2 模型（支持动态群扩张）

    在标准 GPT-2 的每一层后添加 GroupSmoothLayer，将隐状态投影到群流形。
    使用 DynamicGroupExpander 自动管理 MetaGroup + GroupAttention。

    参数：
        base_model_name: 基础 GPT-2 模型名称 ('gpt2', 'gpt2-medium', 'gpt2-large')
        group_d: 群表示维数 d（矩阵大小 d×d）
        num_generators: 生成元个数
        group_type: 群类型 ('orthogonal' 或 'general_linear')
        proj_steps: 投影梯度步数
        smooth_lr: 群光滑层学习率
        use_pretrained: 是否使用预训练权重
        enable_dynamic_expansion: 是否启用动态群扩张
        expansion_threshold: 路径积分闭合误差阈值（超过则触发扩张）
        max_generators_per_group: 单子群最大生成元数
    """

    def __init__(
        self,
        base_model_name: str = 'gpt2',
        group_d: int = 32,
        num_generators: int = 12,
        group_type: str = 'orthogonal',
        proj_steps: int = 1,
        smooth_lr: float = 0.1,
        use_pretrained: bool = True,
        enable_dynamic_expansion: bool = False,
        expansion_threshold: float = 0.5,
        max_generators_per_group: int = 12
    ):
        super().__init__()

        self.base_model_name = base_model_name
        self.group_d = group_d
        self.num_generators = num_generators
        self.group_type = group_type
        self.enable_dynamic_expansion = enable_dynamic_expansion

        # 加载基础 GPT-2 模型
        if use_pretrained:
            self.base_model = GPT2Model.from_pretrained(base_model_name)
        else:
            config = GPT2Config.from_pretrained(base_model_name)
            self.base_model = GPT2Model(config)

        self.config = self.base_model.config
        self.hidden_dim = self.config.n_embd
        self.num_layers = self.config.n_layer
        self.vocab_size = self.config.vocab_size

        if enable_dynamic_expansion:
            # 多群模式：DynamicGroupExpander 内含 MetaGroup + GroupAttention
            self.expander = DynamicGroupExpander(
                hidden_dim=self.hidden_dim,
                initial_group_d=group_d,
                initial_num_generators=num_generators,
                group_type=group_type,
                threshold=expansion_threshold,
                max_generators_per_group=max_generators_per_group
            )
            # 所有层共享同一个 MetaGroup + GroupAttention
            shared_meta_group = self.expander.meta_group
            shared_attention = self.expander.attention
            self.smooth_layers = nn.ModuleList([
                GroupSmoothLayer(
                    meta_group=shared_meta_group,
                    attention=shared_attention,
                    proj_steps=proj_steps,
                    smooth_lr=smooth_lr,
                )
                for _ in range(self.num_layers)
            ])
            self.meta_group = None  # 多群模式下无共享 meta_group
        else:
            # 单群模式（向后兼容）
            self.expander = None
            self.meta_group = MetaGroup(
                num_generators=num_generators,
                d=group_d,
                group_type=group_type
            )
            # 为单群模式创建共享注意力
            self.single_attention = GroupAttention(
                hidden_dim=self.hidden_dim,
                num_groups=1,
            )
            self.smooth_layers = nn.ModuleList([
                GroupSmoothLayer(
                    meta_group=self.meta_group,
                    attention=self.single_attention,
                    proj_steps=proj_steps,
                    smooth_lr=smooth_lr,
                )
                for _ in range(self.num_layers)
            ])

        # 语言模型头（与词嵌入共享权重）
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
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
        return_dict: Optional[bool] = True,
        target_group_state: Optional[torch.Tensor] = None,
        global_step: int = 0,
        expansion_warmup_steps: int = 50,
    ) -> Union[Tuple[torch.Tensor], dict]:
        """
        前向传播

        Args:
            input_ids: 输入 token ID (B, S)
            attention_mask: 注意力掩码 (B, S)
            labels: 标签（用于计算损失）
            target_group_state: 目标群状态 (B, d, d)，用于路径积分闭合分析
            global_step: 当前训练步数（用于动态扩张）
            expansion_warmup_steps: 扩张 warmup 步数

        Returns:
            CausalLMOutputWithCrossAttentions with:
                loss: 语言模型损失
                logits: (B, S, vocab_size)
                manifold_distance: 平均流形距离
                global_group_state: (B, d, d)
                expansion_info: 扩张信息（多群模式下）
        """
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
        else:
            batch_size, seq_len = inputs_embeds.shape[:2]

        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        inputs_embeds = self.base_model.wte(input_ids)
        position_embeds = self.base_model.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None

        global_group_state = torch.eye(
            self.group_d, dtype=hidden_states.dtype, device=device
        ).unsqueeze(0).expand(batch_size, -1, -1)
        manifold_distance = 0.0
        expansion_info = None

        # 逐层处理：Transformer 块 + 群光滑层
        for i, (block, smooth_layer) in enumerate(zip(self.base_model.h, self.smooth_layers)):
            hidden_states = block(hidden_states, attention_mask=extended_attention_mask)
            hidden_states, layer_manifold_dist, delta_group = smooth_layer(hidden_states)
            manifold_distance += layer_manifold_dist

            # 路径积分
            delta_group_mean = delta_group.mean(dim=1)
            delta_skew = 0.5 * (delta_group_mean - delta_group_mean.transpose(-2, -1))
            half_delta = 0.5 * delta_skew

            I = torch.eye(self.group_d, dtype=hidden_states.dtype, device=device)
            I_batch = I.unsqueeze(0).expand(batch_size, -1, -1)
            numerator = I_batch + half_delta
            denominator = I_batch - half_delta

            try:
                denominator_inv = torch.linalg.inv(denominator)
                exp_delta = numerator @ denominator_inv
                global_group_state = global_group_state @ exp_delta
            except RuntimeError:
                global_group_state = global_group_state @ (I_batch + delta_group_mean)

        # 动态群扩张分析
        if self.enable_dynamic_expansion:
            target_gs = target_group_state if global_step >= expansion_warmup_steps else None
            expander_output = self.expander(
                hidden_states=hidden_states,
                target_group_state=target_gs,
                global_step=global_step,
                global_group_state=global_group_state
            )

            if global_step >= expansion_warmup_steps and target_group_state is not None:
                expansion_info = {
                    'triggered': expander_output['expansion_triggered'],
                    'info': expander_output['expansion_info'],
                    'path_integral_error': expander_output.get('path_integral_error'),
                    'num_groups': expander_output['num_groups'],
                }
                if expansion_info['triggered']:
                    print(f"[GPTWithGroup] 触发群扩张！当前子群数量：{expansion_info['num_groups']}")

        # 最终层归一化
        hidden_states = self.base_model.ln_f(hidden_states)
        manifold_distance = manifold_distance / len(self.smooth_layers)

        # 语言模型 logits
        logits = self.lm_head(hidden_states)

        # 计算损失
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))

        if not return_dict:
            result = (logits, manifold_distance, global_group_state)
            if loss is not None:
                result = result + (loss,)
            if expansion_info is not None:
                result = result + (expansion_info,)
            return result

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            manifold_distance=manifold_distance,
            global_group_state=global_group_state,
            router_loss=None,
            expansion_info=expansion_info,
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
        """文本生成（自回归贪婪/采样解码）"""
        if max_new_tokens is not None:
            max_length = input_ids.shape[1] + max_new_tokens

        self.eval()
        generated = input_ids.clone()

        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.config.eos_token_id

        batch_size = generated.shape[0]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=generated.device)

        with torch.no_grad():
            while generated.shape[1] < max_length:
                outputs = self(generated[:, -1024:])
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

                # 屏蔽 PAD token（pad==eos 时避免提前终止）
                if pad_token_id == eos_token_id:
                    logits_for_selection = logits.clone()
                    logits_for_selection[:, pad_token_id] = float('-inf')
                    if not do_sample:
                        next_token = torch.argmax(logits_for_selection, dim=-1, keepdim=True)
                    else:
                        probs = torch.softmax(logits_for_selection, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)

                is_eos = (next_token.squeeze(-1) == eos_token_id)
                finished = finished | is_eos

                next_token = next_token.masked_fill(
                    finished.unsqueeze(-1), pad_token_id
                )
                generated = torch.cat([generated, next_token], dim=1)

                if finished.all():
                    break

        return generated

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_meta_group(self) -> MetaGroup:
        """获取关联的 MetaGroup（单群模式）"""
        if self.meta_group is not None:
            return self.meta_group
        if self.expander is not None:
            return self.expander.registry.meta_group
        return None

    def get_smooth_layers(self) -> nn.ModuleList:
        return self.smooth_layers

    @classmethod
    def from_config(cls, config: dict) -> 'GPTWithGroup':
        return cls(
            base_model_name=config.get('base_model_name', 'gpt2'),
            group_d=config.get('group_d', 32),
            num_generators=config.get('num_generators', 12),
            group_type=config.get('group_type', 'orthogonal'),
            proj_steps=config.get('proj_steps', 1),
            smooth_lr=config.get('smooth_lr', 0.1),
            use_pretrained=config.get('use_pretrained', True),
            enable_dynamic_expansion=config.get('enable_dynamic_expansion', False),
        )

    def save_pretrained(self, save_dir: str) -> None:
        """保存模型（包括群相关参数）"""
        self.base_model.save_pretrained(save_dir)

        if self.enable_dynamic_expansion and self.expander is not None:
            state_dict = {
                'expander': self.expander.state_dict(),
                'smooth_layers': self.smooth_layers.state_dict(),
                'lm_head': self.lm_head.state_dict(),
            }
            config = {
                'enable_dynamic_expansion': True,
                'group_d': self.group_d,
                'num_generators': self.num_generators,
                'group_type': self.group_type,
            }
        else:
            state_dict = {
                'meta_group': self.meta_group.state_dict(),
                'smooth_layers': self.smooth_layers.state_dict(),
                'lm_head': self.lm_head.state_dict(),
            }
            config = {
                'enable_dynamic_expansion': False,
                'group_d': self.group_d,
                'num_generators': self.num_generators,
                'group_type': self.group_type,
            }
        torch.save(state_dict, f'{save_dir}/group_state.pt')
        with open(f'{save_dir}/group_config.json', 'w') as f:
            json.dump(config, f)

    @classmethod
    def load_pretrained(cls, load_dir: str, device: torch.device = None) -> 'GPTWithGroup':
        if device is None:
            device = torch.device('cpu')

        group_state = torch.load(f'{load_dir}/group_state.pt', map_location=device)
        config_path = f'{load_dir}/group_config.json'

        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            model = cls(
                base_model_name=load_dir,
                use_pretrained=True,
                enable_dynamic_expansion=config.get('enable_dynamic_expansion', False),
                group_d=config.get('group_d', 32),
                num_generators=config.get('num_generators', 12),
                group_type=config.get('group_type', 'orthogonal'),
            )
        else:
            is_multi_group = 'expander' in group_state
            model = cls(
                base_model_name=load_dir,
                use_pretrained=True,
                enable_dynamic_expansion=is_multi_group,
            )

        if 'meta_group' in group_state and model.meta_group is not None:
            model.meta_group.load_state_dict(group_state['meta_group'])

        if 'expander' in group_state and model.expander is not None:
            model.expander.load_state_dict(group_state['expander'])

        # Load smooth_layers
        smooth_state = group_state['smooth_layers']
        model.smooth_layers.load_state_dict(smooth_state, strict=False)
        model.lm_head.load_state_dict(group_state['lm_head'])

        return model.to(device)


@dataclass
class CausalLMOutputWithCrossAttentions:
    """
    简化的 CausalLMOutputWithCrossAttentions 实现

    属性：
        loss: 语言模型损失
        logits: (B, S, vocab_size)
        manifold_distance: 平均流形距离
        global_group_state: (B, d, d)
        expansion_info: 扩张信息（多群模式）
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    manifold_distance: Optional[torch.FloatTensor] = None
    global_group_state: Optional[torch.FloatTensor] = None
    router_loss: Optional[torch.FloatTensor] = None
    expansion_info: Optional[dict] = None
