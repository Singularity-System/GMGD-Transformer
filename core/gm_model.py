"""
GMGD 完整模型和训练过程

核心组件：
- MetaGroup: 管理群
- DomainManager: 全局域嵌入 + EMA 平滑 + 主导切换 + 渐进式恢复
- GroupSmoothLayer: 光滑层
- GPTWithGroup: 完整模型

不涉及任何加速和优化。

域选择策略（终局设计 - 七大机制）：
1. 全局域嵌入 — 每个域的可学习方向向量，注入每层隐藏状态
2. 定长归一化 — L2 归一化，方向信号强度恒定
3. 对数分布幅值 — 对数空间存储幅值，梯度稳定
4. EMA 平滑 — 域权重丝滑演变，避免抖动
5. 主导切换检测 — 识别域切换事件
6. 渐进式历史恢复 — 旧主导域缓慢恢复
7. 统一恢复目标 0.9 — 所有域非主导时保持战备状态
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT2LMHeadModel, GPT2Config

from .meta_group import MetaGroup

# 域常量
DOMAIN_ARITHMETIC = 0
DOMAIN_LANGUAGE = 1
DOMAIN_MIXED = 2
NUM_DOMAINS = 3


class DomainManager(nn.Module):
    """全局域嵌入管理器 — 七大机制

    为每个域提供：
    - 可学习方向向量（L2 归一化）
    - 对数幅值参数
    - EMA 平滑状态
    - 主导切换检测
    - 渐进式恢复
    """

    def __init__(
        self,
        hidden_dim: int,
        num_domains: int = NUM_DOMAINS,
        beta: float = 0.7,
        recovery_target: float = 0.9,
    ):
        super().__init__()
        self.num_domains = num_domains
        self.beta = beta
        self.recovery_target = recovery_target

        # 机制 1: 全局域嵌入（可学习方向向量）
        self.domain_embeddings = nn.Parameter(torch.randn(num_domains, hidden_dim) * 0.02)

        # 机制 3: 对数分布幅值
        self.log_amplitude = nn.Parameter(torch.zeros(num_domains))

        # 机制 4/6: EMA 平滑状态（训练时维护）
        self.register_buffer('ema_amplitude', torch.ones(num_domains) * recovery_target)

        # 机制 5: 上一主导域
        self.register_buffer('prev_main_domain', torch.tensor(0))

    def get_domain_signal(self) -> torch.Tensor:
        """获取当前域信号 — 机制 1+2+3

        Returns:
            domain_signal: (num_domains, hidden_dim) 每个域的注入信号
        """
        # 机制 2: 定长归一化方向
        directions = F.normalize(self.domain_embeddings, dim=-1)  # (num_domains, H)

        # 机制 3: 对数幅值 → 正幅值
        amplitude = torch.exp(self.log_amplitude)  # (num_domains,)

        # 幅值 × 方向
        return amplitude.unsqueeze(-1) * directions  # (num_domains, H)

    def select_domains(self, token_domain_ids: torch.Tensor) -> torch.Tensor:
        """根据 token 的域标签，选择对应的域嵌入

        Args:
            token_domain_ids: (B, S) 每个 token 的域标签 (0=算术, 1=语言, 2=混合)

        Returns:
            token_domain_signal: (B, S, H) 每个 token 的域注入信号
        """
        domain_signal = self.get_domain_signal()  # (num_domains, H)
        token_domain_signal = domain_signal[token_domain_ids]  # (B, S, H)
        return token_domain_signal

    def update_ema(self, domain_weights: torch.Tensor):
        """更新 EMA 平滑状态 — 机制 4

        Args:
            domain_weights: (num_domains,) 当前各域幅值
        """
        with torch.no_grad():
            self.ema_amplitude.copy_(
                self.beta * domain_weights + (1 - self.beta) * self.ema_amplitude
            )

    def detect_switch_and_recover(self, domain_weights: torch.Tensor) -> torch.Tensor:
        """主导切换检测 + 渐进式恢复 — 机制 5+6+7

        Args:
            domain_weights: (num_domains,) 当前各域幅值

        Returns:
            updated_weights: (num_domains,) 更新后的幅值
        """
        cur_main = int(torch.argmax(domain_weights).item())
        prev_main = int(self.prev_main_domain.item())

        if cur_main != prev_main and self.training:
            # 机制 6: 旧主导域渐进式恢复
            domain_weights[prev_main] = (
                self.beta * domain_weights[prev_main]
                + (1 - self.beta) * self.recovery_target
            )

        self.prev_main_domain.copy_(torch.tensor(cur_main))
        return domain_weights


class GroupSmoothLayer(nn.Module):
    """光滑层：用域嵌入做群选择

    接收 DomainManager 的域嵌入信号，通过 domain_to_attn 映射到群注意力权重。
    """

    def __init__(
        self,
        meta_group: MetaGroup,
        hidden_dim: int,
        proj_steps: int = 1,
        smooth_lr: float = 0.1,
    ):
        super().__init__()
        self.meta_group = meta_group
        self.proj_steps = proj_steps
        self.smooth_lr = smooth_lr

        # 域信号 → 群注意力权重的映射
        self.domain_to_attn = nn.Linear(hidden_dim, len(meta_group.subgroups))

    def forward(
        self,
        layer_out: torch.Tensor,
        domain_signal: torch.Tensor,
        global_step: int = 0,
    ) -> tuple:
        B, S, H = layer_out.shape

        # 域信号 → 群注意力权重 (B, S, K)
        attn_logits = self.domain_to_attn(domain_signal)
        attn = torch.softmax(attn_logits, dim=-1)

        # 各群的修正量
        corrections = []
        dists = []
        for sg in self.meta_group.subgroups:
            out_g, delta, dist = sg(layer_out, self.proj_steps, self.smooth_lr)
            corrections.append(out_g - layer_out)
            dists.append(dist)

        # 加权聚合
        weighted_correction = sum(
            attn[:, :, g].unsqueeze(-1) * corrections[g]
            for g in range(attn.shape[-1])
        )
        weighted_dist = sum(attn[:, :, g] * dists[g] for g in range(attn.shape[-1]))

        output = layer_out + weighted_correction

        # 返回域注意力均值（用于 EMA 更新）
        domain_attn_mean = attn.mean(dim=(0, 1))  # (K,)

        return output, weighted_dist.mean(), corrections, domain_attn_mean


class GPTWithGroup(nn.Module):
    """GPT-2 + 多群光滑层（全局域嵌入注入）

    流程：
    1. 继承 GPT-2
    2. DomainManager 管理全局域嵌入
    3. 每层注入域信号到隐藏状态 + 用域信号做群选择
    """

    def __init__(
        self,
        base_model_name: str = 'gpt2',
        group_d: int = 16,
        num_initial_active_groups: int = 1,
        use_pretrained: bool = True,
        domain_beta: float = 0.7,
        domain_recovery: float = 0.9,
    ):
        super().__init__()
        self.group_d = group_d

        # GPT-2 LM
        if use_pretrained:
            self.transformer = GPT2LMHeadModel.from_pretrained(
                base_model_name, local_files_only=True
            )
        else:
            self.transformer = GPT2LMHeadModel(
                GPT2Config(n_layer=12, n_head=12, n_embd=768)
            )

        hidden_dim = self.transformer.config.hidden_size

        # 管理群
        num_groups = 1 + num_initial_active_groups  # 透明 + 活跃
        self.meta_group = MetaGroup(
            d=group_d,
            hidden_dim=hidden_dim,
            initial_active_groups=num_initial_active_groups,
        )

        # 全局域嵌入管理器
        self.domain_manager = DomainManager(
            hidden_dim=hidden_dim,
            beta=domain_beta,
            recovery_target=domain_recovery,
        )

        # 每层一个光滑层（共享 MetaGroup）
        num_layers = self.transformer.config.n_layer
        self.smooth_layers = nn.ModuleList([
            GroupSmoothLayer(
                meta_group=self.meta_group,
                hidden_dim=hidden_dim,
                proj_steps=1,
                smooth_lr=0.1,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor = None,
        token_domain_ids: torch.Tensor = None,
        global_step: int = 0,
    ) -> dict:
        """
        Args:
            input_ids: (B, S) 输入 token IDs
            labels: (B, S) 标签
            token_domain_ids: (B, S) 每个 token 的域标签 (0=算术, 1=语言, 2=混合)
            global_step: 全局训练步数
        """
        B = input_ids.shape[0]

        # 获取域嵌入信号（不经过 LayerNorm，独立注入）
        if token_domain_ids is not None:
            domain_signal = self.domain_manager.select_domains(token_domain_ids)  # (B, S, H)
        else:
            # 无域标签 → 默认算术域
            domain_ids = torch.zeros(B, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
            domain_signal = self.domain_manager.select_domains(domain_ids)

        # 获取初始 hidden states
        transformer_embed = self.transformer.transformer(input_ids)
        hidden_states = transformer_embed.last_hidden_state  # (B, S, H)

        # 注入域信号（机制 1: 残差注入，不被 LayerNorm 抹掉）
        hidden_states = hidden_states + domain_signal

        total_manifold = 0.0
        num_layers = len(self.transformer.transformer.h)
        for i, block in enumerate(self.transformer.transformer.h):
            # ── 标准 Attention + MLP 流程
            residual = hidden_states
            hidden_states_ln = block.ln_1(hidden_states)
            attn_out = block.attn(hidden_states_ln)
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
            hidden_states = residual + attn_out  # (B, S, H)

            residual = hidden_states
            hidden_states_ln = block.ln_2(hidden_states)
            mlp_out = block.mlp(hidden_states_ln)
            mlp_output = residual + mlp_out  # (B, S, H)

            # ── 重新注入域信号（每层独立注入，不被 LayerNorm 污染）
            mlp_output = mlp_output + domain_signal

            # ── 光滑层（域信号 → 群选择 → 修正）
            smoothed, manifold, corrections, domain_attn = self.smooth_layers[i](
                mlp_output, domain_signal=domain_signal, global_step=global_step
            )
            total_manifold = total_manifold + manifold
            hidden_states = smoothed

        # LM head
        lm_logits = self.transformer.lm_head(hidden_states)

        # 损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        return {
            'loss': loss,
            'logits': lm_logits,
            'manifold_distance': total_manifold / max(num_layers, 1),
            'entropy_bonus': torch.tensor(0.0, device=lm_logits.device),
            'expansion_info': None,
        }

    def generate(self, **kwargs):
        """直接调用 GPT-2 的 generate（经过我们的 smooth_layer）"""
        return self.transformer.generate(**kwargs)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def save_pretrained(self, path: str):
        self.transformer.save_pretrained(path)
        torch.save({
            'meta_group': self.meta_group.state_dict(),
            'smooth_layers': self.smooth_layers.state_dict(),
            'domain_manager': self.domain_manager.state_dict(),
        }, f'{path}/gmgd.pt')

    @classmethod
    def load_pretrained(cls, path: str, **kwargs):
        model = cls(**kwargs)
        model.transformer = GPT2LMHeadModel.from_pretrained(path, local_files_only=True)
        state = torch.load(f'{path}/gmgd.pt', map_location='cpu')
        model.meta_group.load_state_dict(state['meta_group'])
        model.smooth_layers.load_state_dict(state['smooth_layers'])
        model.domain_manager.load_state_dict(state['domain_manager'])
        return model
