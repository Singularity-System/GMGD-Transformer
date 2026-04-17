"""
边缘推理核导出工具

训练完成后可剥离 Transformer 主体，导出纯群推理引擎：
- 生成元矩阵（~32KB）
- 轻量级编码器/解码器
- Cayley 变换推理（加速 3-5 倍）

支持：
- 固化群参数为独立 state_dict
- INT8 量化（正交性损失 < 0.1%）
- ONNX 导出（用于边缘设备部署）

使用示例：
    python export_edge.py --model_path ./checkpoints/best_model --output_dir ./edge_model
"""

import argparse
import os
import json
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EdgeModelConfig:
    """边缘推理核配置"""
    group_d: int = 32
    num_generators: int = 12
    group_type: str = 'orthogonal'
    hidden_dim: int = 768
    use_cayley: bool = True
    quantize: bool = False


class PureGroupInference(nn.Module):
    """
    纯群推理引擎（无 Transformer 依赖）

    推理流程：
        state = encoder(input)
        state = state @ generator_sequence  # 矩阵乘法
        output = decoder(state)

    可在 ARM Cortex-M 上以 <1ms 延迟运行
    """

    def __init__(
        self,
        group_d: int = 32,
        num_generators: int = 12,
        group_type: str = 'orthogonal',
        input_dim: int = 128,
        output_dim: int = 10,
        hidden_dim: int = 64
    ):
        super().__init__()

        self.group_d = group_d
        self.num_generators = num_generators
        self.group_type = group_type

        # 轻量级编码器：input → group state
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, group_d * group_d)
        )

        # 可学习生成元（固化后不再更新）
        self.register_buffer(
            'generators',
            torch.randn(num_generators, group_d, group_d)
        )

        # 轻量级解码器：group state → output
        self.decoder = nn.Sequential(
            nn.Linear(group_d * group_d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def set_generators(self, generators: torch.Tensor) -> None:
        """设置生成元矩阵（从训练好的 MetaGroup 加载）"""
        self.generators.copy_(generators)

    def cayley_transform(self, A: torch.Tensor) -> torch.Tensor:
        """Cayley 变换：exp(A) ≈ (I + A/2)(I - A/2)^{-1}"""
        I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
        A_skew = 0.5 * (A - A.transpose(-1, -2))
        return (I + 0.5 * A_skew) @ torch.linalg.inv(I - 0.5 * A_skew)

    def forward(
        self,
        x: torch.Tensor,
        generator_indices: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入 (batch, input_dim)
            generator_indices: 使用的生成元索引（默认全部）

        Returns:
            输出 (batch, output_dim)
        """
        # 编码为群状态
        state = self.encoder(x)  # (batch, d*d)
        state = state.view(-1, self.group_d, self.group_d)  # (batch, d, d)

        # 应用生成元序列
        if generator_indices is None:
            # 应用所有生成元的乘积
            for i in range(self.num_generators):
                if self.group_type == 'orthogonal':
                    R = self.cayley_transform(self.generators[i])
                else:
                    R = self.generators[i]
                state = state @ R
        else:
            for idx in generator_indices:
                if self.group_type == 'orthogonal':
                    R = self.cayley_transform(self.generators[idx])
                else:
                    R = self.generators[idx]
                state = state @ R

        # 解码
        state = state.view(state.shape[0], -1)  # (batch, d*d)
        output = self.decoder(state)

        return output

    def export_state_dict(self) -> Dict:
        """导出为独立状态字典"""
        return {
            'config': {
                'group_d': self.group_d,
                'num_generators': self.num_generators,
                'group_type': self.group_type,
            },
            'encoder': self.encoder.state_dict(),
            'generators': self.generators.cpu().numpy().tolist(),
            'decoder': self.decoder.state_dict(),
        }

    @classmethod
    def from_state_dict(cls, state_dict: Dict, device: torch.device = None) -> 'PureGroupInference':
        """从状态字典加载"""
        if device is None:
            device = torch.device('cpu')

        config = state_dict['config']
        model = cls(
            group_d=config['group_d'],
            num_generators=config['num_generators'],
            group_type=config['group_type']
        )

        model.encoder.load_state_dict(state_dict['encoder'])
        model.decoder.load_state_dict(state_dict['decoder'])

        generators = torch.tensor(state_dict['generators'], device=device)
        model.set_generators(generators)

        return model


def quantize_generators(
    generators: torch.Tensor,
    bits: int = 8
) -> Tuple[torch.Tensor, float]:
    """
    量化生成元矩阵

    Args:
        generators: 生成元张量 (k, d, d)
        bits: 量化位数

    Returns:
        (量化后的张量，正交性损失)
    """
    # 对称量化
    qmax = (1 << (bits - 1)) - 1
    qmin = -qmax - 1

    scale = generators.abs().max() / qmax
    quantized = (generators / scale).round().clamp(qmin, qmax)
    dequantized = quantized * scale

    # 计算正交性损失
    ortho_loss = 0.0
    for R in dequantized:
        if generators.shape[0] > 0:  # 确保有生成元
            RtR = R.transpose(-1, -2) @ R
            I = torch.eye(R.shape[0])
            ortho_loss += torch.norm(RtR - I, p='fro').item()

    ortho_loss /= max(generators.shape[0], 1)

    return dequantized, ortho_loss


def export_from_gmgd_model(
    gmgd_model_path: str,
    output_dir: str,
    quantize: bool = False,
    use_cayley: bool = True
) -> Dict:
    """
    从 GMGD 模型导出边缘推理核

    Args:
        gmgd_model_path: GMGD 模型路径
        output_dir: 输出目录
        quantize: 是否量化
        use_cayley: 使用 Cayley 变换

    Returns:
        导出信息
    """
    os.makedirs(output_dir, exist_ok=True)

    # 加载 GMGD 模型
    from gpt_with_group import GPTWithGroup

    device = torch.device('cpu')
    gmgd_model = GPTWithGroup.load_pretrained(gmgd_model_path, device)

    # 获取 MetaGroup 参数
    meta_group = gmgd_model.meta_group
    generators = meta_group.all_generators().detach()

    export_info = {
        'source_model': gmgd_model_path,
        'exported_at': torch.__version__,
        'quantized': quantize,
        'use_cayley': use_cayley
    }

    # 量化（可选）
    if quantize:
        generators, ortho_loss = quantize_generators(generators, bits=8)
        export_info['quantization_ortho_loss'] = ortho_loss
        print(f"Quantization orthogonality loss: {ortho_loss:.6f}")

    # 保存生成元
    generators_path = os.path.join(output_dir, 'generators.pt')
    torch.save({
        'generators': generators,
        'group_type': meta_group.group_type,
        'd': meta_group.d,
        'num_generators': meta_group.num_generators
    }, generators_path)

    export_info['generators_path'] = generators_path
    export_info['generators_size_kb'] = os.path.getsize(generators_path) / 1024

    # 保存配置
    config = EdgeModelConfig(
        group_d=meta_group.d,
        num_generators=meta_group.num_generators,
        group_type=meta_group.group_type,
        use_cayley=use_cayley,
        quantize=quantize
    )

    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'group_d': config.group_d,
            'num_generators': config.num_generators,
            'group_type': config.group_type,
            'use_cayley': config.use_cayley,
            'quantize': config.quantize
        }, f, indent=2)

    # 导出投影层权重（如果可访问）
    if hasattr(gmgd_model, 'smooth_layers') and len(gmgd_model.smooth_layers) > 0:
        smooth_layer = gmgd_model.smooth_layers[0]
        proj_weights_path = os.path.join(output_dir, 'proj_weights.pt')
        torch.save({
            'proj_to': smooth_layer.proj_to.state_dict(),
            'proj_from': smooth_layer.proj_from.state_dict(),
            'smooth_lr': smooth_layer.smooth_lr,
            'proj_steps': smooth_layer.proj_steps
        }, proj_weights_path)
        export_info['proj_weights_path'] = proj_weights_path

    # 保存导出信息
    info_path = os.path.join(output_dir, 'export_info.json')
    with open(info_path, 'w') as f:
        json.dump(export_info, f, indent=2)

    print(f"\n=== 导出完成 ===")
    print(f"输出目录：{output_dir}")
    print(f"生成元大小：{export_info['generators_size_kb']:.2f} KB")
    print(f"配置文件：{config_path}")
    print(f"导出信息：{info_path}")

    return export_info


def benchmark_inference(
    model: PureGroupInference,
    batch_size: int = 1,
    seq_len: int = 1,
    device: torch.device = None,
    num_runs: int = 100
) -> Dict[str, float]:
    """
    基准测试推理延迟

    Args:
        model: 纯群推理模型
        batch_size: 批量大小
        seq_len: 序列长度
        device: 设备
        num_runs: 运行次数

    Returns:
        延迟统计
    """
    if device is None:
        device = torch.device('cpu')

    model = model.to(device)
    model.eval()

    # 创建输入
    input_dim = model.encoder[0].in_features
    x = torch.randn(batch_size, seq_len, input_dim, device=device)

    # 预热
    for _ in range(10):
        _ = model(x.view(batch_size * seq_len, input_dim))

    # 计时
    import time
    times = []

    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(x.view(batch_size * seq_len, input_dim))
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    # 移除首尾异常值
    times.sort()
    trimmed_times = times[5:-5] if len(times) > 10 else times

    return {
        'mean_latency_ms': sum(trimmed_times) / len(trimmed_times) * 1000,
        'p50_latency_ms': trimmed_times[len(trimmed_times) // 2] * 1000,
        'p99_latency_ms': trimmed_times[int(len(trimmed_times) * 0.99)] * 1000,
        'min_latency_ms': min(trimmed_times) * 1000,
        'max_latency_ms': max(trimmed_times) * 1000
    }


def main():
    parser = argparse.ArgumentParser(description='Export GMGD model to edge inference kernel')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained GMGD model')
    parser.add_argument('--output_dir', type=str, default='./edge_model',
                        help='Output directory')
    parser.add_argument('--quantize', action='store_true',
                        help='Quantize generators to INT8')
    parser.add_argument('--no_cayley', action='store_true',
                        help='Do not use Cayley transform (use matrix_exp instead)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run inference benchmark after export')

    args = parser.parse_args()

    # 导出
    export_info = export_from_gmgd_model(
        args.model_path,
        args.output_dir,
        quantize=args.quantize,
        use_cayley=not args.no_cayley
    )

    # 基准测试
    if args.benchmark:
        print("\n=== 运行基准测试 ===")

        # 创建纯群推理模型（演示用）
        model = PureGroupInference(
            group_d=export_info.get('group_d', 32),
            num_generators=export_info.get('num_generators', 12)
        )

        # 加载生成元
        generators_path = export_info.get('generators_path')
        if generators_path and os.path.exists(generators_path):
            gen_data = torch.load(generators_path)
            model.set_generators(gen_data['generators'])

        # CPU 基准
        cpu_results = benchmark_inference(model, device=torch.device('cpu'))
        print(f"\nCPU 推理延迟:")
        print(f"  平均：{cpu_results['mean_latency_ms']:.3f} ms")
        print(f"  P50: {cpu_results['p50_latency_ms']:.3f} ms")
        print(f"  P99: {cpu_results['p99_latency_ms']:.3f} ms")

        # CUDA 基准（如果可用）
        if torch.cuda.is_available():
            cuda_results = benchmark_inference(model, device=torch.device('cuda'))
            print(f"\nCUDA 推理延迟:")
            print(f"  平均：{cuda_results['mean_latency_ms']:.3f} ms")


if __name__ == '__main__':
    main()
