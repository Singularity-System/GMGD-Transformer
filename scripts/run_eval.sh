#!/bin/bash
# GMGD 群扩展 Transformer 评估脚本

set -e

# 默认参数
MODEL_PATH="./checkpoints/best_model"
OUTPUT_DIR="./eval_results"
DEVICE="cpu"
MODE="full"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --help)
            echo "用法：bash run_eval.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --model_path  模型路径 (默认：./checkpoints/best_model)"
            echo "  --output_dir  输出目录 (默认：./eval_results)"
            echo "  --device      设备类型 (默认：cpu)"
            echo "  --mode        评估模式：full/length/transitivity/ablation (默认：full)"
            echo "  --help        显示帮助"
            exit 0
            ;;
        *)
            echo "未知选项：$1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

echo "=== GMGD 群扩展 Transformer 评估 ==="
echo "模型路径：$MODEL_PATH"
echo "输出目录：$OUTPUT_DIR"
echo "设备：$DEVICE"
echo "评估模式：$MODE"
echo ""

# 运行评估
python evaluate.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --mode "$MODE"

echo ""
echo "=== 评估完成 ==="
echo "结果已保存至：$OUTPUT_DIR"
