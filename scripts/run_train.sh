#!/bin/bash
# GMGD 群扩展 Transformer 训练脚本

set -e

# 默认参数
MODEL="gpt2"
EPOCHS=3
BATCH_SIZE=16
OUTPUT_DIR="./checkpoints"
DEVICE="cpu"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
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
        --help)
            echo "用法：bash run_train.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --model       基础模型名称 (默认：gpt2)"
            echo "  --epochs      训练轮数 (默认：3)"
            echo "  --batch_size  批量大小 (默认：16)"
            echo "  --output_dir  输出目录 (默认：./checkpoints)"
            echo "  --device      设备类型 (默认：cpu)"
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

echo "=== GMGD 群扩展 Transformer 训练 ==="
echo "模型：$MODEL"
echo "轮数：$EPOCHS"
echo "批量大小：$BATCH_SIZE"
echo "输出目录：$OUTPUT_DIR"
echo "设备：$DEVICE"
echo ""

# 运行训练
python train.py \
    --base_model "$MODEL" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE"

echo ""
echo "=== 训练完成 ==="
echo "模型已保存至：$OUTPUT_DIR"
