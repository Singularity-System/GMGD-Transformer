#!/bin/bash
# GMGD 项目清理脚本

echo "清理临时文件和日志..."

# 删除 __pycache__ 目录
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
echo "✓ 清理 __pycache__"

# 删除 .pyc 文件
find . -name "*.pyc" -delete 2>/dev/null
echo "✓ 清理 .pyc 文件"

# 删除日志文件
rm -f *.log 2>/dev/null
echo "✓ 清理日志文件"

# 删除临时测试文件（保留核心测试）
rm -f test_minimal.py test_eval_logic.py test_manifold_loss.py 2>/dev/null
echo "✓ 清理临时测试文件"

# 删除旧的重命名文件
rm -f *_fixed.py *_backup.py *_old.py 2>/dev/null
echo "✓ 清理备份文件"

echo ""
echo "清理完成！"
echo ""
echo "保留的重要文件:"
echo "  - core/: 核心模块"
echo "  - data/: 数据生成器"
echo "  - configs/: 配置文件"
echo "  - scripts/: 工具脚本"
echo "  - checkpoints/: 模型检查点"
echo "  - length_extrapolation_results/: 实验结果"
