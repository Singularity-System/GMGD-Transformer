#!/bin/bash
# Git 自动提交脚本
# 功能：检测文件变化并自动提交到本地 Git 仓库

set -e

# 配置
COMMIT_PREFIX="auto:"
CHECK_INTERVAL=600  # 检测间隔（秒）
LOG_FILE=".git_auto_commit.log"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 检查是否在 git 仓库中
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    log "${RED}错误：当前目录不是 Git 仓库${NC}"
    exit 1
fi

# 检查是否有远程仓库
REMOTE_EXISTS=$(git remote | wc -l)
if [ "$REMOTE_EXISTS" -eq 0 ]; then
    log "${YELLOW}提示：没有配置远程仓库，仅本地提交${NC}"
else
    log "${GREEN}检测到远程仓库，将同步推送${NC}"
fi

log "${GREEN}=== Git 自动提交服务已启动 ===${NC}"
log "检测间隔：${CHECK_INTERVAL}秒"
log "按 Ctrl+C 停止服务"

# 主循环
while true; do
    # 检查是否有未提交的更改
    if ! git diff --quiet || ! git diff --cached --quiet; then
        # 获取更改的文件列表
        CHANGED_FILES=$(git diff --name-only 2>/dev/null || true)
        STAGED_FILES=$(git diff --cached --name-only 2>/dev/null || true)

        if [ -n "$CHANGED_FILES" ] || [ -n "$STAGED_FILES" ]; then
            log "${YELLOW}检测到文件变化:${NC}"
            echo "$CHANGED_FILES $STAGED_FILES" | tr ' ' '\n' | sort -u | head -10

            # 添加所有更改
            git add -A

            # 生成提交信息
            FILE_COUNT=$(echo "$CHANGED_FILES $STAGED_FILES" | tr ' ' '\n' | sort -u | wc -l)
            COMMIT_MSG="${COMMIT_PREFIX} 自动提交 ${FILE_COUNT} 个文件 at $(date '+%H:%M:%S')"

            # 提交
            git commit -m "$COMMIT_MSG" 2>/dev/null && log "${GREEN}提交成功：${COMMIT_MSG}${NC}" || log "${RED}提交失败（可能没有实际变化）${NC}"

            # 如果有远程仓库，尝试推送
            if [ "$REMOTE_EXISTS" -gt 0 ]; then
                git push 2>/dev/null && log "${GREEN}推送成功${NC}" || log "${YELLOW}推送失败（可能没有网络连接）${NC}"
            fi
        fi
    fi

    # 等待下一次检测
    sleep "$CHECK_INTERVAL"
done
