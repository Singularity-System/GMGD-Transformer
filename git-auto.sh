#!/bin/bash
# Git 自动提交服务管理器

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTO_COMMIT_SCRIPT="$SCRIPT_DIR/.git_auto_commit.sh"
PID_FILE="$SCRIPT_DIR/.git_auto_commit.pid"

case "$1" in
    start)
        if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
            echo "自动提交服务已在运行 (PID: $(cat "$PID_FILE"))"
        else
            echo "启动 Git 自动提交服务..."
            nohup bash "$AUTO_COMMIT_SCRIPT" > /dev/null 2>&1 &
            echo $! > "$PID_FILE"
            echo "服务已启动 (PID: $(cat "$PID_FILE"))"
            echo "日志文件：$SCRIPT_DIR/.git_auto_commit.log"
        fi
        ;;
    stop)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if kill -0 "$PID" 2>/dev/null; then
                kill "$PID"
                rm "$PID_FILE"
                echo "服务已停止 (PID: $PID)"
            else
                rm "$PID_FILE"
                echo "服务未运行，清理 PID 文件"
            fi
        else
            echo "服务未运行 (无 PID 文件)"
        fi
        ;;
    status)
        if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
            echo "✓ 服务运行中 (PID: $(cat "$PID_FILE"))"
        else
            echo "✗ 服务未运行"
        fi
        ;;
    log)
        if [ -f "$SCRIPT_DIR/.git_auto_commit.log" ]; then
            tail -50 "$SCRIPT_DIR/.git_auto_commit.log"
        else
            echo "暂无日志"
        fi
        ;;
    restart)
        $0 stop
        sleep 1
        $0 start
        ;;
    *)
        echo "用法：$0 {start|stop|status|log|restart}"
        echo ""
        echo "命令说明:"
        echo "  start   - 启动自动提交服务"
        echo "  stop    - 停止自动提交服务"
        echo "  status  - 查看服务状态"
        echo "  log     - 查看最近 50 行日志"
        echo "  restart - 重启服务"
        ;;
esac
