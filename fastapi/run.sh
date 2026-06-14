#!/usr/bin/env bash
# FastAPI 服务端启动脚本
# 使用方式:
#   1. 安装模式 (推荐): pip install -e .. && ./run.sh
#   2. 开发模式: ./run.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if ! python -c "import car_pricing" 2>/dev/null; then
    echo "[INFO] car_pricing not installed as package, using PYTHONPATH=$PROJECT_ROOT"
    export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
fi

if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "[INFO] Loading environment from $PROJECT_ROOT/.env"
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

export PORT="${PORT:-8000}"
export HOST="${HOST:-0.0.0.0}"

echo "[INFO] Starting FastAPI server on $HOST:$PORT"
cd "$SCRIPT_DIR"
uvicorn app:app --host "$HOST" --port "$PORT" --reload "$@"
