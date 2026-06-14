#!/usr/bin/env bash
# Streamlit 启动脚本：确保从任何目录运行都能导入 car_pricing
# 使用方式:
#   1. 安装模式 (推荐): pip install -e .. && ./run.sh
#   2. 开发模式: ./run.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 如果 car_pricing 未安装，使用 PYTHONPATH 指向项目根目录
if ! python -c "import car_pricing" 2>/dev/null; then
    echo "[INFO] car_pricing not installed as package, using PYTHONPATH=$PROJECT_ROOT"
    export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
fi

# 加载环境变量
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "[INFO] Loading environment from $PROJECT_ROOT/.env"
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# 默认 API 配置
export API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"
export API_REQUEST_TIMEOUT="${API_REQUEST_TIMEOUT:-10}"

echo "[INFO] Starting Streamlit with API_BASE_URL=$API_BASE_URL"
cd "$SCRIPT_DIR"
streamlit run streamlit_app.py "$@"
