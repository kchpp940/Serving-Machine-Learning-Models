#!/bin/bash
# === 统一 FastAPI 部署脚本 (使用 car_pricing.config 环境变量约定) ===

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# === 统一运行时配置 (与 car_pricing/config.py 变量名完全对齐) ===
export API_HOST="${API_HOST:-0.0.0.0}"
export API_PORT="${API_PORT:-8000}"
export MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/shared_models}"
export MODEL_FILENAME="${MODEL_FILENAME:-sklearn_gbr.pkl}"
export MODEL_METADATA_FILENAME="${MODEL_METADATA_FILENAME:-model_metadata.json}"
export MODEL_STATUS_FILENAME="${MODEL_STATUS_FILENAME:-model_status.json}"
export DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/Data}"
export DATA_CSV_FILENAME="${DATA_CSV_FILENAME:-cars.csv}"
export API_BASE_URL="${API_BASE_URL:-http://localhost:$API_PORT}"
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-10}"
export BENTOML_MODEL_TAG="${BENTOML_MODEL_TAG:-gbr:latest}"
export PYTHONPATH="$PROJECT_ROOT:$SCRIPT_DIR:$PYTHONPATH"

echo "=== Car Pricing API 启动配置 ==="
echo "API_HOST: $API_HOST"
echo "API_PORT: $API_PORT"
echo "MODEL_DIR: $MODEL_DIR"
echo "API_BASE_URL: $API_BASE_URL"
echo "PYTHONPATH: $PYTHONPATH"
echo ""

cd "$SCRIPT_DIR"

if [ "$1" = "train" ]; then
    echo "开始训练模型..."
    python train.py
elif [ "$1" = "dev" ]; then
    echo "启动开发服务器 (uvicorn --reload)..."
    python -m uvicorn app:app --host "$API_HOST" --port "$API_PORT" --reload
else
    echo "启动生产服务器 (gunicorn)..."
    gunicorn -w 3 -k uvicorn.workers.UvicornWorker app:app \
        --bind "$API_HOST:$API_PORT"
fi
