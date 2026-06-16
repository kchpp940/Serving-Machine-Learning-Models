# === 统一启动命令 (使用 car_pricing.config 环境变量) ===
# 所有环境变量与 car_pricing/config.py 变量名完全对齐
# Heroku 会自动注入 PORT 环境变量，我们将其映射到统一的 API_PORT
# === PROCFILE_EXPORT_GENERATED_START ===
web: export API_HOST=${API_HOST:-0.0.0.0} && export API_PORT=${PORT:-${API_PORT:-8000}} && export MODEL_DIR=${MODEL_DIR:-/app/shared_models} && export MODEL_FILENAME=${MODEL_FILENAME:-sklearn_gbr.pkl} && export MODEL_METADATA_FILENAME=${MODEL_METADATA_FILENAME:-model_metadata.json} && export MODEL_STATUS_FILENAME=${MODEL_STATUS_FILENAME:-model_status.json} && export DATA_CSV_FILENAME=${DATA_CSV_FILENAME:-cars.csv} && export DATA_DIR=${DATA_DIR:-/app/Data} && export API_BASE_URL=${API_BASE_URL:-http://localhost:${API_PORT}} && export REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-10} && export BENTOML_MODEL_TAG=${BENTOML_MODEL_TAG:-gbr:latest} && \
# === PROCFILE_EXPORT_GENERATED_END ===
     export PYTHONPATH=/app:/app/car_pricing_api:$PYTHONPATH && \
     cd car_pricing_api && \
     gunicorn -w 3 -k uvicorn.workers.UvicornWorker app:app \
     --bind ${API_HOST}:${API_PORT}
