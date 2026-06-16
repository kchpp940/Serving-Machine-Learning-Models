FROM python:3.8-slim-buster

# ============================================
# 重要：构建命令必须从项目根目录执行！
#   docker build -f fastapi/Dockerfile .
# ============================================

WORKDIR /app

RUN apt-get -y update  && apt-get install -y \
  python3-dev \
  apt-utils \
  python-dev \
  build-essential \
&& rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade setuptools 

# === 复制依赖和代码 ===
COPY fastapi/requirements.txt /app/fastapi/
RUN pip install -r /app/fastapi/requirements.txt

COPY car_pricing/ /app/car_pricing/
COPY fastapi/ /app/fastapi/
COPY Data/ /app/Data/
COPY shared_models/ /app/shared_models/

WORKDIR /app/fastapi

ENV PYTHONPATH=/app:/app/fastapi

# === DOCKER_ENV_GENERATED_START ===
# 由 scripts/generate_config_artifacts.py 从 car_pricing.config.export_env_schema() 自动生成
# 请勿手动编辑此段，修改后运行生成脚本覆盖
# 共 11 个环境变量（部署目标: docker）

# === API server configuration ===
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
# === Model file locations ===
ENV MODEL_DIR=/app/shared_models
ENV MODEL_FILENAME=sklearn_gbr.pkl
ENV MODEL_METADATA_FILENAME=model_metadata.json
ENV MODEL_STATUS_FILENAME=model_status.json
# === Training data locations ===
ENV DATA_CSV_FILENAME=cars.csv
ENV DATA_DIR=/app/Data
# === Client configuration ===
ENV API_BASE_URL=http://localhost:${API_PORT}
ENV REQUEST_TIMEOUT=10
# === BentoML configuration ===
ENV BENTOML_MODEL_TAG=gbr:latest
# === DOCKER_ENV_GENERATED_END ===

# === 启动脚本：将 Heroku 的 PORT 映射到 API_PORT，然后启动 ===
# 这样 car_pricing.config 能正确读取统一的 API_PORT 变量
CMD export API_PORT=${PORT:-${API_PORT}} && \
    export API_BASE_URL="http://localhost:${API_PORT}" && \
    gunicorn -w 3 -k uvicorn.workers.UvicornWorker app:app \
    --bind ${API_HOST}:${API_PORT}
