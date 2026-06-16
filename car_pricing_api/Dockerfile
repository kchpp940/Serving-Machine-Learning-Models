FROM python:3.8-slim-buster

# ============================================
# 重要：构建命令必须从项目根目录执行！
#   docker build -f car_pricing_api/Dockerfile .
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
COPY car_pricing_api/requirements.txt /app/car_pricing_api/
RUN pip install -r /app/car_pricing_api/requirements.txt

COPY car_pricing/ /app/car_pricing/
COPY car_pricing_api/ /app/car_pricing_api/
COPY Data/ /app/Data/
COPY shared_models/ /app/shared_models/

WORKDIR /app/car_pricing_api

ENV PYTHONPATH=/app:/app/car_pricing_api

# === 统一运行时配置 (与 car_pricing/config.py 变量名完全对齐) ===
# 这些是默认值，部署时可通过 docker run -e 或平台环境变量覆盖
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV MODEL_DIR=/app/shared_models
ENV MODEL_FILENAME=sklearn_gbr.pkl
ENV MODEL_METADATA_FILENAME=model_metadata.json
ENV MODEL_STATUS_FILENAME=model_status.json
ENV DATA_DIR=/app/Data
ENV DATA_CSV_FILENAME=cars.csv
ENV API_BASE_URL=http://localhost:${API_PORT}
ENV REQUEST_TIMEOUT=10
ENV BENTOML_MODEL_TAG=gbr:latest

# === 启动脚本：将 Heroku 的 PORT 映射到 API_PORT，然后启动 ===
# 这样 car_pricing.config 能正确读取统一的 API_PORT 变量
CMD export API_PORT=${PORT:-${API_PORT}} && \
    export API_BASE_URL="http://localhost:${API_PORT}" && \
    gunicorn -w 3 -k uvicorn.workers.UvicornWorker app:app \
    --bind ${API_HOST}:${API_PORT}
