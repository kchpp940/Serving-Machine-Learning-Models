FROM python:3.10-slim

WORKDIR /app

# BEGIN AUTO-GENERATED ENV VARS — 不要手动修改此段，由 scripts/generate_config_artifacts.py 管理
# 环境变量默认值（Docker 构建与运行阶段均可用，来源: export_env_schema）
ENV API_HOST="0.0.0.0"
ENV API_PORT="8000"
ENV MODEL_DIR="/app/shared_models"
ENV MODEL_FILENAME="sklearn_gbr.pkl"
ENV MODEL_METADATA_FILENAME="model_metadata.json"
ENV MODEL_STATUS_FILENAME="model_status.json"
ENV SCHEMA_SNAPSHOT_FILENAME="schema_snapshot.json"
ENV LINEAGE_FILENAME="model_lineage.json"
ENV DATA_CSV_FILENAME="cars.csv"
ENV DATA_DIR="/app/Data"
ENV API_BASE_URL="http://localhost:${API_PORT}"
ENV REQUEST_TIMEOUT="10"
ENV BENTOML_MODEL_TAG="gbr:latest"
# END AUTO-GENERATED ENV VARS

RUN apt-get -y update  && apt-get install -y \
  python3-dev \
  apt-utils \
  build-essential \
&& rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade setuptools pip wheel

# 复制根目录的 pyproject.toml 和核心包源码
COPY pyproject.toml /app/
COPY car_pricing /app/car_pricing/

# 复制 API 目录文件
COPY car_pricing_api /app/car_pricing_api/

# 安装核心包 + API 依赖（可编辑安装以便导入）
RUN pip install -e ".[api]"

# 设置工作目录到 API 目录
WORKDIR /app/car_pricing_api

CMD gunicorn -w 3 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:$PORT
