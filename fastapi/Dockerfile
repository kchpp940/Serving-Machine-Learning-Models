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

# 步骤 1：复制项目根配置和 car_pricing 核心包（供可编辑安装使用）
COPY pyproject.toml /app/
COPY car_pricing /app/car_pricing/

# 步骤 2：复制部署目录
COPY fastapi /app/fastapi/

# 步骤 3：统一安装模式
#   - 先安装第三方依赖（从生成的 requirements.txt）
#   - 再以 --no-deps 安装 car_pricing 包本身，确保导入路径稳定
WORKDIR /app/fastapi
RUN pip install -r requirements.txt \
 && pip install --no-deps -e /app

CMD gunicorn -w 3 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:$PORT
