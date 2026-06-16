FROM python:3.8-slim-buster

WORKDIR /app

RUN apt-get -y update  && apt-get install -y \
  python3-dev \
  apt-utils \
  python-dev \
  build-essential \
&& rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade setuptools 
    

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD gunicorn -w 3 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:$PORT

# BEGIN AUTO-GENERATED ENV VARS — 不要手动修改此段，由 scripts/generate_config_artifacts.py 管理
# 环境变量默认值（单一真相: export_env_schema().deployment_defaults['Docker/Heroku']）
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
