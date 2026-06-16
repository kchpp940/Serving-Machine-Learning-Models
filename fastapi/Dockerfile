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

ENV PYTHONPATH=/app:/app/..

# Runtime configuration (defaults defined in car_pricing/config.py):
#   API_HOST=0.0.0.0
#   API_PORT=8000
#   MODEL_DIR=/app/models
#   API_BASE_URL=http://localhost:$PORT
#   REQUEST_TIMEOUT=10
# See ../car_pricing/config.py for the unified runtime config

CMD gunicorn -w 3 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:$PORT