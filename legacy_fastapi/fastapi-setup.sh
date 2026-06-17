#!/bin/bash
set -e

# Update and install requirements
sudo apt-get update
sudo apt install -y python3-pip nginx
sudo apt install -y uvicorn
# Copy the configuration file to the nginx enabled sites folder
sudo cp -R fastapi_setup /etc/nginx/sites-enabled/
sudo service nginx restart

# 切换到项目根目录安装依赖
cd "$(dirname "$0")/.."
# Install core package + API dependencies (editable install)
pip3 install -e ".[api]"

# Kill any service running on port 80
sudo kill -9 $(sudo lsof -t -i:80) || true
sudo service nginx restart

# Run the application with nohup so the application runs as a background process
cd legacy_fastapi
nohup python3 -m uvicorn app:app --reload --host 0.0.0.0
