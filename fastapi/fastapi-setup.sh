#!/bin/bash
set -e

# Update and install requirements
sudo apt-get update
sudo apt install -y python3-pip nginx
sudo apt install -y uvicorn
# Copy the configuration file to the nginx enabled sites folder
sudo cp -R fastapi_setup /etc/nginx/sites-enabled/
sudo service nginx restart

# 切换到项目根目录进行统一安装
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# 统一安装模式：
#   1. 先安装第三方依赖（从生成的 requirements.txt）
#   2. 再以 --no-deps 安装 car_pricing 包本身，确保导入路径稳定
pip3 install -r "$(dirname "$0")/requirements.txt"
pip3 install --no-deps -e "$PROJECT_ROOT"

# Kill any service running on port 80
sudo kill -9 $(sudo lsof -t -i:80) || true
sudo service nginx restart

# Run the application with nohup so the application runs as a background process
cd "$(dirname "$0")"
nohup python3 -m uvicorn app:app --reload --host 0.0.0.0
