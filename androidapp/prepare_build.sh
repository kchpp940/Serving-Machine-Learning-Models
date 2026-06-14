#!/usr/bin/env bash
# Android/Buildozer 构建预处理脚本
# 作用：将 car_pricing 共享模块复制到 androidapp/ 目录下，确保打包时被包含
# 使用方式: ./prepare_build.sh && buildozer android debug

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET_DIR="$SCRIPT_DIR/car_pricing"
SOURCE_DIR="$PROJECT_ROOT/car_pricing"

echo "[INFO] Project root: $PROJECT_ROOT"
echo "[INFO] Copying car_pricing module to androidapp/"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "[ERROR] car_pricing source directory not found at $SOURCE_DIR"
    exit 1
fi

# 清理旧的复制
rm -rf "$TARGET_DIR"

# 复制 car_pricing 包（排除测试文件和缓存）
mkdir -p "$TARGET_DIR"
rsync -a \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='tests' \
    --exclude='.pytest_cache' \
    "$SOURCE_DIR/" "$TARGET_DIR/"

# 验证复制成功
if [ -f "$TARGET_DIR/api_client.py" ] && [ -f "$TARGET_DIR/model_runtime.py" ]; then
    echo "[OK] car_pricing module copied successfully to $TARGET_DIR"
    echo "     Files:"
    find "$TARGET_DIR" -name "*.py" | sed 's/^/       - /'
else
    echo "[ERROR] Failed to copy car_pricing module correctly"
    exit 1
fi

# 验证关键常量定义
if grep -q 'DEFAULT_API_BASE_URL' "$TARGET_DIR/api_client.py"; then
    echo "[OK] api_client.py contains DEFAULT_API_BASE_URL"
else
    echo "[ERROR] api_client.py missing DEFAULT_API_BASE_URL constant"
    exit 1
fi

echo ""
echo "[INFO] Preparation complete. You can now run:"
echo "       buildozer android debug"
echo "       OR"
echo "       buildozer android release"
