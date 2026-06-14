@echo off
REM Streamlit 启动脚本（Windows）
REM 使用方式:
REM   1. 安装模式 (推荐): pip install -e .. && run.bat
REM   2. 开发模式: run.bat

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
for %%i in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fi"

REM 检查 car_pricing 是否已安装
python -c "import car_pricing" 2>nul
if errorlevel 1 (
    echo [INFO] car_pricing not installed as package, using PYTHONPATH=%PROJECT_ROOT%
    set "PYTHONPATH=%PROJECT_ROOT%;%PYTHONPATH%"
)

REM 加载 .env 文件（如果存在）
if exist "%PROJECT_ROOT%\.env" (
    echo [INFO] Loading environment from %PROJECT_ROOT%\.env
    for /f "usebackq tokens=1,2 delims==" %%a in ("%PROJECT_ROOT%\.env") do (
        if not "%%a"=="" if "!%%a!"=="" set "%%a=%%b"
    )
)

REM 默认 API 配置
if not defined API_BASE_URL set "API_BASE_URL=http://localhost:8000"
if not defined API_REQUEST_TIMEOUT set "API_REQUEST_TIMEOUT=10"

echo [INFO] Starting Streamlit with API_BASE_URL=%API_BASE_URL%
cd /d "%SCRIPT_DIR%"
streamlit run streamlit_app.py %*

endlocal
