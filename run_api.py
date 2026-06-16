#!/usr/bin/env python
"""
统一启动入口脚本，用于启动 Car Price Prediction API。

使用方式:
    python run_api.py
"""

import os
import sys

import uvicorn

from car_pricing_api import create_app


def main():
    model_dir = os.path.join(os.path.dirname(__file__), "car_pricing_api", "models")
    app = create_app(model_dir=model_dir)

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    print(f"Starting Car Price Prediction API on {host}:{port}")
    print(f"Model directory: {model_dir}")
    print(f"Swagger UI: http://{host}:{port}/docs")
    print(f"ReDoc: http://{host}:{port}/redoc")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
