#!/usr/bin/env python3
"""
验证 car_pricing 包可以从任意目录导入，确保所有组件可正常工作。
运行: python verify_imports.py
"""
import sys
import os
import subprocess
import tempfile


TESTS = []


def test(name):
    def decorator(fn):
        TESTS.append((name, fn))
        return fn
    return decorator


def run_python_in_dir(cwd, code):
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


@test("Import car_pricing from /tmp")
def _():
    code = """
from car_pricing.api_client import PredictionAPIClient, DEFAULT_API_BASE_URL, DEFAULT_TIMEOUT, PREDICT_PATH, ENV_API_BASE_URL
print('DEFAULT_API_BASE_URL:', DEFAULT_API_BASE_URL)
print('DEFAULT_TIMEOUT:', DEFAULT_TIMEOUT)
print('PREDICT_PATH:', PREDICT_PATH)
c = PredictionAPIClient(base_url='http://example.com', timeout=5)
print('client.base_url:', c.base_url)
print('client.timeout:', c.timeout)
"""
    code, out, err = run_python_in_dir("/tmp", code)
    assert code == 0, f"stdout: {out}\nstderr: {err}"
    assert "http://example.com" in out
    assert "PREDICT_PATH: /predict" in out
    return True


@test("Import from streamlitapp/ subdirectory")
def _():
    code = """
from car_pricing.api_client import PredictionAPIClient, PredictionResult
c = PredictionAPIClient(base_url='http://test:9000', timeout=5)
print('OK:', c.base_url, c.timeout)
"""
    streamlit_dir = os.path.join(os.path.dirname(__file__), "streamlitapp")
    code, out, err = run_python_in_dir(streamlit_dir, code)
    assert code == 0, f"stdout: {out}\nstderr: {err}"
    assert "http://test:9000" in out
    return True


@test("Import from androidapp/ subdirectory")
def _():
    code = """
from car_pricing.api_client import PredictionAPIClient
from car_pricing import CarPriceModel, PredictionResult
c = PredictionAPIClient()
print('OK default:', c.base_url, c.timeout)
"""
    android_dir = os.path.join(os.path.dirname(__file__), "androidapp")
    code, out, err = run_python_in_dir(android_dir, code)
    assert code == 0, f"stdout: {out}\nstderr: {err}"
    assert "http://localhost:8000" in out
    return True


@test("Import from fastapi/ subdirectory")
def _():
    code = """
from car_pricing.api_client import PredictionAPIClient
from car_pricing import CarPriceModel
print('OK')
"""
    fastapi_dir = os.path.join(os.path.dirname(__file__), "fastapi")
    code, out, err = run_python_in_dir(fastapi_dir, code)
    assert code == 0, f"stdout: {out}\nstderr: {err}"
    return True


@test("Streamlit app fallback import logic")
def _():
    streamlit_app_py = os.path.join(os.path.dirname(__file__), "streamlitapp", "streamlit_app.py")
    with open(streamlit_app_py) as f:
        content = f.read()
    assert "from car_pricing.api_client import" in content
    assert "try:" in content
    return True


@test("Android app fallback import logic")
def _():
    main_py = os.path.join(os.path.dirname(__file__), "androidapp", "main.py")
    with open(main_py) as f:
        content = f.read()
    assert "from car_pricing.api_client import" in content
    assert "try:" in content
    return True


@test("CLI entry point exists")
def _():
    code = """
from car_pricing.cli import main
print('CLI main imported OK')
"""
    code, out, err = run_python_in_dir(os.path.dirname(__file__), code)
    assert code == 0, f"stdout: {out}\nstderr: {err}"
    return True


@test("Android prepare_build.sh script exists")
def _():
    script = os.path.join(os.path.dirname(__file__), "androidapp", "prepare_build.sh")
    assert os.path.isfile(script), "prepare_build.sh not found"
    assert os.access(script, os.X_OK), "prepare_build.sh not executable"
    with open(script) as f:
        content = f.read()
    assert "car_pricing" in content
    assert "copying" in content.lower()
    return True


@test("Buildozer.spec includes car_pricing pattern")
def _():
    spec = os.path.join(os.path.dirname(__file__), "androidapp", "buildozer.spec")
    with open(spec) as f:
        content = f.read()
    assert "car_pricing" in content
    assert "source.include_patterns" in content
    return True


@test("Streamlit run.sh script exists")
def _():
    script = os.path.join(os.path.dirname(__file__), "streamlitapp", "run.sh")
    assert os.path.isfile(script)
    assert os.access(script, os.X_OK)
    with open(script) as f:
        content = f.read()
    assert "car_pricing" in content
    assert "PYTHONPATH" in content
    return True


@test("FastAPI run.sh script exists")
def _():
    script = os.path.join(os.path.dirname(__file__), "fastapi", "run.sh")
    assert os.path.isfile(script)
    assert os.access(script, os.X_OK)
    with open(script) as f:
        content = f.read()
    assert "car_pricing" in content
    return True


@test(".env.example documents all variables")
def _():
    env_file = os.path.join(os.path.dirname(__file__), ".env.example")
    with open(env_file) as f:
        content = f.read()
    assert "API_BASE_URL" in content
    assert "API_REQUEST_TIMEOUT" in content
    assert "ANDROID_API_BASE_URL" in content
    assert "HOST" in content
    assert "PORT" in content
    return True


@test(".gitignore ignores androidapp/car_pricing/")
def _():
    gitignore = os.path.join(os.path.dirname(__file__), ".gitignore")
    with open(gitignore) as f:
        content = f.read()
    assert "/androidapp/car_pricing/" in content
    return True


@test("pyproject.toml defines car-pricing package")
def _():
    toml_file = os.path.join(os.path.dirname(__file__), "pyproject.toml")
    with open(toml_file) as f:
        content = f.read()
    assert 'name = "car-pricing"' in content
    assert "[project.optional-dependencies]" in content
    assert "fastapi" in content
    assert "streamlit" in content
    assert "android" in content
    return True


@test("car_pricing __init__.py exports api_client symbols")
def _():
    code = """
from car_pricing import (
    PredictionAPIClient, PredictionResult, HealthResult, SchemaResult,
    APIClientError, APIConnectionError, APITimeoutError, APIHTTPError, APIInvalidResponseError,
    ENV_API_BASE_URL, ENV_API_TIMEOUT, DEFAULT_API_BASE_URL, DEFAULT_TIMEOUT,
)
print('All symbols exported OK')
"""
    code, out, err = run_python_in_dir(os.path.dirname(__file__), code)
    assert code == 0, f"stdout: {out}\nstderr: {err}"
    return True


def main():
    print("=" * 60)
    print("Verifying car_pricing package deployment readiness")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    for name, fn in TESTS:
        try:
            result = fn()
            if result:
                print(f"  ✓ {name}")
                passed += 1
            else:
                print(f"  ✗ {name} (returned falsy)")
                failed += 1
        except Exception as e:
            print(f"  ✗ {name}")
            print(f"    {type(e).__name__}: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Result: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        print()
        print("All deployment checks passed!")
        print()
        print("Quick start:")
        print("  pip install -e .                          # Install package")
        print("  cp .env.example .env && vim .env          # Configure")
        print("  cd fastapi && ./run.sh                    # Start API server")
        print("  cd streamlitapp && ./run.sh               # Start web client")
        print("  cd androidapp && ./prepare_build.sh       # Prep Android build")
        sys.exit(0)


if __name__ == "__main__":
    main()
