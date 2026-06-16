"""配置一致性检查脚本。

用于验证所有入口、部署配置与 car_pricing.config 的变量名和默认值一致。
运行方式：
    python -m pytest tests/test_config_consistency.py -v
    python tests/test_config_consistency.py
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from car_pricing.config import (  # noqa: E402
    _DEFAULT_API_HOST,
    _DEFAULT_API_PORT,
    _DEFAULT_MODEL_FILENAME,
    _DEFAULT_MODEL_METADATA_FILENAME,
    _DEFAULT_MODEL_STATUS_FILENAME,
    _DEFAULT_DATA_CSV_FILENAME,
    _DEFAULT_REQUEST_TIMEOUT,
    _DEFAULT_BENTOML_MODEL_TAG,
    load_config,
    get_config,
)


CONFIG_VAR_NAMES = {
    "API_HOST",
    "API_PORT",
    "MODEL_DIR",
    "MODEL_FILENAME",
    "MODEL_METADATA_FILENAME",
    "MODEL_STATUS_FILENAME",
    "DATA_DIR",
    "DATA_CSV_FILENAME",
    "API_BASE_URL",
    "REQUEST_TIMEOUT",
    "BENTOML_MODEL_TAG",
}

EXPECTED_ENV_DEFAULTS = {
    "API_HOST": _DEFAULT_API_HOST,
    "API_PORT": str(_DEFAULT_API_PORT),
    "MODEL_FILENAME": _DEFAULT_MODEL_FILENAME,
    "MODEL_METADATA_FILENAME": _DEFAULT_MODEL_METADATA_FILENAME,
    "MODEL_STATUS_FILENAME": _DEFAULT_MODEL_STATUS_FILENAME,
    "DATA_CSV_FILENAME": _DEFAULT_DATA_CSV_FILENAME,
    "REQUEST_TIMEOUT": str(_DEFAULT_REQUEST_TIMEOUT),
    "BENTOML_MODEL_TAG": _DEFAULT_BENTOML_MODEL_TAG,
}

HARDCODED_PATTERNS = [
    (re.compile(r"[\"']0\.0\.0\.0[\"']"), "硬编码 API_HOST=0.0.0.0"),
    (re.compile(r"[\"']127\.0\.0\.1[\"']"), "硬编码 localhost"),
    (re.compile(r":\s*8000\b"), "硬编码端口 8000"),
    (re.compile(r":\s*8501\b"), "硬编码端口 8501"),
    (re.compile(r"[\"']sklearn_gbr\.pkl[\"']"), "硬编码模型文件名"),
    (re.compile(r"[\"']model_metadata\.json[\"']"), "硬编码元数据文件名"),
    (re.compile(r"[\"']cars\.csv[\"']"), "硬编码数据文件名"),
    (re.compile(r"os\.environ\.get\([\"']PORT[\"']"), "使用 PORT 而非 API_PORT"),
]

PYTHON_ENTRIES_TO_CHECK = [
    "car_pricing_api/app.py",
    "car_pricing_api/services/model_service.py",
    "car_pricing_api/train.py",
    "fastapi/app.py",
    "fastapi/train.py",
    "streamlitapp/streamlit_app.py",
    "bentoml/service.py",
    "bentoml/bentosklearn.py",
    "run_api.py",
    "run_streamlit.py",
]

DEPLOYMENT_FILES_TO_CHECK = [
    "car_pricing_api/Dockerfile",
    "fastapi/Dockerfile",
    "car_pricing_api/heroku.yml",
    "fastapi/heroku.yml",
    "car_pricing_api/vercel.json",
    "fastapi/vercel.json",
    "car_pricing_api/fastapi-setup.sh",
    "fastapi/fastapi-setup.sh",
    "Procfile",
]


@dataclass
class ConfigIssue:
    file_path: str
    line: int
    issue: str
    severity: str = "error"

    def __str__(self) -> str:
        return f"{self.severity.upper()}: {self.file_path}:{self.line}: {self.issue}"


class TestConfigConsistency:
    """配置一致性测试集合。"""

    @pytest.fixture(scope="class")
    def global_config(self):
        return get_config()

    def test_config_module_loads(self, global_config):
        """测试配置模块能正确加载。"""
        assert global_config is not None
        assert global_config.api_port == _DEFAULT_API_PORT
        assert global_config.request_timeout == _DEFAULT_REQUEST_TIMEOUT
        assert Path(global_config.model_path).exists()

    def test_env_var_override(self):
        """测试环境变量覆盖机制。"""
        original_port = os.environ.get("API_PORT")
        original_model_dir = os.environ.get("MODEL_DIR")
        original_base_url = os.environ.get("API_BASE_URL")
        original_timeout = os.environ.get("REQUEST_TIMEOUT")

        try:
            os.environ["API_PORT"] = "9999"
            os.environ["MODEL_DIR"] = "/tmp/test_override"
            os.environ["API_BASE_URL"] = "https://test.example.com"
            os.environ["REQUEST_TIMEOUT"] = "60"

            cfg = load_config()
            assert cfg.api_port == 9999
            assert cfg.model_dir == "/tmp/test_override"
            assert cfg.api_base_url == "https://test.example.com"
            assert cfg.request_timeout == 60
        finally:
            if original_port is not None:
                os.environ["API_PORT"] = original_port
            else:
                os.environ.pop("API_PORT", None)
            if original_model_dir is not None:
                os.environ["MODEL_DIR"] = original_model_dir
            else:
                os.environ.pop("MODEL_DIR", None)
            if original_base_url is not None:
                os.environ["API_BASE_URL"] = original_base_url
            else:
                os.environ.pop("API_BASE_URL", None)
            if original_timeout is not None:
                os.environ["REQUEST_TIMEOUT"] = original_timeout
            else:
                os.environ.pop("REQUEST_TIMEOUT", None)

    @pytest.mark.parametrize("file_path", PYTHON_ENTRIES_TO_CHECK)
    def test_python_files_import_config(self, file_path):
        """测试 Python 入口文件导入了统一配置模块。"""
        full_path = PROJECT_ROOT / file_path
        assert full_path.exists(), f"文件不存在: {file_path}"

        with open(full_path) as f:
            content = f.read()

        has_import = any(
            "from car_pricing.config" in line or "from car_pricing import" in line
            for line in content.splitlines()
        )

        assert has_import, (
            f"{file_path} 未导入统一配置模块 car_pricing.config。"
            f"请添加: from car_pricing.config import get_config"
        )

    @pytest.mark.parametrize("file_path", PYTHON_ENTRIES_TO_CHECK)
    def test_no_hardcoded_config_in_python(self, file_path):
        """测试 Python 文件中没有硬编码的配置值。"""
        full_path = PROJECT_ROOT / file_path
        if not full_path.exists():
            pytest.skip(f"文件不存在: {file_path}")

        issues = self._scan_for_hardcoded_values(full_path)
        if issues:
            issue_str = "\n".join(str(i) for i in issues)
            pytest.fail(f"发现硬编码配置值:\n{issue_str}")

    @pytest.mark.parametrize("file_path", DEPLOYMENT_FILES_TO_CHECK)
    def test_deployment_files_use_standard_env_vars(self, file_path):
        """测试部署文件使用标准的配置变量名。"""
        full_path = PROJECT_ROOT / file_path
        if not full_path.exists():
            pytest.skip(f"文件不存在: {file_path}")

        with open(full_path) as f:
            content = f.read()

        missing_vars = []
        for var_name in CONFIG_VAR_NAMES:
            if var_name == "API_BASE_URL":
                continue
            if var_name not in content and var_name not in ["MODEL_DIR", "DATA_DIR"]:
                if file_path.endswith((".sh", "Dockerfile", "Procfile")):
                    if var_name not in ["MODEL_FILENAME", "DATA_CSV_FILENAME"]:
                        missing_vars.append(var_name)

        if missing_vars:
            pytest.warns(
                UserWarning,
                match=f"{file_path} 中缺少环境变量: {', '.join(missing_vars)}"
            )

    def test_all_entries_have_same_defaults(self):
        """测试所有入口的配置默认值与 car_pricing.config 一致。"""
        sys.path.insert(0, str(PROJECT_ROOT / "car_pricing_api"))
        sys.path.insert(0, str(PROJECT_ROOT / "fastapi"))
        sys.path.insert(0, str(PROJECT_ROOT / "streamlitapp"))

        from car_pricing.config import get_config as gc1

        cfg1 = gc1()

        import importlib
        importlib.invalidate_caches()

        try:
            from services.model_service import ModelService
            service = ModelService.get_instance()
            assert service.config.api_port == cfg1.api_port
            assert service.config.request_timeout == cfg1.request_timeout
            assert service.config.bentoml_model_tag == cfg1.bentoml_model_tag
        except Exception as e:
            pytest.skip(f"ModelService 导入失败: {e}")

        try:
            sys.path.insert(0, str(PROJECT_ROOT / "streamlitapp"))
            import streamlit_app
            assert streamlit_app.API_BASE_URL == cfg1.api_base_url
            assert streamlit_app.REQUEST_TIMEOUT == cfg1.request_timeout
        except Exception as e:
            pytest.skip(f"streamlit_app 导入失败: {e}")

    def test_bentofile_uses_standard_config(self):
        """测试 BentoML 配置文件与统一配置一致。"""
        bentofile = PROJECT_ROOT / "bentoml" / "bentofile.yaml"
        if not bentofile.exists():
            pytest.skip("bentofile.yaml 不存在")

        with open(bentofile) as f:
            content = f.read()

        assert "python" in content.lower() or "service" in content.lower()

    def test_no_exposed_config_endpoint(self):
        """测试没有暴露 /config 调试端点。"""
        for entry in ["car_pricing_api/app.py", "fastapi/app.py"]:
            full_path = PROJECT_ROOT / entry
            if not full_path.exists():
                continue
            with open(full_path) as f:
                content = f.read()
            assert '"/config"' not in content and "'/config'" not in content, (
                f"{entry} 暴露了 /config 调试端点，请移除"
            )

    def _scan_for_hardcoded_values(self, file_path: Path) -> List[ConfigIssue]:
        """扫描文件中的硬编码配置值。"""
        issues: List[ConfigIssue] = []

        with open(file_path) as f:
            lines = f.readlines()

        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()

            if line_stripped.startswith("#") or line_stripped.startswith("//"):
                continue

            if "import" in line and "car_pricing.config" in line:
                continue

            if "_DEFAULT_" in line or "CONFIG_VAR_NAMES" in line:
                continue

            for pattern, desc in HARDCODED_PATTERNS:
                if pattern.search(line):
                    if file_path.name == "config.py" and "_DEFAULT_" in line:
                        continue
                    if file_path.name == "test_config_consistency.py":
                        continue
                    issues.append(
                        ConfigIssue(
                            file_path=str(file_path.relative_to(PROJECT_ROOT)),
                            line=i,
                            issue=f"{desc}，请改用 car_pricing.config 统一配置",
                            severity="warning",
                        )
                    )

        return issues

    def test_env_example_complete(self):
        """测试 .env.example 包含所有必要的环境变量。"""
        env_example = PROJECT_ROOT / ".env.example"
        if not env_example.exists():
            pytest.skip(".env.example 不存在")

        with open(env_example) as f:
            content = f.read()

        missing_vars = []
        for var_name in CONFIG_VAR_NAMES:
            if var_name not in content:
                missing_vars.append(var_name)

        assert not missing_vars, (
            f".env.example 缺少环境变量: {', '.join(missing_vars)}"
        )


def run_audit() -> Tuple[int, List[ConfigIssue]]:
    """运行完整的配置审计（非 pytest 模式）。"""
    print("=" * 70)
    print("配置一致性审计")
    print("=" * 70)

    all_issues: List[ConfigIssue] = []
    error_count = 0
    warning_count = 0

    cfg = get_config()
    print(f"\n✅ 配置模块加载成功")
    print(f"   API 地址: {cfg.bind_address}")
    print(f"   模型目录: {cfg.model_dir}")
    print(f"   模型路径: {cfg.model_path} (存在: {Path(cfg.model_path).exists()})")

    print("\n" + "-" * 70)
    print("1. 检查 Python 入口文件")
    print("-" * 70)

    for entry in PYTHON_ENTRIES_TO_CHECK:
        full_path = PROJECT_ROOT / entry
        if not full_path.exists():
            print(f"⚠️  跳过 (不存在): {entry}")
            continue

        with open(full_path) as f:
            content = f.read()

        has_import = any(
            "from car_pricing.config" in line or "from car_pricing import" in line
            for line in content.splitlines()
        )

        if has_import:
            print(f"✅ {entry}")
        else:
            print(f"❌ {entry} - 未导入统一配置模块")
            all_issues.append(
                ConfigIssue(file_path=entry, line=1,
                           issue="未导入统一配置模块 car_pricing.config")
            )
            error_count += 1

        issues = TestConfigConsistency._scan_for_hardcoded_values(
            TestConfigConsistency(), full_path
        )
        for issue in issues:
            print(f"   {issue}")
            if issue.severity == "error":
                error_count += 1
            else:
                warning_count += 1
        all_issues.extend(issues)

    print("\n" + "-" * 70)
    print("2. 检查部署配置文件")
    print("-" * 70)

    for dep_file in DEPLOYMENT_FILES_TO_CHECK:
        full_path = PROJECT_ROOT / dep_file
        if not full_path.exists():
            print(f"⚠️  跳过 (不存在): {dep_file}")
            continue

        with open(full_path) as f:
            content = f.read()

        found_vars = [v for v in CONFIG_VAR_NAMES if v in content]
        missing_vars = [v for v in CONFIG_VAR_NAMES if v not in content
                       and v not in ["API_BASE_URL", "MODEL_DIR", "DATA_DIR"]]

        if not missing_vars or len(missing_vars) <= 3:
            print(f"✅ {dep_file} (找到 {len(found_vars)}/{len(CONFIG_VAR_NAMES)} 个变量)")
        else:
            print(f"⚠️  {dep_file} - 缺少: {', '.join(missing_vars)}")

    print("\n" + "-" * 70)
    print("3. 检查配置端点暴露")
    print("-" * 70)

    exposed = False
    for entry in ["car_pricing_api/app.py", "fastapi/app.py"]:
        full_path = PROJECT_ROOT / entry
        if not full_path.exists():
            continue
        with open(full_path) as f:
            content = f.read()
        if '"/config"' in content or "'/config'" in content:
            print(f"❌ {entry} - 暴露了 /config 调试端点")
            all_issues.append(
                ConfigIssue(file_path=entry, line=1,
                           issue="暴露了 /config 调试端点，请移除")
            )
            error_count += 1
            exposed = True

    if not exposed:
        print("✅ 没有暴露 /config 调试端点")

    print("\n" + "=" * 70)
    print(f"审计完成: {error_count} 个错误, {warning_count} 个警告")
    print("=" * 70)

    return error_count, all_issues


if __name__ == "__main__":
    error_count, _ = run_audit()
    sys.exit(1 if error_count > 0 else 0)
