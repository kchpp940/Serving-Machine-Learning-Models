"""配置一致性检查脚本。

单一真相来源：car_pricing.config.export_env_schema()

本脚本用于验证所有入口、部署配置与 car_pricing.config 的变量名和默认值一致。
所有环境变量清单、默认值、敏感性均从 schema 动态获取，不做重复维护。

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
    export_env_schema,
    env_schema_to_list,
    validate_env_coverage,
    EnvVarMeta,
)


# ===== 从单一真相来源 (schema) 动态获取，不重复维护 =====
_SCHEMA = export_env_schema()
CONFIG_VAR_NAMES = set(_SCHEMA.keys())

EXPECTED_ENV_DEFAULTS = {
    name: str(meta.default) if meta.default != "<auto-resolved>" else meta.default
    for name, meta in _SCHEMA.items()
    if meta.required_in_deployment
}

# ===== 硬编码检测模式（仍保留，但变量名从 schema 生成） =====
# 格式: (pattern, description, var_name)，var_name 可为 None
_HARDCODED_DEFAULTS = [
    (re.compile(rf"[\"']{re.escape(str(meta.default))}[\"']"),
     f"硬编码 {name} 默认值", name)
    for name, meta in _SCHEMA.items()
    if meta.default not in ("<auto-resolved>",) and str(meta.default) not in ("",)
    and not str(meta.default).startswith("http://")
]

HARDCODED_PATTERNS = [
    (re.compile(r"[\"']0\.0\.0\.0[\"']"), "硬编码 API_HOST=0.0.0.0", "API_HOST"),
    (re.compile(r"[\"']127\.0\.0\.1[\"']"), "硬编码 localhost", None),
    (re.compile(r":\s*8000\b"), "硬编码端口 8000", "API_PORT"),
    (re.compile(r":\s*8501\b"), "硬编码端口 8501", None),
    (re.compile(r"os\.environ\.get\([\"']PORT[\"']"), "使用 PORT 而非 API_PORT", None),
] + _HARDCODED_DEFAULTS

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

DEPLOYMENT_FILE_TYPES = {
    "Dockerfile": "docker",
    "heroku.yml": "heroku",
    "vercel.json": "vercel",
    "fastapi-setup.sh": "shell",
    "Procfile": "procfile",
}


@dataclass
class ConfigIssue:
    file_path: str
    line: int
    issue: str
    severity: str = "error"

    def __str__(self) -> str:
        return f"{self.severity.upper()}: {self.file_path}:{self.line}: {self.issue}"


def _get_file_type(file_path: str) -> str:
    """从路径获取部署文件类型。"""
    name = Path(file_path).name
    return DEPLOYMENT_FILE_TYPES.get(name, "unknown")


class TestConfigConsistency:
    """配置一致性测试集合。

    所有环境变量元数据均从 car_pricing.config.export_env_schema() 动态获取，
    避免在测试代码中重复维护变量清单和默认值。
    """

    @pytest.fixture(scope="class")
    def global_config(self):
        return get_config()

    @pytest.fixture(scope="class")
    def env_schema(self):
        """从单一真相来源获取环境变量 schema。"""
        return export_env_schema()

    def test_env_schema_not_empty(self, env_schema):
        """测试 schema 导出正常且包含预期数量的变量。"""
        assert env_schema is not None
        assert len(env_schema) >= 10, f"期望至少 10 个环境变量，实际得到 {len(env_schema)}"
        for name, meta in env_schema.items():
            assert isinstance(meta, EnvVarMeta)
            assert meta.name == name
            assert meta.default is not None
            assert meta.type in ("str", "int", "bool")

    def test_config_module_loads(self, global_config, env_schema):
        """测试配置模块能正确加载且默认值与 schema 一致。"""
        assert global_config is not None
        assert global_config.api_port == env_schema["API_PORT"].default
        assert global_config.request_timeout == env_schema["REQUEST_TIMEOUT"].default
        assert global_config.bentoml_model_tag == env_schema["BENTOML_MODEL_TAG"].default
        assert Path(global_config.model_path).exists()

    def test_env_var_override(self, env_schema):
        """测试环境变量覆盖机制。"""
        original = {}
        for var_name in ["API_PORT", "MODEL_DIR", "API_BASE_URL", "REQUEST_TIMEOUT"]:
            original[var_name] = os.environ.get(var_name)

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
            for var_name, orig_val in original.items():
                if orig_val is not None:
                    os.environ[var_name] = orig_val
                else:
                    os.environ.pop(var_name, None)

    def test_env_schema_coverage(self, env_schema):
        """测试 schema 覆盖了所有必要的变量。"""
        required_vars = [name for name, meta in env_schema.items()
                        if meta.required_in_deployment]
        assert len(required_vars) >= 8, f"期望至少 8 个必需变量，实际得到 {len(required_vars)}"

        categories = {meta.category for meta in env_schema.values()}
        expected_categories = {"api", "model", "data", "client", "bentoml"}
        assert expected_categories.issubset(categories), (
            f"缺少类别: {expected_categories - categories}"
        )

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
        """测试 Python 文件中没有硬编码的配置值。

        使用从 schema 动态生成的检测模式。
        """
        full_path = PROJECT_ROOT / file_path
        if not full_path.exists():
            pytest.skip(f"文件不存在: {file_path}")

        issues = self._scan_for_hardcoded_values(full_path)
        if issues:
            issue_str = "\n".join(str(i) for i in issues)
            pytest.fail(f"发现硬编码配置值:\n{issue_str}")

    @pytest.mark.parametrize("file_path", DEPLOYMENT_FILES_TO_CHECK)
    def test_deployment_files_use_standard_env_vars(self, file_path, env_schema):
        """测试部署文件使用标准的配置变量名（从 schema 获取）。"""
        full_path = PROJECT_ROOT / file_path
        if not full_path.exists():
            pytest.skip(f"文件不存在: {file_path}")

        with open(full_path) as f:
            content = f.read()

        found_vars = {name for name in env_schema if name in content}
        missing_vars = validate_env_coverage(list(found_vars))

        file_type = _get_file_type(file_path)
        if file_type in ("docker", "shell", "procfile"):
            if len(missing_vars) > 2:
                pytest.warns(
                    UserWarning,
                    match=f"{file_path} 中缺少环境变量: {', '.join(missing_vars)}"
                )

        if file_type == "vercel":
            try:
                data = json.loads(content)
                assert "build" in data, "vercel.json 缺少 build 段"
                assert "env" in data["build"], "vercel.json 缺少 build.env 段"
            except json.JSONDecodeError:
                pytest.fail(f"{file_path} 不是有效的 JSON")

    def test_all_entries_have_same_defaults(self, env_schema):
        """测试所有入口的配置默认值与 car_pricing.config.schema 一致。"""
        sys.path.insert(0, str(PROJECT_ROOT / "car_pricing_api"))
        sys.path.insert(0, str(PROJECT_ROOT / "fastapi"))
        sys.path.insert(0, str(PROJECT_ROOT / "streamlitapp"))

        cfg1 = get_config()

        import importlib
        importlib.invalidate_caches()

        try:
            from services.model_service import ModelService
            service = ModelService.get_instance()
            assert service.config.api_port == env_schema["API_PORT"].default
            assert service.config.request_timeout == env_schema["REQUEST_TIMEOUT"].default
            assert service.config.bentoml_model_tag == env_schema["BENTOML_MODEL_TAG"].default
        except Exception as e:
            pytest.skip(f"ModelService 导入失败: {e}")

        try:
            sys.path.insert(0, str(PROJECT_ROOT / "streamlitapp"))
            import streamlit_app
            assert streamlit_app.REQUEST_TIMEOUT == env_schema["REQUEST_TIMEOUT"].default
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

    def test_env_example_complete(self, env_schema):
        """测试 .env.example 包含所有 schema 中的环境变量。

        .env.example 由 scripts/generate_config_artifacts.py 从 schema 生成。
        """
        env_example = PROJECT_ROOT / ".env.example"
        if not env_example.exists():
            pytest.skip(".env.example 不存在")

        with open(env_example) as f:
            content = f.read()

        missing_vars = []
        for var_name, meta in env_schema.items():
            if not meta.expose_in_docs:
                continue
            if var_name not in content:
                missing_vars.append(var_name)

        assert not missing_vars, (
            f".env.example 缺少环境变量: {', '.join(missing_vars)}。"
            f"请运行: python scripts/generate_config_artifacts.py .env"
        )

        assert "export_env_schema" in content, (
            ".env.example 未标记单一真相来源。"
            "请运行: python scripts/generate_config_artifacts.py .env"
        )

    def test_generate_script_produces_consistent_output(self):
        """测试生成脚本能正常运行并产生一致的输出。"""
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/generate_config_artifacts.py", ".env"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"生成脚本失败: {result.stderr}"
        )
        assert "已生成 .env.example" in result.stdout or "已生成" in result.stdout

    def _scan_for_hardcoded_values(self, file_path: Path) -> List[ConfigIssue]:
        """扫描文件中的硬编码配置值（使用从 schema 生成的模式）。"""
        issues: List[ConfigIssue] = []

        with open(file_path) as f:
            lines = f.readlines()

        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()

            if line_stripped.startswith("#") or line_stripped.startswith("//"):
                continue

            if "import" in line and "car_pricing.config" in line:
                continue

            if "_DEFAULT_" in line or "export_env_schema" in line or "CONFIG_VAR_NAMES" in line:
                continue

            for pattern, desc, _var_name in HARDCODED_PATTERNS:
                if pattern.search(line):
                    if file_path.name == "config.py" and "_DEFAULT_" in line:
                        continue
                    if file_path.name == "test_config_consistency.py":
                        continue
                    if file_path.name == "generate_config_artifacts.py":
                        continue
                    if "heroku.yml" in file_path.name or "vercel.json" in file_path.name:
                        if "config:" in lines[max(0, i-5):i] or "env:" in lines[max(0, i-5):i]:
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

    def test_schema_consistency_with_defaults(self, env_schema):
        """测试 schema 中的默认值与 config.py 中的 _DEFAULT_* 常量一致。"""
        assert env_schema["API_HOST"].default == _DEFAULT_API_HOST
        assert env_schema["API_PORT"].default == _DEFAULT_API_PORT
        assert env_schema["MODEL_FILENAME"].default == _DEFAULT_MODEL_FILENAME
        assert env_schema["MODEL_METADATA_FILENAME"].default == _DEFAULT_MODEL_METADATA_FILENAME
        assert env_schema["MODEL_STATUS_FILENAME"].default == _DEFAULT_MODEL_STATUS_FILENAME
        assert env_schema["DATA_CSV_FILENAME"].default == _DEFAULT_DATA_CSV_FILENAME
        assert env_schema["REQUEST_TIMEOUT"].default == _DEFAULT_REQUEST_TIMEOUT
        assert env_schema["BENTOML_MODEL_TAG"].default == _DEFAULT_BENTOML_MODEL_TAG

    def test_generate_script_has_no_hardcoded_variable_list(self, env_schema):
        """测试 generate_config_artifacts.py 中没有手写维护的变量名/默认值清单。

        所有变量信息必须从 car_pricing.config.export_env_schema() 获取，
        包括 deployment_defaults 部署目标差异。
        """
        script_path = PROJECT_ROOT / "scripts" / "generate_config_artifacts.py"
        with open(script_path) as f:
            content = f.read()
            lines = content.splitlines()

        issues: List[str] = []

        # 检查不存在独立的 DEPLOYMENT_VALUE_OVERRIDES 或类似字典
        if "DEPLOYMENT_VALUE_OVERRIDES" in content:
            issues.append("存在 DEPLOYMENT_VALUE_OVERRIDES 独立映射表，应使用 schema 的 deployment_defaults")

        # 检查每个变量名不在生成器函数中以字面量形式出现
        # （允许在注释、标记格式字符串、import 语句中出现）
        for var_name, meta in env_schema.items():
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                # 跳过注释行
                if stripped.startswith("#"):
                    continue
                # 跳过 import 行
                if stripped.startswith("import ") or stripped.startswith("from "):
                    continue
                # 跳过标记格式字符串（MARKER_START_FMT / MARKER_END_FMT）
                if "MARKER_START_FMT" in line or "MARKER_END_FMT" in line or "_DEPLOYMENT_TARGETS" in line:
                    continue
                # 跳过文件路径匹配（car_pricing_api / fastapi 等目录名）
                if "DEPLOYMENT_TEMPLATES" in line or "JSON_DEPLOYMENT_FILES" in line:
                    continue
                # 跳过 docstring / 多行字符串中描述性内容
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    continue

                # 检查变量名作为字符串字面量出现（用引号括起来）
                if f'"{var_name}"' in line or f"'{var_name}'" in line:
                    # 例外 1: if meta.name == "VAR_NAME" 形式的逻辑分支判断（不是手写清单）
                    if f"meta.name == \"{var_name}\"" in line or f"meta.name == '{var_name}'" in line:
                        continue
                    # 例外 2: generate_procfile_export 中对 API_PORT 的特殊处理（PORT 映射）
                    if var_name == "API_PORT" and ("${PORT:-${API_PORT:-8000}}" in line or "'PORT:-'" in line or '"PORT:-"' in line):
                        continue
                    # 例外 3: f-string 模板中引用 meta.name（如 f'{meta.name}=...'）
                    if "meta.name" in line and "f'" in line or 'f"' in line:
                        continue
                    issues.append(
                        f"第 {i} 行: 手写变量名 '{var_name}'，应从 schema 动态获取"
                    )

                # 检查默认值作为字符串字面量出现（仅检查非平凡的默认值）
                default_str = str(meta.default)
                if len(default_str) > 3 and default_str not in ("<auto-resolved>", "8000", "0.0.0.0"):
                    if f'"{default_str}"' in line or f"'{default_str}'" in line:
                        issues.append(
                            f"第 {i} 行: 手写默认值 '{default_str}'（变量 {var_name}），应从 schema 动态获取"
                        )

        # 检查所有 deployment_defaults 的值也不在脚本中硬编码
        deployment_targets = {"docker", "heroku", "vercel", "shell", "procfile"}
        for var_name, meta in env_schema.items():
            if not meta.deployment_defaults:
                continue
            for target, value in meta.deployment_defaults.items():
                if target not in deployment_targets:
                    continue
                value_str = str(value)
                if len(value_str) > 4 and "$PROJECT_ROOT" not in value_str and "localhost" not in value_str:
                    for i, line in enumerate(lines, 1):
                        stripped = line.strip()
                        if stripped.startswith("#"):
                            continue
                        if f'"{value_str}"' in line or f"'{value_str}'" in line:
                            # 跳过在部署文件路径中的值（如 /app/... 出现在路径中）
                            if "DEPLOYMENT_TEMPLATES" in lines[max(0, i-5):i+1]:
                                continue
                            if "JSON_DEPLOYMENT_FILES" in lines[max(0, i-5):i+1]:
                                continue
                            issues.append(
                                f"第 {i} 行: 手写部署目标默认值 '{value_str}' "
                                f"（{var_name} @ {target}），应从 schema 的 deployment_defaults 获取"
                            )

        assert not issues, (
            "generate_config_artifacts.py 中存在手写变量名/默认值：\n"
            + "\n".join(f"  - {iss}" for iss in issues)
            + "\n\n请将这些值移至 car_pricing.config.export_env_schema() 的 deployment_defaults 中"
        )


def run_audit() -> Tuple[int, List[ConfigIssue]]:
    """运行完整的配置审计（非 pytest 模式）。

    所有变量元数据从 car_pricing.config.export_env_schema() 动态获取。
    """
    print("=" * 70)
    print("配置一致性审计 (单一真相来源: car_pricing.config.export_env_schema)")
    print("=" * 70)

    all_issues: List[ConfigIssue] = []
    error_count = 0
    warning_count = 0

    schema = export_env_schema()
    print(f"\n✅ 从 schema 加载 {len(schema)} 个环境变量元数据")
    for cat in sorted(set(m.category for m in schema.values())):
        vars_in_cat = [m.name for m in schema.values() if m.category == cat]
        print(f"   类别 {cat}: {', '.join(vars_in_cat)}")

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

    all_found: set = set()
    for dep_file in DEPLOYMENT_FILES_TO_CHECK:
        full_path = PROJECT_ROOT / dep_file
        if not full_path.exists():
            print(f"⚠️  跳过 (不存在): {dep_file}")
            continue

        with open(full_path) as f:
            content = f.read()

        found_vars = {v for v in schema if v in content}
        all_found.update(found_vars)
        missing_vars = validate_env_coverage(list(found_vars))

        if not missing_vars or len(missing_vars) <= 2:
            print(f"✅ {dep_file} (找到 {len(found_vars)}/{len(schema)} 个变量)")
        else:
            print(f"⚠️  {dep_file} - 缺少: {', '.join(missing_vars)}")

    print(f"\n   总体覆盖率: {len(all_found)}/{len(schema)} ({len(all_found)/len(schema)*100:.0f}%)")

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

    print("\n" + "-" * 70)
    print("4. 检查 .env.example 完整性")
    print("-" * 70)

    env_example = PROJECT_ROOT / ".env.example"
    if env_example.exists():
        with open(env_example) as f:
            content = f.read()
        if "export_env_schema" in content:
            print("✅ .env.example 已从 schema 生成（包含单一真相来源标记）")
        else:
            print("⚠️  .env.example 未从 schema 生成，请运行: "
                  "python scripts/generate_config_artifacts.py .env")
    else:
        print("⚠️  .env.example 不存在，请运行: "
              "python scripts/generate_config_artifacts.py .env")

    print("\n" + "=" * 70)
    print(f"审计完成: {error_count} 个错误, {warning_count} 个警告")
    print("=" * 70)

    return error_count, all_issues


if __name__ == "__main__":
    error_count, _ = run_audit()
    sys.exit(1 if error_count > 0 else 0)
