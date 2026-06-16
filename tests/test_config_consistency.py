#!/usr/bin/env python3
"""运行时配置一致性测试。

校验多向一致性:
  1. 变量数一致 (13 个)
  2. RuntimeConfig 字段默认值 == EnvVarMeta.default
  3. EnvVarMeta 注册中心 == export_env_schema().variables
  4. deployment_columns/deployment_defaults 结构正确
  5. EnvVarMeta.value_for() 逻辑正确（无独立 DEPLOYMENT_VALUE_OVERRIDES）
  6. .env.example 条目与 schema 一致
  7. SCHEMA_SNAPSHOT_FILENAME / LINEAGE_FILENAME 完整纳入整个链路
  8. config.py 中不存在独立的第二套部署默认值来源（DEPLOYMENT_VALUE_OVERRIDES 已删除）
  9. 部署文件（Dockerfile/Procfile/heroku.yml/vercel.json）标记段存在且包含 13 个 env var

运行:
  python tests/test_config_consistency.py
  python -m pytest tests/test_config_consistency.py
"""

from __future__ import annotations

import ast
import os
import re
import sys
from dataclasses import fields
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from car_pricing.config import (
    RuntimeConfig,
    export_env_schema,
    get_env_var_registry,
    find_env_var_meta,
    _DEPLOYMENT_COLUMNS,
    AUTO_RESOLVED_MARKER,
)


ENV_EXAMPLE_PATH = os.path.join(PROJECT_ROOT, ".env.example")
CONFIG_PY_PATH = os.path.join(PROJECT_ROOT, "car_pricing", "config.py")

FIELD_TO_ENV = {
    "api_host": "API_HOST",
    "api_port": "API_PORT",
    "model_dir": "MODEL_DIR",
    "model_filename": "MODEL_FILENAME",
    "model_metadata_filename": "MODEL_METADATA_FILENAME",
    "model_status_filename": "MODEL_STATUS_FILENAME",
    "schema_snapshot_filename": "SCHEMA_SNAPSHOT_FILENAME",
    "lineage_filename": "LINEAGE_FILENAME",
    "data_dir": "DATA_DIR",
    "data_csv_filename": "DATA_CSV_FILENAME",
    "api_base_url": "API_BASE_URL",
    "request_timeout": "REQUEST_TIMEOUT",
    "bentoml_model_tag": "BENTOML_MODEL_TAG",
}

EXPECTED_ENV_VARS = sorted(FIELD_TO_ENV.values())
EXPECTED_VAR_COUNT = 13

DEPLOY_FILES_MARKER: List[Tuple[str, str, str, str]] = [
    ("fastapi/Dockerfile", "# BEGIN AUTO-GENERATED ENV VARS", "# END AUTO-GENERATED ENV VARS", "dockerfile"),
    ("car_pricing_api/Dockerfile", "# BEGIN AUTO-GENERATED ENV VARS", "# END AUTO-GENERATED ENV VARS", "dockerfile"),
    ("flaskapp/Procfile", "# BEGIN AUTO-GENERATED ENV VARS", "# END AUTO-GENERATED ENV VARS", "procfile"),
    ("fastapi/heroku.yml", "# BEGIN AUTO-GENERATED ENV VARS", "# END AUTO-GENERATED ENV VARS", "heroku"),
    ("car_pricing_api/heroku.yml", "# BEGIN AUTO-GENERATED ENV VARS", "# END AUTO-GENERATED ENV VARS", "heroku"),
]

DEPLOY_FILES_FIELD: List[str] = [
    "fastapi/vercel.json",
    "car_pricing_api/vercel.json",
]


def assert_eq(actual: Any, expected: Any, msg: str) -> None:
    if actual != expected:
        raise AssertionError(f"{msg}\n  期望: {expected!r}\n  实际: {actual!r}")


def assert_in(item: Any, collection: Any, msg: str) -> None:
    if item not in collection:
        raise AssertionError(f"{msg}: {item!r} 不在 {collection!r}")


def test_1_var_count() -> None:
    schema = export_env_schema()
    registry = get_env_var_registry()

    assert_eq(len(registry), EXPECTED_VAR_COUNT, "EnvVarMeta 注册中心变量数不匹配")
    assert_eq(len(schema["variables"]), EXPECTED_VAR_COUNT, "export_env_schema.variables 长度不匹配")
    assert_eq(schema["variable_count"], EXPECTED_VAR_COUNT, "schema.variable_count 不匹配")
    assert_eq(len(FIELD_TO_ENV), EXPECTED_VAR_COUNT, "RuntimeConfig 字段映射数不匹配")
    assert_eq(len(list(fields(RuntimeConfig))), EXPECTED_VAR_COUNT, "RuntimeConfig 实际字段数不匹配")
    print(f"[✓] 变量数一致: {EXPECTED_VAR_COUNT} 个")


def test_2_runtimeconfig_vs_registry_defaults() -> None:
    default_config = RuntimeConfig()
    mismatches: List[str] = []

    for f in fields(RuntimeConfig):
        py_field = f.name
        env_name = FIELD_TO_ENV.get(py_field)
        if env_name is None:
            mismatches.append(f"RuntimeConfig 字段 {py_field} 没有对应的环境变量映射")
            continue

        meta = find_env_var_meta(env_name)
        if meta is None:
            mismatches.append(f"环境变量 {env_name} (字段 {py_field}) 不在 EnvVarMeta 注册中心")
            continue

        dc_default = getattr(default_config, py_field)
        meta_default = meta.default
        if str(dc_default) != str(meta_default):
            mismatches.append(
                f"默认值不一致: {py_field}/{env_name}: "
                f"RuntimeConfig={dc_default!r}, EnvVarMeta={meta_default!r}"
            )

    if mismatches:
        raise AssertionError("\n".join(["RuntimeConfig vs EnvVarMeta 默认值不一致:"] + mismatches))
    print("[✓] RuntimeConfig 字段默认值 与 EnvVarMeta.default 一致")


def test_3_registry_vs_schema_vars() -> None:
    schema = export_env_schema()
    registry = get_env_var_registry()
    schema_vars_by_name = {v["name"]: v for v in schema["variables"]}
    registry_by_name = {m.name: m for m in registry}

    mismatches: List[str] = []

    for name in EXPECTED_ENV_VARS:
        if name not in schema_vars_by_name:
            mismatches.append(f"schema 缺少变量: {name}")
            continue
        if name not in registry_by_name:
            mismatches.append(f"registry 缺少变量: {name}")
            continue

        sv = schema_vars_by_name[name]
        rm = registry_by_name[name]

        if sv["name"] != rm.name:
            mismatches.append(f"{name}: name 不一致")
        if sv["type"] != rm.type:
            mismatches.append(f"{name}: type 不一致 (schema={sv['type']}, meta={rm.type})")
        if sv["sensitive"] != rm.sensitive:
            mismatches.append(f"{name}: sensitive 不一致")
        if sv["required"] != rm.required:
            mismatches.append(f"{name}: required 不一致")
        if sv["group"] != rm.group:
            mismatches.append(f"{name}: group 不一致 (schema={sv['group']}, meta={rm.group})")

    if mismatches:
        raise AssertionError("\n".join(["registry vs schema 不一致:"] + mismatches))
    print("[✓] EnvVarMeta 注册中心 与 export_env_schema().variables 一致")


def test_4_deployment_structure() -> None:
    schema = export_env_schema()
    cols = schema["deployment_columns"]
    for col in _DEPLOYMENT_COLUMNS:
        assert_in(col, cols, f"缺少部署列 {col}")

    dep_defs = schema["deployment_defaults"]
    assert_eq(len(dep_defs), EXPECTED_VAR_COUNT, "deployment_defaults 条目数不匹配")
    for env_name in EXPECTED_ENV_VARS:
        assert_in(env_name, dep_defs, f"deployment_defaults 缺少 {env_name}")
        for col in _DEPLOYMENT_COLUMNS:
            assert_in(col, dep_defs[env_name], f"{env_name}.deployment_defaults 缺少列 {col}")
    print(f"[✓] deployment_columns/deployment_defaults 结构正确 ({len(cols)} 列 × {len(dep_defs)} 条)")


def test_5_value_for_no_second_overrides_source() -> None:
    """EnvVarMeta.value_for() 直接读 deployment_defaults，不存在独立第二套 overrides 表"""
    with open(CONFIG_PY_PATH, "r", encoding="utf-8") as f:
        source = f.read()

    if "DEPLOYMENT_VALUE_OVERRIDES" in source:
        raise AssertionError(
            "config.py 中仍然存在 DEPLOYMENT_VALUE_OVERRIDES 独立表；"
            "部署默认值必须内联到每个 EnvVarMeta.deployment_defaults"
        )

    # value_for() 的实现必须直接从 self.deployment_defaults 读
    tree = ast.parse(source)
    value_for_method = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "EnvVarMeta":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "value_for":
                    value_for_method = item
                    break
            break
    if value_for_method is None:
        raise AssertionError("EnvVarMeta.value_for() 方法不存在")

    method_src = ast.get_source_segment(source, value_for_method) or ""
    if "self.deployment_defaults" not in method_src:
        raise AssertionError(
            f"EnvVarMeta.value_for() 未读取 self.deployment_defaults，方法体:\n{method_src}"
        )

    # 实际行为验证
    model_dir_meta = find_env_var_meta("MODEL_DIR")
    assert model_dir_meta is not None
    assert_eq(
        model_dir_meta.value_for("Docker/Heroku"),
        "/app/shared_models",
        "MODEL_DIR Docker/Heroku 默认值",
    )
    assert_eq(
        model_dir_meta.value_for("Schema本地默认"),
        AUTO_RESOLVED_MARKER,
        "MODEL_DIR Schema本地默认 应为 <auto-resolved>",
    )
    assert_eq(
        model_dir_meta.value_for("不存在的部署"),
        "",
        "value_for 未知部署应回退到 EnvVarMeta.default",
    )

    api_base_meta = find_env_var_meta("API_BASE_URL")
    assert api_base_meta is not None
    if "${API_PORT}" not in api_base_meta.value_for("Docker/Heroku"):
        raise AssertionError(
            f"API_BASE_URL Docker 默认值应包含 ${{API_PORT}}, 实际 {api_base_meta.value_for('Docker/Heroku')!r}"
        )

    for env_name in ("SCHEMA_SNAPSHOT_FILENAME", "LINEAGE_FILENAME"):
        meta = find_env_var_meta(env_name)
        assert meta is not None
        for col in _DEPLOYMENT_COLUMNS:
            v = meta.value_for(col)
            expected = {"SCHEMA_SNAPSHOT_FILENAME": "schema_snapshot.json",
                        "LINEAGE_FILENAME": "model_lineage.json"}[env_name]
            assert v == expected, f"{env_name}.{col} 应为 {expected!r}, 实际 {v!r}"

    print("[✓] value_for() 直接读 EnvVarMeta.deployment_defaults，无独立 DEPLOYMENT_VALUE_OVERRIDES")


def _parse_env_example() -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    result: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    if not os.path.isfile(ENV_EXAMPLE_PATH):
        raise FileNotFoundError(f".env.example 不存在: {ENV_EXAMPLE_PATH}")

    with open(ENV_EXAMPLE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("# ---") or (
                line.startswith("#") and "=" not in line
            ):
                continue
            m_comment = re.match(r"^#\s*([A-Z0-9_]+)=(.*)$", line)
            if m_comment:
                name = m_comment.group(1)
                val = m_comment.group(2)
                if name not in result:
                    result[name] = (None, None)
                result[name] = (val, result[name][1])
                continue
            m_actual = re.match(r"^([A-Z0-9_]+)=(.*)$", line)
            if m_actual:
                name = m_actual.group(1)
                val = m_actual.group(2)
                if name not in result:
                    result[name] = (None, None)
                result[name] = (result[name][0], val)

    return result


def test_6_env_example_structure() -> None:
    env_vars = _parse_env_example()

    missing = [e for e in EXPECTED_ENV_VARS if e not in env_vars]
    extra = [e for e in env_vars if e not in EXPECTED_ENV_VARS]
    if missing:
        raise AssertionError(f".env.example 缺少变量: {missing}")
    if extra:
        raise AssertionError(f".env.example 多出未知变量: {extra}")

    schema = export_env_schema()
    schema_defaults = {v["name"]: v["default"] for v in schema["variables"]}
    mismatches: List[str] = []

    for env_name in EXPECTED_ENV_VARS:
        comment_val, actual_val = env_vars[env_name]
        schema_default = schema_defaults[env_name]

        display_default = (
            "<auto-resolved>"
            if schema_default is None or schema_default == AUTO_RESOLVED_MARKER
            else str(schema_default)
        )

        if actual_val != display_default:
            mismatches.append(
                f"{env_name}: .env.example 实际赋值={actual_val!r}, "
                f"schema 默认值={display_default!r}"
            )
        if comment_val != display_default:
            mismatches.append(
                f"{env_name}: .env.example 注释赋值={comment_val!r}, "
                f"schema 默认值={display_default!r}"
            )

    if mismatches:
        raise AssertionError("\n".join([".env.example 与 schema 默认值不一致:"] + mismatches))

    print(f"[✓] .env.example 一致: {len(EXPECTED_ENV_VARS)} 个变量，默认值与 schema 对齐")


def test_7_schema_snapshot_lineage_fully_included() -> None:
    """SCHEMA_SNAPSHOT_FILENAME / LINEAGE_FILENAME 完整纳入 7 层一致性链路"""
    for env_name in ("SCHEMA_SNAPSHOT_FILENAME", "LINEAGE_FILENAME"):
        dc_field = {
            "SCHEMA_SNAPSHOT_FILENAME": "schema_snapshot_filename",
            "LINEAGE_FILENAME": "lineage_filename",
        }[env_name]
        dc_default = getattr(RuntimeConfig(), dc_field)
        assert dc_default in {"schema_snapshot.json", "model_lineage.json"}, (
            f"{env_name}: RuntimeConfig 默认值异常: {dc_default!r}"
        )
        assert FIELD_TO_ENV.get(dc_field) == env_name, f"{env_name}: FIELD_TO_ENV 缺失"

        meta = find_env_var_meta(env_name)
        assert meta is not None, f"{env_name}: 不在 EnvVarMeta 注册中心"
        assert meta.group == "Model file locations", f"{env_name}: group 不正确"
        assert meta.required is True, f"{env_name}: required 不为 True"

        schema = export_env_schema()
        schema_vars_by_name = {v["name"]: v for v in schema["variables"]}
        assert env_name in schema_vars_by_name, f"{env_name}: 不在 export_env_schema.variables"
        assert env_name in schema["deployment_defaults"], f"{env_name}: 不在 deployment_defaults"

        env_vars = _parse_env_example()
        assert env_name in env_vars, f"{env_name}: 不在 .env.example"

        assert meta.type == "str", f"{env_name}: type 不为 str"

    print("[✓] SCHEMA_SNAPSHOT_FILENAME / LINEAGE_FILENAME 完整纳入 7 层一致性链路")


def test_8_deploy_files_env_vars_present() -> None:
    """所有部署文件中都存在 13 个 schema 环境变量（标记段 或 字段级）。"""
    import json
    schema = export_env_schema()
    expected_names = [v["name"] for v in schema["variables"]]

    missing_files: List[str] = []
    missing_vars: List[str] = []

    # 标记段式文件
    for rel, begin, end, _kind in DEPLOY_FILES_MARKER:
        full_path = os.path.join(PROJECT_ROOT, rel)
        if not os.path.isfile(full_path):
            missing_files.append(rel)
            continue
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        if begin not in content or end not in content:
            missing_vars.append(f"{rel}: 缺少标记段 {begin}..{end}")
            continue
        pattern = re.compile(
            rf"{re.escape(begin)}(.*?){re.escape(end)}",
            flags=re.DOTALL,
        )
        m = pattern.search(content)
        if m is None:
            missing_vars.append(f"{rel}: 标记段无内容")
            continue
        segment = m.group(1)
        for env_name in expected_names:
            if env_name not in segment:
                missing_vars.append(f"{rel}: 标记段中缺少 {env_name}")

    # 字段级文件 (vercel.json)
    for rel in DEPLOY_FILES_FIELD:
        full_path = os.path.join(PROJECT_ROOT, rel)
        if not os.path.isfile(full_path):
            missing_files.append(rel)
            continue
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            missing_vars.append(f"{rel}: JSON 解析失败")
            continue
        env_obj = data.get("build", {}).get("env", {})
        for env_name in expected_names:
            if env_name not in env_obj:
                missing_vars.append(f"{rel}: build.env 中缺少 {env_name}")

    if missing_files:
        raise AssertionError("部署文件不存在:\n  " + "\n  ".join(missing_files))
    if missing_vars:
        raise AssertionError(
            "部署文件环境变量不完整:\n  " + "\n  ".join(missing_vars)
        )

    total = len(DEPLOY_FILES_MARKER) + len(DEPLOY_FILES_FIELD)
    print(f"[✓] {total} 个部署文件均包含 {len(expected_names)} 个环境变量")


def test_9_deploy_files_platform_native_locations() -> None:
    """环境变量放在平台原生读取的位置（不是追加到文件末尾的展示区）。"""
    import json

    # 1. Dockerfile: ENV 在 WORKDIR 之后，且是合法 ENV 指令
    for rel in ["fastapi/Dockerfile", "car_pricing_api/Dockerfile"]:
        full_path = os.path.join(PROJECT_ROOT, rel)
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        workdir_matches = list(re.finditer(r"^WORKDIR\s+", content, re.MULTILINE))
        env_matches = list(re.finditer(r"^ENV\s+", content, re.MULTILINE))
        assert workdir_matches, f"{rel}: 缺少 WORKDIR 指令"
        assert env_matches, f"{rel}: 缺少 ENV 指令"
        assert len(env_matches) == 13, f"{rel}: ENV 指令数应为 13，实际 {len(env_matches)}"
        assert env_matches[0].start() > workdir_matches[-1].start(), (
            f"{rel}: ENV 指令不在 WORKDIR 之后"
        )
        assert "BEGIN AUTO-GENERATED" in content, f"{rel}: 缺少 BEGIN 标记"
        assert content.count("BEGIN AUTO-GENERATED") == 1, f"{rel}: 有多个 BEGIN 标记"

    # 2. heroku.yml: 在 build.config 层级（YAML 解析验证）
    try:
        import yaml
        has_yaml = True
    except ImportError:
        has_yaml = False
        print("    (跳过 heroku.yml YAML 解析验证: 未安装 PyYAML)")

    if has_yaml:
        for rel in ["fastapi/heroku.yml", "car_pricing_api/heroku.yml"]:
            full_path = os.path.join(PROJECT_ROOT, rel)
            with open(full_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            assert "build" in data, f"{rel}: 缺少 build 节点"
            assert "config" in data["build"], f"{rel}: build 下缺少 config"
            assert isinstance(data["build"]["config"], dict), f"{rel}: build.config 不是 dict"
            assert len(data["build"]["config"]) == 13, (
                f"{rel}: build.config 变量数应为 13，实际 {len(data['build']['config'])}"
            )

    # 3. vercel.json: 在 build.env 字段（JSON 解析验证）
    for rel in DEPLOY_FILES_FIELD:
        full_path = os.path.join(PROJECT_ROOT, rel)
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "build" in data, f"{rel}: 缺少 build"
        assert "env" in data["build"], f"{rel}: build 缺少 env"
        assert isinstance(data["build"]["env"], dict), f"{rel}: build.env 不是 dict"
        assert len(data["build"]["env"]) == 13, (
            f"{rel}: build.env 变量数应为 13，实际 {len(data['build']['env'])}"
        )

    # 4. Procfile: web: 命令带 env 前缀，shell 实际执行
    for rel in ["flaskapp/Procfile"]:
        full_path = os.path.join(PROJECT_ROOT, rel)
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        web_lines = [
            l.strip() for l in content.splitlines()
            if l.strip() and not l.strip().startswith("#") and l.strip().startswith("web:")
        ]
        assert len(web_lines) == 1, f"{rel}: 应只有 1 条 web: 命令，实际 {len(web_lines)}"
        web_line = web_lines[0]
        assert web_line.startswith("web: env "), f"{rel}: web 命令应以 'web: env ' 开头"
        after_env = web_line[len("web: env "):]
        parts = after_env.split(" ", 1)
        assert len(parts) >= 2, f"{rel}: env 前缀后没有实际命令"
        first_var = parts[0]
        assert "=" in first_var, f"{rel}: env 前缀后第一个 token 不是变量赋值"
        assert "gunicorn" in after_env, f"{rel}: 命令中没有 gunicorn（原命令被覆盖）"

    print("[✓] 环境变量位于平台原生位置：Docker ENV/Heroku build.config/Vercel build.env/Procfile env 前缀")


def test_10_generate_script_single_source() -> None:
    """scripts/generate_config_artifacts.py 只从 export_env_schema() 读取，不维护独立 overrides"""
    script_path = os.path.join(PROJECT_ROOT, "scripts", "generate_config_artifacts.py")
    with open(script_path, "r", encoding="utf-8") as f:
        source = f.read()

    if "DEPLOYMENT_VALUE_OVERRIDES" in source:
        raise AssertionError(
            "generate_config_artifacts.py 仍然引用 DEPLOYMENT_VALUE_OVERRIDES；"
            "生成脚本必须只读取 export_env_schema()"
        )

    if "from car_pricing.config import export_env_schema" not in source:
        raise AssertionError(
            "generate_config_artifacts.py 未从 car_pricing.config 导入 export_env_schema()"
        )

    if "get_env_var_registry" in source or "find_env_var_meta" in source:
        raise AssertionError(
            "generate_config_artifacts.py 不应直接读 EnvVarMeta 注册中心；"
            "只应通过 export_env_schema() 读取"
        )

    print("[✓] generate_config_artifacts.py 单一真相来源: 仅调用 export_env_schema()")


def run_all() -> int:
    tests = [
        test_1_var_count,
        test_2_runtimeconfig_vs_registry_defaults,
        test_3_registry_vs_schema_vars,
        test_4_deployment_structure,
        test_5_value_for_no_second_overrides_source,
        test_6_env_example_structure,
        test_7_schema_snapshot_lineage_fully_included,
        test_8_deploy_files_env_vars_present,
        test_9_deploy_files_platform_native_locations,
        test_10_generate_script_single_source,
    ]
    failures = 0
    print(f"运行时配置一致性测试 ({len(tests)} 项)\n" + "=" * 60)
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"[✗] {t.__name__}:")
            for line in str(e).splitlines():
                print(f"    {line}")
        except Exception as e:
            failures += 1
            print(f"[✗] {t.__name__}: {type(e).__name__}: {e}")

    print("=" * 60)
    passed = len(tests) - failures
    if failures == 0:
        print(f"全部通过: {passed}/{len(tests)} ✓")
        return 0
    print(f"失败 {failures} 项, 通过 {passed}/{len(tests)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(run_all())
