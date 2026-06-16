#!/usr/bin/env python3
"""运行时配置一致性测试。

校验四向一致性:
  1. RuntimeConfig dataclass 字段默认值
  2. export_env_schema() 返回的 variables 数组
  3. EnvVarMeta 注册中心 (get_env_var_registry())
  4. .env.example 文件中的实际条目

运行:
  python tests/test_config_consistency.py
  python -m pytest tests/test_config_consistency.py
"""

from __future__ import annotations

import os
import re
import sys
import dataclasses
from dataclasses import fields
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from car_pricing.config import (
    RuntimeConfig,
    export_env_schema,
    get_env_var_registry,
    find_env_var_meta,
    EnvVarMeta,
    _DEPLOYMENT_COLUMNS,
    AUTO_RESOLVED_MARKER,
)


ENV_EXAMPLE_PATH = os.path.join(PROJECT_ROOT, ".env.example")

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


def assert_eq(actual: Any, expected: Any, msg: str) -> None:
    if actual != expected:
        raise AssertionError(f"{msg}\n  期望: {expected!r}\n  实际: {actual!r}")


def assert_in(item: Any, collection: Any, msg: str) -> None:
    if item not in collection:
        raise AssertionError(f"{msg}: {item!r} 不在 {collection!r}")


def test_1_var_count() -> None:
    """schema / registry / dataclass / .env.example 变量数 = 13"""
    schema = export_env_schema()
    registry = get_env_var_registry()

    assert_eq(
        len(registry),
        EXPECTED_VAR_COUNT,
        f"EnvVarMeta 注册中心变量数不匹配",
    )
    assert_eq(
        len(schema["variables"]),
        EXPECTED_VAR_COUNT,
        f"export_env_schema.variables 长度不匹配",
    )
    assert_eq(
        schema["variable_count"],
        EXPECTED_VAR_COUNT,
        f"schema.variable_count 不匹配",
    )
    assert_eq(
        len(FIELD_TO_ENV),
        EXPECTED_VAR_COUNT,
        f"RuntimeConfig dataclass 字段映射数不匹配",
    )
    assert_eq(
        len([f for f in fields(RuntimeConfig)]),
        EXPECTED_VAR_COUNT,
        f"RuntimeConfig 实际字段数不匹配",
    )
    print(f"[✓] 变量数一致: {EXPECTED_VAR_COUNT} 个")


def test_2_runtimeconfig_vs_registry_defaults() -> None:
    """RuntimeConfig 字段默认值 == EnvVarMeta.default"""
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
    """EnvVarMeta 注册中心 == export_env_schema().variables"""
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


def test_4_deployment_columns_and_values() -> None:
    """deployment_columns 存在且 deployment_defaults 条目为 13 条"""
    schema = export_env_schema()
    cols = schema["deployment_columns"]
    for col in _DEPLOYMENT_COLUMNS:
        assert_in(col, cols, f"缺少部署列 {col}")

    dep_defs = schema["deployment_defaults"]
    assert_eq(
        len(dep_defs),
        EXPECTED_VAR_COUNT,
        f"deployment_defaults 条目数不匹配",
    )
    for env_name in EXPECTED_ENV_VARS:
        assert_in(env_name, dep_defs, f"deployment_defaults 缺少 {env_name}")
        for col in _DEPLOYMENT_COLUMNS:
            assert_in(col, dep_defs[env_name], f"{env_name}.deployment_defaults 缺少列 {col}")
    print(f"[✓] deployment_columns/deployment_defaults 结构正确 ({len(cols)} 列 × {len(dep_defs)} 条)")


def test_5_env_var_meta_value_for() -> None:
    """EnvVarMeta.value_for() 遵循优先级: 显式 overrides → 本地默认 → dataclass default"""
    # MODEL_DIR Docker/Heroku 应为 /app/shared_models
    model_dir_meta = find_env_var_meta("MODEL_DIR")
    assert model_dir_meta is not None
    docker_val = model_dir_meta.value_for("Docker/Heroku")
    assert_eq(docker_val, "/app/shared_models", "MODEL_DIR.Docker/Heroku 部署默认值")

    # API_BASE_URL Docker/Heroku 应包含 ${API_PORT}
    api_base_meta = find_env_var_meta("API_BASE_URL")
    assert api_base_meta is not None
    docker_val = api_base_meta.value_for("Docker/Heroku")
    if "${API_PORT}" not in str(docker_val):
        raise AssertionError(f"API_BASE_URL Docker 默认值应包含 ${{API_PORT}}, 实际 {docker_val!r}")

    # SCHEMA_SNAPSHOT_FILENAME 无 override 时所有部署列 = 本地默认
    schema_meta = find_env_var_meta("SCHEMA_SNAPSHOT_FILENAME")
    assert schema_meta is not None
    for col in _DEPLOYMENT_COLUMNS:
        v = schema_meta.value_for(col)
        assert v == "schema_snapshot.json", (
            f"SCHEMA_SNAPSHOT_FILENAME.{col} 应为 schema_snapshot.json, 实际 {v!r}"
        )

    # LINEAGE_FILENAME 无 override 时所有部署列 = 本地默认
    lineage_meta = find_env_var_meta("LINEAGE_FILENAME")
    assert lineage_meta is not None
    for col in _DEPLOYMENT_COLUMNS:
        v = lineage_meta.value_for(col)
        assert v == "model_lineage.json", (
            f"LINEAGE_FILENAME.{col} 应为 model_lineage.json, 实际 {v!r}"
        )

    print("[✓] EnvVarMeta.value_for() 与 DEPLOYMENT_VALUE_OVERRIDES 工作正确")


def _parse_env_example() -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """解析 .env.example，返回 {env_name: (comment_value, actual_value)}"""
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
            # # XXX=Y 形式（注释掉的赋值）
            m_comment = re.match(r"^#\s*([A-Z0-9_]+)=(.*)$", line)
            if m_comment:
                name = m_comment.group(1)
                val = m_comment.group(2)
                if name not in result:
                    result[name] = (None, None)
                result[name] = (val, result[name][1])
                continue
            # XXX=Y 形式（实际赋值）
            m_actual = re.match(r"^([A-Z0-9_]+)=(.*)$", line)
            if m_actual:
                name = m_actual.group(1)
                val = m_actual.group(2)
                if name not in result:
                    result[name] = (None, None)
                result[name] = (result[name][0], val)

    return result


def test_6_env_example_structure() -> None:
    """.env.example 条目数、变量名、与 schema 默认值一致"""
    env_vars = _parse_env_example()

    missing_in_env_example = [e for e in EXPECTED_ENV_VARS if e not in env_vars]
    extra_in_env_example = [e for e in env_vars if e not in EXPECTED_ENV_VARS]
    if missing_in_env_example:
        raise AssertionError(
            f".env.example 缺少变量: {missing_in_env_example}"
        )
    if extra_in_env_example:
        raise AssertionError(
            f".env.example 多出未知变量: {extra_in_env_example}"
        )

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
        raise AssertionError(
            "\n".join([".env.example 与 schema 默认值不一致:"] + mismatches)
        )

    print(f"[✓] .env.example 一致: {len(EXPECTED_ENV_VARS)} 个变量，默认值与 schema 对齐")


def test_7_schema_snapshot_lineage_included() -> None:
    """专项验证 SCHEMA_SNAPSHOT_FILENAME 和 LINEAGE_FILENAME 完整纳入整个链路"""
    for env_name in ("SCHEMA_SNAPSHOT_FILENAME", "LINEAGE_FILENAME"):
        # 1. dataclass
        dc_field = {
            "SCHEMA_SNAPSHOT_FILENAME": "schema_snapshot_filename",
            "LINEAGE_FILENAME": "lineage_filename",
        }[env_name]
        dc_default = getattr(RuntimeConfig(), dc_field)
        assert dc_default in {"schema_snapshot.json", "model_lineage.json"}, (
            f"{env_name}: RuntimeConfig 默认值异常: {dc_default!r}"
        )

        # 2. FIELD_TO_ENV 映射
        assert FIELD_TO_ENV.get(dc_field) == env_name, f"{env_name}: FIELD_TO_ENV 缺失"

        # 3. EnvVarMeta 注册
        meta = find_env_var_meta(env_name)
        assert meta is not None, f"{env_name}: 不在 EnvVarMeta 注册中心"
        assert meta.group == "Model file locations", f"{env_name}: group 不正确"
        assert meta.required is True, f"{env_name}: required 不为 True"

        # 4. schema.variables
        schema = export_env_schema()
        schema_vars_by_name = {v["name"]: v for v in schema["variables"]}
        assert env_name in schema_vars_by_name, f"{env_name}: 不在 export_env_schema.variables"

        # 5. deployment_defaults
        assert env_name in schema["deployment_defaults"], f"{env_name}: 不在 deployment_defaults"

        # 6. .env.example
        env_vars = _parse_env_example()
        assert env_name in env_vars, f"{env_name}: 不在 .env.example"

        # 7. 类型为 str
        assert meta.type == "str", f"{env_name}: type 不为 str"

    print("[✓] SCHEMA_SNAPSHOT_FILENAME / LINEAGE_FILENAME 完整纳入 7 层一致性链路")


def run_all() -> int:
    tests = [
        test_1_var_count,
        test_2_runtimeconfig_vs_registry_defaults,
        test_3_registry_vs_schema_vars,
        test_4_deployment_columns_and_values,
        test_5_env_var_meta_value_for,
        test_6_env_example_structure,
        test_7_schema_snapshot_lineage_included,
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
