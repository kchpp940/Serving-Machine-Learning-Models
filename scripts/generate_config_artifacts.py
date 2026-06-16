"""从 RuntimeConfig schema 生成配置产物。

单一真相来源：car_pricing.config.export_env_schema()
本脚本用于生成：
  - .env.example
  - 校验部署文件的环境变量覆盖
  - 生成部署文档摘要

运行方式：
    python scripts/generate_config_artifacts.py          # 全部生成
    python scripts/generate_config_artifacts.py .env     # 仅生成 .env.example
    python scripts/generate_config_artifacts.py audit    # 仅审计部署文件
    python scripts/generate_config_artifacts.py docs     # 仅生成文档
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from car_pricing.config import (  # noqa: E402
    EnvVarMeta,
    export_env_schema,
    env_schema_to_list,
    validate_env_coverage,
)


DEPLOYMENT_FILES = [
    ("car_pricing_api/Dockerfile", "docker"),
    ("fastapi/Dockerfile", "docker"),
    ("car_pricing_api/heroku.yml", "heroku"),
    ("fastapi/heroku.yml", "heroku"),
    ("car_pricing_api/vercel.json", "vercel"),
    ("fastapi/vercel.json", "vercel"),
    ("car_pricing_api/fastapi-setup.sh", "shell"),
    ("fastapi/fastapi-setup.sh", "shell"),
    ("Procfile", "procfile"),
]


CATEGORY_TITLES = {
    "api": "API server configuration",
    "model": "Model file locations",
    "data": "Training data locations",
    "client": "Client configuration",
    "bentoml": "BentoML configuration",
    "general": "Other",
}


def generate_env_example() -> str:
    """从 schema 生成 .env.example 内容。"""
    schema_list = env_schema_to_list()
    lines: List[str] = []
    lines.append("# Unified runtime configuration for Serving-Machine-Learning-Models")
    lines.append("# Single source of truth: car_pricing.config.export_env_schema()")
    lines.append("# Default values are defined in car_pricing/config.py")
    lines.append("# Copy this file to .env and adjust as needed")
    lines.append("")

    current_category = None
    for meta in schema_list:
        if not meta.expose_in_docs:
            continue
        if meta.category != current_category:
            current_category = meta.category
            lines.append(f"# === {CATEGORY_TITLES.get(current_category, current_category.upper())} ===")
        lines.append(f"# {meta.description}")
        default_str = str(meta.default)
        if meta.sensitive:
            lines.append(f"# {meta.name}=")
        else:
            lines.append(f"#{meta.name}={default_str}")
        lines.append("")

    return "\n".join(lines) + "\n"


def write_env_example() -> Path:
    """生成并写入 .env.example 文件。"""
    content = generate_env_example()
    target = PROJECT_ROOT / ".env.example"
    target.write_text(content, encoding="utf-8")
    print(f"✅ 已生成 {target.relative_to(PROJECT_ROOT)}")
    return target


def audit_deployment_file(file_path: Path, file_type: str) -> Tuple[List[str], List[str], Set[str]]:
    """审计单个部署文件的环境变量覆盖。

    Returns:
        (errors, warnings, found_vars)
    """
    errors: List[str] = []
    warnings: List[str] = []
    found_vars: Set[str] = set()

    if not file_path.exists():
        return [f"文件不存在: {file_path}"], warnings, found_vars

    content = file_path.read_text(encoding="utf-8")
    schema = export_env_schema()

    # 提取文件中出现的所有环境变量名
    for var_name in schema.keys():
        if re.search(rf"\b{var_name}\b", content):
            found_vars.add(var_name)

    # 检查 required_in_deployment 的变量是否存在
    missing = validate_env_coverage(list(found_vars))

    # 不同类型的文件有不同的期望
    expected_vars = {
        name for name, meta in schema.items()
        if meta.required_in_deployment
    }

    if file_type in ("docker", "shell", "procfile"):
        # 这些文件应包含全部变量的声明或引用
        actually_missing = [v for v in missing if v in expected_vars]
        if actually_missing:
            warnings.append(f"缺少变量声明: {', '.join(actually_missing)}")

    elif file_type == "heroku":
        # heroku.yml 的 build.config 段应包含变量
        if "build:" in content and "config:" in content:
            pass  # 结构正确
        else:
            warnings.append("未找到 build.config 段")

    elif file_type == "vercel":
        # vercel.json 的 build.env 段应包含变量
        try:
            data = json.loads(content)
            if "build" in data and "env" in data["build"]:
                pass  # 结构正确
            else:
                warnings.append("未找到 build.env 段")
        except json.JSONDecodeError as e:
            errors.append(f"JSON 解析失败: {e}")

    # 检查是否有硬编码的端口或路径
    hardcoded_patterns = [
        (r"[\"']0\.0\.0\.0[\"']", "硬编码 API_HOST"),
        (r":\s*8000\b", "硬编码端口 8000"),
        (r"[\"']sklearn_gbr\.pkl[\"']", "硬编码模型文件名"),
    ]
    for pattern, desc in hardcoded_patterns:
        for i, line in enumerate(content.splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            if "car_pricing.config" in line or "export_env_schema" in line:
                continue
            if re.search(pattern, line):
                warnings.append(f"第 {i} 行: {desc}，应使用 ${schema['API_HOST'].name} 等环境变量")

    return errors, warnings, found_vars


def audit_all_deployment_files() -> Dict[str, Dict]:
    """审计所有部署文件。"""
    results: Dict[str, Dict] = {}
    all_found: Set[str] = set()

    print("\n" + "=" * 70)
    print("部署文件环境变量审计")
    print("=" * 70)

    for rel_path, file_type in DEPLOYMENT_FILES:
        full_path = PROJECT_ROOT / rel_path
        errors, warnings, found_vars = audit_deployment_file(full_path, file_type)
        all_found.update(found_vars)

        status = "✅" if not errors and not warnings else "⚠️ " if not errors else "❌"
        print(f"\n{status} {rel_path} (类型: {file_type})")
        print(f"   找到变量: {len(found_vars)}/{len(export_env_schema())}")
        if found_vars:
            print(f"   变量列表: {', '.join(sorted(found_vars))}")
        for err in errors:
            print(f"   ❌ 错误: {err}")
        for warn in warnings:
            print(f"   ⚠️  警告: {warn}")

        results[rel_path] = {
            "errors": errors,
            "warnings": warnings,
            "found_vars": sorted(found_vars),
            "file_type": file_type,
        }

    # 总体覆盖率
    schema = export_env_schema()
    total = len(schema)
    covered = len(all_found)
    print(f"\n{'=' * 70}")
    print(f"总体覆盖率: {covered}/{total} 个变量 ({covered/total*100:.0f}%)")
    missing_global = validate_env_coverage(list(all_found))
    if missing_global:
        print(f"全局缺失: {', '.join(missing_global)}")
    else:
        print("✅ 所有 required_in_deployment 变量均已覆盖")
    print("=" * 70)

    return results


def generate_docs_summary() -> str:
    """生成配置文档摘要（Markdown 格式）。"""
    schema_list = env_schema_to_list()
    lines: List[str] = []

    lines.append("# 运行时配置环境变量")
    lines.append("")
    lines.append("> 单一真相来源：`car_pricing.config.export_env_schema()`")
    lines.append("> 本文件由 `scripts/generate_config_artifacts.py docs` 自动生成，请勿手动修改")
    lines.append("")

    current_category = None
    for meta in schema_list:
        if meta.category != current_category:
            current_category = meta.category
            lines.append("")
            lines.append(f"## {CATEGORY_TITLES.get(current_category, current_category.upper())}")
            lines.append("")
            lines.append("| 变量名 | 默认值 | 类型 | 敏感 | 必需 | 说明 |")
            lines.append("|--------|--------|------|------|------|------|")

        default_str = str(meta.default)
        if meta.sensitive:
            default_str = "***"
        sensitive = "✅" if meta.sensitive else "❌"
        required = "✅" if meta.required_in_deployment else "❌"
        lines.append(
            f"| `{meta.name}` | `{default_str}` | {meta.type} | {sensitive} | {required} | {meta.description} |"
        )

    lines.append("")
    lines.append("## 生成方式")
    lines.append("")
    lines.append("```bash")
    lines.append("# 从 schema 重新生成所有配置产物")
    lines.append("python scripts/generate_config_artifacts.py")
    lines.append("")
    lines.append("# 仅重新生成 .env.example")
    lines.append("python scripts/generate_config_artifacts.py .env")
    lines.append("")
    lines.append("# 仅审计部署文件")
    lines.append("python scripts/generate_config_artifacts.py audit")
    lines.append("")
    lines.append("# 仅生成文档")
    lines.append("python scripts/generate_config_artifacts.py docs > CONFIG_REFERENCE.md")
    lines.append("```")

    return "\n".join(lines) + "\n"


def write_docs_summary() -> Path:
    """生成并写入配置参考文档。"""
    content = generate_docs_summary()
    target = PROJECT_ROOT / "CONFIG_REFERENCE.md"
    target.write_text(content, encoding="utf-8")
    print(f"✅ 已生成 {target.relative_to(PROJECT_ROOT)}")
    return target


def main():
    parser = argparse.ArgumentParser(
        description="从 RuntimeConfig schema 生成配置产物",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
单一真相来源：car_pricing.config.export_env_schema()

命令：
  all     (默认) 执行所有操作：生成 .env.example + 审计部署文件
  .env    仅重新生成 .env.example
  audit   仅审计部署文件的环境变量覆盖
  docs    生成 CONFIG_REFERENCE.md 文档到 stdout
""",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=["all", ".env", "audit", "docs"],
        help="要执行的操作",
    )
    args = parser.parse_args()

    # 验证 schema 本身能正常导出
    schema = export_env_schema()
    print(f"✅ 从 car_pricing.config 导出 schema，共 {len(schema)} 个环境变量")

    if args.command in ("all", ".env"):
        write_env_example()

    if args.command in ("all", "audit"):
        audit_all_deployment_files()

    if args.command == "docs":
        print(generate_docs_summary())

    if args.command == "all":
        print("\n" + "=" * 70)
        print("所有配置产物生成完成")
        print("=" * 70)


if __name__ == "__main__":
    main()
