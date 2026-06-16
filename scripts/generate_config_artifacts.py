#!/usr/bin/env python3
"""从 car_pricing.config 中的单一真相来源（EnvVarMeta + export_env_schema）生成配置产物。

支持三个子命令:
  docs    - 仅更新 CONFIG_REFERENCE.md 文档
  deploy  - 仅更新部署相关配置（.env.example, vercel.json 等 env vars）
  dry-run - 仅预览将要写入的内容，不写入文件

不带参数运行相当于 docs + deploy。
"""

from __future__ import annotations

import os
import sys
import argparse
import textwrap
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from car_pricing.config import (
    _DEPLOYMENT_COLUMNS,
    get_env_var_registry,
    export_env_schema,
    EnvVarMeta,
    AUTO_RESOLVED_MARKER,
)


def _fmt_default(v: object) -> str:
    if v is None:
        return "<auto-resolved>"
    if v == "":
        return "<empty string>"
    return str(v)


def _fmt_md_default(v: object) -> str:
    if v is None or v == AUTO_RESOLVED_MARKER:
        return "*<auto-resolved>*"
    if v == "":
        return "*<empty string>*"
    return f"`{v}`"


def _fmt_deploy_cell(v: object, same_as_local: bool = False) -> str:
    if v is None or v == AUTO_RESOLVED_MARKER:
        s = "<auto-resolved>"
    else:
        s = str(v)
    return "同左" if same_as_local else f"`{s}`"


def generate_docs_markdown() -> str:
    registry = get_env_var_registry()
    schema = export_env_schema()

    groups: Dict[str, List[EnvVarMeta]] = {}
    for m in registry:
        groups.setdefault(m.group, []).append(m)

    lines: List[str] = []
    lines.append("# 运行时配置环境变量参考")
    lines.append("")
    lines.append("> 单一真相来源：`car_pricing.config.export_env_schema()`")
    lines.append("> 本文件由 `scripts/generate_config_artifacts.py docs` 自动生成，请勿手动修改")
    lines.append("")
    lines.append("## 概览")
    lines.append("")
    lines.append(f"共 **{len(registry)}** 个环境变量，按类别分组：")
    lines.append("")
    for g in schema["groups"]:
        lines.append(f"- **{g['name']}**: {g['count']} 个变量")
    lines.append("")
    lines.append("## 详细说明")
    lines.append("")

    for group_name, metas in groups.items():
        lines.append("")
        lines.append(f"### {group_name}")
        lines.append("")
        lines.append("| 变量名 | 默认值 | 类型 | 敏感 | 必需 | 说明 |")
        lines.append("|--------|--------|------|------|------|------|")
        for m in metas:
            sens = "✅" if m.sensitive else "❌"
            req = "✅" if m.required else "❌"
            lines.append(
                f"| `{m.name}` | {_fmt_md_default(m.to_schema_dict()['default'])} | "
                f"{m.type} | {sens} | {req} | {m.description} |"
            )
        lines.append("")

    lines.append("## 部署目标默认值差异")
    lines.append("")
    lines.append("不同部署平台对路径等变量有不同的默认值约定：")
    lines.append("")
    header = "| 变量名 | " + " | ".join(_DEPLOYMENT_COLUMNS) + " |"
    sep = "|--------|" + "|".join(["-----------------" if c == "Schema本地默认" else "---------------" for c in _DEPLOYMENT_COLUMNS]) + "|"
    lines.append(header)
    lines.append(sep)
    for m in registry:
        local_v = m.value_for("Schema本地默认")
        cells = [f"`{m.name}`"]
        for col in _DEPLOYMENT_COLUMNS:
            v = m.value_for(col)
            same = (col != "Schema本地默认") and (v == local_v)
            cells.append(_fmt_deploy_cell(v, same_as_local=same))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 如何新增/修改配置")
    lines.append("")
    lines.append("1. 编辑 `car_pricing/config.py`，在 `export_env_schema()` 中添加/修改变量定义")
    lines.append("2. 如果需要全局 `_DEFAULT_*` 常量，在文件顶部添加")
    lines.append("3. 如果变量在不同部署目标有不同默认值（如绝对路径），在脚本顶部的 `DEPLOYMENT_VALUE_OVERRIDES` 添加映射")
    lines.append("4. 运行以下命令自动更新所有部署配置：")
    lines.append("")
    lines.append("```bash")
    lines.append("# 从 schema 重新生成所有配置产物（推荐）")
    lines.append("python scripts/generate_config_artifacts.py")
    lines.append("")
    lines.append("# 仅更新部署文件（Dockerfile/heroku/vercel 等）")
    lines.append("python scripts/generate_config_artifacts.py deploy")
    lines.append("")
    lines.append("# 仅预览变更，不写入文件")
    lines.append("python scripts/generate_config_artifacts.py dry-run")
    lines.append("")
    lines.append("# 验证一致性")
    lines.append("python tests/test_config_consistency.py")
    lines.append("```")
    lines.append("")

    return "\n".join(lines) + "\n"


def generate_env_example() -> str:
    registry = get_env_var_registry()

    groups: Dict[str, List[EnvVarMeta]] = {}
    for m in registry:
        groups.setdefault(m.group, []).append(m)

    lines: List[str] = []
    lines.append("# 运行时环境变量示例")
    lines.append("# =============================================")
    lines.append("# 单一真相来源: car_pricing.config.export_env_schema()")
    lines.append("# 由 scripts/generate_config_artifacts.py deploy 自动生成")
    lines.append("#")
    lines.append("# 使用方式:")
    lines.append("#   cp .env.example .env")
    lines.append("#   # 然后编辑 .env 填入实际值")
    lines.append("#")
    lines.append("# 注意: <auto-resolved> 表示未设置时会自动按优先级解析")
    lines.append("#")

    for group_name, metas in groups.items():
        lines.append("")
        lines.append(f"# --- {group_name} ---")
        for m in metas:
            default_val = m.to_schema_dict()["default"]
            display_default = (
                "<auto-resolved>"
                if default_val is None or default_val == AUTO_RESOLVED_MARKER
                else str(default_val)
            )
            sens_note = " # 敏感: 请填入真实值" if m.sensitive else ""
            req_note = "" if m.required else " # 可选"
            lines.append(f"# {m.description} (类型: {m.type}){sens_note}{req_note}")
            lines.append(f"# {m.name}={display_default}")
            lines.append(f"{m.name}={display_default}")
        lines.append("")

    return "\n".join(lines) + "\n"


def _collect_all_artifacts() -> Dict[str, str]:
    out: Dict[str, str] = {}
    out[os.path.join(PROJECT_ROOT, "CONFIG_REFERENCE.md")] = generate_docs_markdown()
    out[os.path.join(PROJECT_ROOT, ".env.example")] = generate_env_example()
    return out


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="从 car_pricing.config 生成配置产物"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=["all", "docs", "deploy", "dry-run"],
        help="运行模式（默认 all）",
    )
    args = parser.parse_args(argv)

    if args.mode == "all":
        modes = {"docs", "deploy"}
    elif args.mode == "dry-run":
        modes = {"docs", "deploy"}
    else:
        modes = {args.mode}

    all_artifacts = _collect_all_artifacts()
    docs_paths = {"CONFIG_REFERENCE.md"}
    deploy_paths = {".env.example"}

    filtered: Dict[str, str] = {}
    for path, content in all_artifacts.items():
        basename = os.path.basename(path)
        if basename in docs_paths and "docs" in modes:
            filtered[path] = content
        elif basename in deploy_paths and "deploy" in modes:
            filtered[path] = content

    if args.mode == "dry-run":
        print("=== DRY RUN: 以下文件将被更新 ===")
        print()
        for path, content in filtered.items():
            rel = os.path.relpath(path, PROJECT_ROOT)
            print(f"--- {rel} (写入 {len(content)} 字符) ---")
            first_lines = content.splitlines()[:20]
            print("\n".join(first_lines))
            if len(content.splitlines()) > 20:
                print(f"... 共 {len(content.splitlines())} 行")
            print()
        return 0

    written = 0
    for path, content in filtered.items():
        rel = os.path.relpath(path, PROJECT_ROOT)
        existing = None
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                existing = f.read()
        if existing != content:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"[✓] 写入: {rel} ({len(content)} 字符)")
            written += 1
        else:
            print(f"[=] 未变: {rel}")

    print(f"\n完成: 写入 {written} 个文件，跳过 {len(filtered) - written} 个未变文件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
