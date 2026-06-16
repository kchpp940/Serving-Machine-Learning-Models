#!/usr/bin/env python3
"""从 car_pricing.config.export_env_schema() 生成配置产物。

单一真相来源: export_env_schema()，不维护独立 overrides。
不覆盖文件已有内容，仅更新 BEGIN/END AUTO-GENERATED ENV VARS 标记段。

支持三个子命令:
  docs    - 仅更新 CONFIG_REFERENCE.md
  deploy  - 仅更新部署相关配置（.env.example + 标记段内的 Dockerfile/Procfile/vercel.json/heroku.yml）
  dry-run - 仅预览变更，不写入文件

不带参数运行相当于 docs + deploy。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from car_pricing.config import export_env_schema

BEGIN_MARKER = "# BEGIN AUTO-GENERATED ENV VARS"
END_MARKER = "# END AUTO-GENERATED ENV VARS"

BEGIN_MARKER_YAML = "# BEGIN AUTO-GENERATED ENV VARS"
END_MARKER_YAML = "# END AUTO-GENERATED ENV VARS"
BEGIN_MARKER_JSON = "\"__BEGIN_AUTO_GENERATED_ENV_VARS__\":"
END_MARKER_JSON = "\"__END_AUTO_GENERATED_ENV_VARS__\":"


def _fmt_default(v: Any) -> str:
    if v is None or v == "<auto-resolved>":
        return "<auto-resolved>"
    if v == "":
        return "<empty string>"
    return str(v)


def _fmt_md_default(v: Any) -> str:
    if v is None or v == "<auto-resolved>":
        return "*<auto-resolved>*"
    if v == "":
        return "*<empty string>*"
    return f"`{v}`"


def _fmt_deploy_cell(v: Any, same_as_local: bool = False) -> str:
    if v is None or v == "<auto-resolved>":
        s = "<auto-resolved>"
    else:
        s = str(v)
    return "同左" if same_as_local else f"`{s}`"


def _env_vars_for_deployment(schema: Dict[str, Any], deployment: str) -> Dict[str, str]:
    cols = schema["deployment_columns"]
    out: Dict[str, str] = {}
    for v in schema["variables"]:
        name = v["name"]
        if deployment in cols and name in schema["deployment_defaults"]:
            val = schema["deployment_defaults"][name].get(deployment, v["default"])
        else:
            val = v["default"]
        out[name] = "<auto-resolved>" if val is None or val == "<auto-resolved>" else str(val)
    return out


def generate_docs_markdown(schema: Dict[str, Any]) -> str:
    variables = schema["variables"]
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for v in variables:
        groups.setdefault(v["group"], []).append(v)
    cols = schema["deployment_columns"]
    local_col = cols[0]

    lines: List[str] = []
    lines.append("# 运行时配置环境变量参考")
    lines.append("")
    lines.append("> 单一真相来源：`car_pricing.config.export_env_schema()`")
    lines.append("> 本文件由 `scripts/generate_config_artifacts.py docs` 自动生成，请勿手动修改")
    lines.append("")
    lines.append("## 概览")
    lines.append("")
    lines.append(f"共 **{schema['variable_count']}** 个环境变量，按类别分组：")
    lines.append("")
    for g in schema["groups"]:
        lines.append(f"- **{g['name']}**: {g['count']} 个变量")
    lines.append("")
    lines.append("## 详细说明")
    lines.append("")

    for group_name, vars_in_group in groups.items():
        lines.append("")
        lines.append(f"### {group_name}")
        lines.append("")
        lines.append("| 变量名 | 默认值 | 类型 | 敏感 | 必需 | 说明 |")
        lines.append("|--------|--------|------|------|------|------|")
        for v in vars_in_group:
            sens = "✅" if v["sensitive"] else "❌"
            req = "✅" if v["required"] else "❌"
            lines.append(
                f"| `{v['name']}` | {_fmt_md_default(v['default'])} | "
                f"{v['type']} | {sens} | {req} | {v['description']} |"
            )
        lines.append("")

    lines.append("## 部署目标默认值差异")
    lines.append("")
    lines.append("不同部署平台对路径等变量有不同的默认值约定：")
    lines.append("")
    header = "| 变量名 | " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["--------"] + ["-----------------" for _ in cols]) + "|"
    lines.append(header)
    lines.append(sep)
    dep_defs = schema["deployment_defaults"]
    for v in variables:
        local_v = dep_defs[v["name"]][local_col]
        cells = [f"`{v['name']}`"]
        for col in cols:
            val = dep_defs[v["name"]][col]
            same = (col != local_col) and (val == local_v)
            cells.append(_fmt_deploy_cell(val, same_as_local=same))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 如何新增/修改配置")
    lines.append("")
    lines.append("1. 编辑 `car_pricing/config.py`，在 `_build_env_var_registry()` 中添加/修改变量的 EnvVarMeta 定义")
    lines.append("2. 如果需要全局 `_DEFAULT_*` 常量，在文件顶部添加")
    lines.append("3. 如果变量在不同部署目标有不同默认值，直接在该 EnvVarMeta 的 `deployment_defaults` 字段中定义")
    lines.append("4. 运行以下命令自动更新所有配置产物：")
    lines.append("")
    lines.append("```bash")
    lines.append("# 从 schema 重新生成所有配置产物（推荐）")
    lines.append("python scripts/generate_config_artifacts.py")
    lines.append("")
    lines.append("# 仅更新部署文件（.env.example + Dockerfile/Procfile/vercel.json/heroku.yml 标记段）")
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


def generate_env_example(schema: Dict[str, Any]) -> str:
    variables = schema["variables"]
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for v in variables:
        groups.setdefault(v["group"], []).append(v)

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

    for group_name, vars_in_group in groups.items():
        lines.append("")
        lines.append(f"# --- {group_name} ---")
        for v in vars_in_group:
            default_val = v["default"]
            display_default = (
                "<auto-resolved>"
                if default_val is None or default_val == "<auto-resolved>"
                else str(default_val)
            )
            sens_note = " # 敏感: 请填入真实值" if v["sensitive"] else ""
            req_note = "" if v["required"] else " # 可选"
            lines.append(f"# {v['description']} (类型: {v['type']}){sens_note}{req_note}")
            lines.append(f"# {v['name']}={display_default}")
            lines.append(f"{v['name']}={display_default}")
        lines.append("")

    return "\n".join(lines) + "\n"


def _gen_dockerfile_env_block(schema: Dict[str, Any]) -> str:
    env_vars = _env_vars_for_deployment(schema, "Docker/Heroku")
    lines = [
        f"{BEGIN_MARKER} — 不要手动修改此段，由 scripts/generate_config_artifacts.py 管理",
        f"# 环境变量默认值（单一真相: export_env_schema().deployment_defaults['Docker/Heroku']）",
    ]
    for name, val in env_vars.items():
        lines.append(f"ENV {name}=\"{val}\"")
    lines.append(f"{END_MARKER}")
    return "\n".join(lines) + "\n"


def _gen_procfile_env_block(schema: Dict[str, Any]) -> str:
    env_vars = _env_vars_for_deployment(schema, "Docker/Heroku")
    lines = [
        f"{BEGIN_MARKER} — 不要手动修改此段，由 scripts/generate_config_artifacts.py 管理",
        f"# 环境变量默认值（单一真相: export_env_schema().deployment_defaults['Docker/Heroku']）",
    ]
    pairs = ";".join(f"{name}={val}" for name, val in env_vars.items())
    lines.append(f"# Env defaults: {pairs}")
    lines.append(f"{END_MARKER}")
    return "\n".join(lines) + "\n"


def _gen_heroku_env_block(schema: Dict[str, Any]) -> str:
    env_vars = _env_vars_for_deployment(schema, "Docker/Heroku")
    lines = [
        f"{BEGIN_MARKER_YAML} — 不要手动修改此段，由 scripts/generate_config_artifacts.py 管理",
        f"# 环境变量默认值（单一真相: export_env_schema().deployment_defaults['Docker/Heroku']）",
        "  config:",
    ]
    for name, val in env_vars.items():
        lines.append(f"    {name}: \"{val}\"")
    lines.append(f"{END_MARKER_YAML}")
    return "\n".join(lines) + "\n"


def _gen_vercel_env_block(schema: Dict[str, Any]) -> str:
    env_vars = _env_vars_for_deployment(schema, "Vercel")
    entries = []
    for name, val in env_vars.items():
        entries.append(f"    {{\"key\": \"{name}\", \"value\": \"{val}\"}}")
    inner = ",\n".join(entries)
    return (
        f"  {BEGIN_MARKER_JSON} \"不要手动修改此段，由 scripts/generate_config_artifacts.py 管理\",\n"
        f"  \"__AUTO_GENERATED_ENV_VARS__\": [\n{inner}\n  ],\n"
        f"  {END_MARKER_JSON} \"标记结束\"\n"
    )


def _replace_marker_block(content: str, new_block: str, begin: str, end: str) -> str:
    pattern = re.compile(
        rf"{re.escape(begin)}.*?{re.escape(end)}",
        flags=re.DOTALL,
    )
    if pattern.search(content):
        return pattern.sub(new_block.rstrip("\n"), content, count=1)

    return content.rstrip("\n") + "\n\n" + new_block


def _update_file_block(
    path: str,
    new_block: str,
    begin: str,
    end: str,
    dry_run: bool = False,
) -> Tuple[bool, int]:
    if not os.path.isfile(path):
        return False, 0

    with open(path, "r", encoding="utf-8") as f:
        original = f.read()

    updated = _replace_marker_block(original, new_block, begin, end)

    if updated == original:
        return False, len(updated)

    if not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            f.write(updated)
    return True, len(updated)


DEPLOY_FILES: List[Tuple[str, str, str, str, str]] = [
    ("fastapi/Dockerfile", "dockerfile", BEGIN_MARKER, END_MARKER, "fastapi"),
    ("car_pricing_api/Dockerfile", "dockerfile", BEGIN_MARKER, END_MARKER, "car_pricing_api"),
    ("flaskapp/Procfile", "procfile", BEGIN_MARKER, END_MARKER, "flaskapp"),
    ("fastapi/heroku.yml", "heroku", BEGIN_MARKER_YAML, END_MARKER_YAML, "fastapi"),
    ("car_pricing_api/heroku.yml", "heroku", BEGIN_MARKER_YAML, END_MARKER_YAML, "car_pricing_api"),
    ("fastapi/vercel.json", "vercel", BEGIN_MARKER_JSON, END_MARKER_JSON, "fastapi"),
    ("car_pricing_api/vercel.json", "vercel", BEGIN_MARKER_JSON, END_MARKER_JSON, "car_pricing_api"),
]


def _gen_block(kind: str, schema: Dict[str, Any]) -> str:
    if kind == "dockerfile":
        return _gen_dockerfile_env_block(schema)
    if kind == "procfile":
        return _gen_procfile_env_block(schema)
    if kind == "heroku":
        return _gen_heroku_env_block(schema)
    if kind == "vercel":
        return _gen_vercel_env_block(schema)
    raise ValueError(f"Unknown block kind: {kind}")


def _collect_all_artifacts(schema: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    out[os.path.join(PROJECT_ROOT, "CONFIG_REFERENCE.md")] = generate_docs_markdown(schema)
    out[os.path.join(PROJECT_ROOT, ".env.example")] = generate_env_example(schema)
    return out


def _collect_deploy_marked_files(schema: Dict[str, Any]) -> List[Tuple[str, str, str, str]]:
    """返回 [(绝对路径, 新内容块, begin_marker, end_marker), ...]"""
    results: List[Tuple[str, str, str, str]] = []
    for rel, kind, begin, end, _ in DEPLOY_FILES:
        full_path = os.path.join(PROJECT_ROOT, rel)
        if not os.path.isfile(full_path):
            continue
        try:
            block = _gen_block(kind, schema)
            results.append((full_path, block, begin, end))
        except ValueError:
            continue
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="从 export_env_schema() 生成配置产物")
    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=["all", "docs", "deploy", "dry-run"],
        help="运行模式（默认 all）",
    )
    args = parser.parse_args(argv)

    schema = export_env_schema()

    if args.mode == "all":
        modes = {"docs", "deploy"}
    elif args.mode == "dry-run":
        modes = {"docs", "deploy"}
    else:
        modes = {args.mode}

    dry_run = args.mode == "dry-run"

    full_files: Dict[str, str] = {}
    if "docs" in modes:
        full_files.update(_collect_all_artifacts(schema))
    if "deploy" in modes and "docs" not in modes:
        env_example_path = os.path.join(PROJECT_ROOT, ".env.example")
        full_files[env_example_path] = generate_env_example(schema)

    marked_files: List[Tuple[str, str, str, str]] = []
    if "deploy" in modes:
        marked_files = _collect_deploy_marked_files(schema)

    if dry_run:
        print("=== DRY RUN: 以下文件将被更新 ===\n")
        for path, content in full_files.items():
            rel = os.path.relpath(path, PROJECT_ROOT)
            print(f"--- [完整重写] {rel} ({len(content)} 字符) ---")
            first_lines = content.splitlines()[:15]
            print("\n".join(first_lines))
            if len(content.splitlines()) > 15:
                print(f"... 共 {len(content.splitlines())} 行")
            print()
        for path, block, begin, end in marked_files:
            rel = os.path.relpath(path, PROJECT_ROOT)
            print(f"--- [标记段更新] {rel} ({len(block)} 字符, marker={begin!r}..{end!r}) ---")
            print(block)
        return 0

    written_full = 0
    skipped_full = 0
    for path, content in full_files.items():
        rel = os.path.relpath(path, PROJECT_ROOT)
        existing = None
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                existing = f.read()
        if existing != content:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"[✓] 完整写入: {rel} ({len(content)} 字符)")
            written_full += 1
        else:
            print(f"[=] 未变: {rel}")
            skipped_full += 1

    updated_marked = 0
    skipped_marked = 0
    for path, block, begin, end in marked_files:
        rel = os.path.relpath(path, PROJECT_ROOT)
        changed, size = _update_file_block(path, block, begin, end, dry_run=False)
        if changed:
            print(f"[✓] 标记段更新: {rel} ({size} 字符写入)")
            updated_marked += 1
        else:
            print(f"[=] 标记段未变: {rel}")
            skipped_marked += 1

    print(
        f"\n完成: 完整写入 {written_full} (跳过 {skipped_full}), "
        f"标记段更新 {updated_marked} (跳过 {skipped_marked})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
