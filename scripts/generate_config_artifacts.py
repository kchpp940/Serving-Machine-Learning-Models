#!/usr/bin/env python3
"""从 car_pricing.config.export_env_schema() 生成配置产物。

单一真相来源: export_env_schema()，不维护独立 overrides。

docs 模式: 完整重写 CONFIG_REFERENCE.md
deploy 模式:
  - 完整重写 .env.example
  - 标记段更新（仅更新 BEGIN/END 之间的内容，不覆盖其余代码）:
    * Dockerfile  → ENV 指令段（WORKDIR 之后，Docker 原生读取）
    * heroku.yml  → build.config 子项（Heroku 构建阶段读取）
    * Procfile   → web: 命令行 env 前缀（shell 实际执行）
  - 字段级更新（只改 schema 管理的变量，保留自定义配置）:
    * vercel.json → build.env 对象（Vercel 构建阶段读取）

支持子命令: all / docs / deploy / dry-run
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


# ---------------------------------------------------------------------------
# 文档 & .env.example（完整重写）
# ---------------------------------------------------------------------------

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


def generate_docs_markdown(schema: Dict[str, Any]) -> str:
    variables = schema["variables"]
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for v in variables:
        groups.setdefault(v["group"], []).append(v)
    cols = schema["deployment_columns"]
    local_col = cols[0]

    lines: List[str] = [
        "# 运行时配置环境变量参考",
        "",
        "> 单一真相来源：`car_pricing.config.export_env_schema()`",
        "> 本文件由 `scripts/generate_config_artifacts.py docs` 自动生成，请勿手动修改",
        "",
        "## 概览",
        "",
        f"共 **{schema['variable_count']}** 个环境变量，按类别分组：",
        "",
    ]
    for g in schema["groups"]:
        lines.append(f"- **{g['name']}**: {g['count']} 个变量")
    lines += ["", "## 详细说明", ""]

    for group_name, vars_in_group in groups.items():
        lines += ["", f"### {group_name}", ""]
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

    lines += [
        "## 部署目标默认值差异",
        "",
        "不同部署平台对路径等变量有不同的默认值约定：",
        "",
    ]
    header = "| 变量名 | " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["--------"] + ["-----------------" for _ in cols]) + "|"
    lines += [header, sep]
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

    lines += [
        "## 如何新增/修改配置",
        "",
        "1. 编辑 `car_pricing/config.py`，在 `_build_env_var_registry()` 中添加/修改变量的 EnvVarMeta 定义",
        "2. 如果需要全局 `_DEFAULT_*` 常量，在文件顶部添加",
        "3. 如果变量在不同部署目标有不同默认值，直接在该 EnvVarMeta 的 `deployment_defaults` 字段中定义",
        "4. 运行以下命令自动更新所有配置产物：",
        "",
        "```bash",
        "# 从 schema 重新生成所有配置产物（推荐）",
        "python scripts/generate_config_artifacts.py",
        "",
        "# 仅更新部署文件（.env.example + Dockerfile/Procfile/vercel.json/heroku.yml）",
        "python scripts/generate_config_artifacts.py deploy",
        "",
        "# 仅预览变更，不写入文件",
        "python scripts/generate_config_artifacts.py dry-run",
        "",
        "# 验证一致性",
        "python tests/test_config_consistency.py",
        "```",
        "",
    ]

    return "\n".join(lines) + "\n"


def generate_env_example(schema: Dict[str, Any]) -> str:
    variables = schema["variables"]
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for v in variables:
        groups.setdefault(v["group"], []).append(v)

    lines: List[str] = [
        "# 运行时环境变量示例",
        "# =============================================",
        "# 单一真相来源: car_pricing.config.export_env_schema()",
        "# 由 scripts/generate_config_artifacts.py deploy 自动生成",
        "#",
        "# 使用方式:",
        "#   cp .env.example .env",
        "#   # 然后编辑 .env 填入实际值",
        "#",
        "# 注意: <auto-resolved> 表示未设置时会自动按优先级解析",
        "#",
    ]

    for group_name, vars_in_group in groups.items():
        lines += ["", f"# --- {group_name} ---"]
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


# ---------------------------------------------------------------------------
# 标记段工具
# ---------------------------------------------------------------------------

def _remove_marker_segments(content: str) -> str:
    """移除所有 BEGIN/END 标记段（含标记行本身）。"""
    pattern = re.compile(
        rf"^\s*{re.escape(BEGIN_MARKER)}.*?\n.*?{re.escape(END_MARKER)}.*\n?",
        flags=re.DOTALL | re.MULTILINE,
    )
    cleaned = pattern.sub("", content)
    return cleaned.rstrip("\n") + "\n"


def _insert_after_pattern(
    content: str,
    pattern: re.Pattern,
    insert_block: str,
) -> str:
    """在匹配 pattern 的最后一行之后插入 insert_block。"""
    matches = list(pattern.finditer(content))
    if not matches:
        return content.rstrip("\n") + "\n\n" + insert_block
    last = matches[-1]
    insert_pos = last.end()
    if content[insert_pos:insert_pos + 1] != "\n":
        insert_block = "\n" + insert_block
    return content[:insert_pos] + "\n" + insert_block + content[insert_pos + 1:]


def _update_file(
    path: str,
    new_content: str,
    dry_run: bool = False,
) -> Tuple[bool, int]:
    """完整写入文件。返回 (是否变更, 字符数)。"""
    existing = None
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            existing = f.read()
    if existing == new_content:
        return False, len(new_content)
    if not dry_run:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
    return True, len(new_content)


# ---------------------------------------------------------------------------
# Dockerfile: ENV 指令段（WORKDIR 之后）
# ---------------------------------------------------------------------------

def _gen_dockerfile_env_block(schema: Dict[str, Any]) -> str:
    env_vars = _env_vars_for_deployment(schema, "Docker/Heroku")
    lines = [
        f"{BEGIN_MARKER} — 不要手动修改此段，由 scripts/generate_config_artifacts.py 管理",
        "# 环境变量默认值（Docker 构建与运行阶段均可用，来源: export_env_schema）",
    ]
    for name, val in env_vars.items():
        lines.append(f"ENV {name}=\"{val}\"")
    lines.append(f"{END_MARKER}")
    return "\n".join(lines) + "\n"


def update_dockerfile(path: str, schema: Dict[str, Any], dry_run: bool = False) -> Tuple[bool, int]:
    if not os.path.isfile(path):
        return False, 0

    with open(path, "r", encoding="utf-8") as f:
        original = f.read()

    cleaned = _remove_marker_segments(original)
    block = _gen_dockerfile_env_block(schema)

    workdir_pattern = re.compile(r"^WORKDIR\s+.*$", re.MULTILINE)
    updated = _insert_after_pattern(cleaned, workdir_pattern, block)

    if updated == original:
        return False, len(updated)

    if not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            f.write(updated)
    return True, len(updated)


# ---------------------------------------------------------------------------
# heroku.yml: build.config 子项
# ---------------------------------------------------------------------------

def _gen_heroku_config_block(schema: Dict[str, Any]) -> str:
    env_vars = _env_vars_for_deployment(schema, "Docker/Heroku")
    lines = [
        f"{BEGIN_MARKER} — 不要手动修改此段，由 scripts/generate_config_artifacts.py 管理",
        "# 构建阶段环境变量（Heroku build.config，来源: export_env_schema）",
        "  config:",
    ]
    for name, val in env_vars.items():
        lines.append(f"    {name}: \"{val}\"")
    lines.append(f"{END_MARKER}")
    return "\n".join(lines) + "\n"


def update_heroku_yml(path: str, schema: Dict[str, Any], dry_run: bool = False) -> Tuple[bool, int]:
    if not os.path.isfile(path):
        return False, 0

    with open(path, "r", encoding="utf-8") as f:
        original = f.read()

    cleaned = _remove_marker_segments(original)
    block = _gen_heroku_config_block(schema)

    build_pattern = re.compile(r"^build:\s*$", re.MULTILINE)
    updated = _insert_after_pattern(cleaned, build_pattern, block)

    if updated == original:
        return False, len(updated)

    if not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            f.write(updated)
    return True, len(updated)


# ---------------------------------------------------------------------------
# Procfile: web: 命令行 env 前缀
# ---------------------------------------------------------------------------

def _gen_procfile_web_line(schema: Dict[str, Any], existing_cmd: str) -> str:
    env_vars = _env_vars_for_deployment(schema, "Docker/Heroku")
    env_parts = [f"{name}={val}" for name, val in env_vars.items()]
    env_str = " ".join(env_parts)

    cmd = existing_cmd.strip()
    if cmd.startswith("env "):
        rest = cmd[4:].strip()
        cmd = rest

    return f"web: env {env_str} {cmd}"


def update_procfile(path: str, schema: Dict[str, Any], dry_run: bool = False) -> Tuple[bool, int]:
    if not os.path.isfile(path):
        return False, 0

    with open(path, "r", encoding="utf-8") as f:
        original = f.read()

    cleaned = _remove_marker_segments(original)

    lines = cleaned.splitlines()
    web_idx = None
    web_cmd = ""
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("web:"):
            web_idx = i
            web_cmd = stripped[4:].strip()
            break

    if web_idx is None:
        return False, len(original)

    new_web_line = _gen_procfile_web_line(schema, web_cmd)

    block_lines = [
        f"{BEGIN_MARKER} — 不要手动修改此段，由 scripts/generate_config_artifacts.py 管理",
        "# 启动环境变量（shell env 前缀，实际生效，来源: export_env_schema）",
        new_web_line,
        f"{END_MARKER}",
    ]
    block = "\n".join(block_lines) + "\n"

    before = "\n".join(lines[:web_idx])
    after = "\n".join(lines[web_idx + 1:])

    if before:
        updated = before + "\n\n" + block + after
    else:
        updated = block + after
    if after:
        updated = updated.rstrip("\n") + "\n"

    if updated == original:
        return False, len(updated)

    if not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            f.write(updated)
    return True, len(updated)


# ---------------------------------------------------------------------------
# vercel.json: build.env 字段级更新
# ---------------------------------------------------------------------------

def update_vercel_json(path: str, schema: Dict[str, Any], dry_run: bool = False) -> Tuple[bool, int]:
    if not os.path.isfile(path):
        return False, 0

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    try:
        data = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        data = {}

    if not isinstance(data, dict):
        data = {}

    if "build" not in data or not isinstance(data["build"], dict):
        data["build"] = {}

    if "env" not in data["build"] or not isinstance(data["build"]["env"], dict):
        data["build"]["env"] = {}

    env_vars = _env_vars_for_deployment(schema, "Vercel")

    for name, val in env_vars.items():
        data["build"]["env"][name] = val

    updated_raw = json.dumps(data, indent=2, ensure_ascii=False) + "\n"

    if updated_raw == raw:
        return False, len(updated_raw)

    if not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            f.write(updated_raw)
    return True, len(updated_raw)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

DEPLOY_FILE_DEFS: List[Tuple[str, str, Any]] = [
    ("fastapi/Dockerfile", "dockerfile", update_dockerfile),
    ("car_pricing_api/Dockerfile", "dockerfile", update_dockerfile),
    ("flaskapp/Procfile", "procfile", update_procfile),
    ("fastapi/heroku.yml", "heroku", update_heroku_yml),
    ("car_pricing_api/heroku.yml", "heroku", update_heroku_yml),
    ("fastapi/vercel.json", "vercel", update_vercel_json),
    ("car_pricing_api/vercel.json", "vercel", update_vercel_json),
]


def _collect_full_files(schema: Dict[str, Any], modes: set) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if "docs" in modes:
        out[os.path.join(PROJECT_ROOT, "CONFIG_REFERENCE.md")] = generate_docs_markdown(schema)
        out[os.path.join(PROJECT_ROOT, ".env.example")] = generate_env_example(schema)
    if "deploy" in modes and "docs" not in modes:
        out[os.path.join(PROJECT_ROOT, ".env.example")] = generate_env_example(schema)
    return out


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

    # --- 完整重写的文件 ---
    full_files = _collect_full_files(schema, modes)

    # --- 标记段 / 字段级更新的文件 ---
    deploy_files: List[Tuple[str, str, Any]] = []
    if "deploy" in modes:
        deploy_files = DEPLOY_FILE_DEFS

    if dry_run:
        print("=== DRY RUN: 以下文件将被更新 ===\n")
        for path, content in full_files.items():
            rel = os.path.relpath(path, PROJECT_ROOT)
            print(f"--- [完整重写] {rel} ({len(content)} 字符) ---")
            first_lines = content.splitlines()[:10]
            print("\n".join(first_lines))
            if len(content.splitlines()) > 10:
                print(f"... 共 {len(content.splitlines())} 行")
            print()

        for rel, kind, updater in deploy_files:
            full_path = os.path.join(PROJECT_ROOT, rel)
            if not os.path.isfile(full_path):
                print(f"--- [跳过] {rel} (文件不存在) ---")
                continue
            changed, size = updater(full_path, schema, dry_run=True)
            status = "变更" if changed else "未变"
            print(f"--- [{status}] {rel} ({kind} 模式, {size} 字符) ---")
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            m = re.search(
                rf"{re.escape(BEGIN_MARKER)}.*?{re.escape(END_MARKER)}",
                content,
                flags=re.DOTALL,
            )
            if m:
                preview = m.group(0).splitlines()[:8]
                print("\n".join(preview))
                if len(m.group(0).splitlines()) > 8:
                    print("...")
            elif kind == "vercel":
                try:
                    data = json.loads(content)
                    env = data.get("build", {}).get("env", {})
                    print(f"  build.env 含 {len(env)} 个变量")
                    for k in list(env.keys())[:5]:
                        print(f"    {k}={env[k]}")
                except Exception:
                    print("  (无法解析 JSON)")
            else:
                print("  (无标记段)")
            print()
        return 0

    # --- 实际写入 ---
    written_full = 0
    skipped_full = 0
    for path, content in full_files.items():
        rel = os.path.relpath(path, PROJECT_ROOT)
        changed, size = _update_file(path, content, dry_run=False)
        if changed:
            print(f"[✓] 完整写入: {rel} ({size} 字符)")
            written_full += 1
        else:
            print(f"[=] 未变: {rel}")
            skipped_full += 1

    updated_deploy = 0
    skipped_deploy = 0
    for rel, kind, updater in deploy_files:
        full_path = os.path.join(PROJECT_ROOT, rel)
        if not os.path.isfile(full_path):
            print(f"[ ] 跳过: {rel} (文件不存在)")
            continue
        changed, size = updater(full_path, schema, dry_run=False)
        if changed:
            print(f"[✓] 平台原生位置更新: {rel} ({kind}, {size} 字符)")
            updated_deploy += 1
        else:
            print(f"[=] 未变: {rel} ({kind})")
            skipped_deploy += 1

    print(
        f"\n完成: 完整写入 {written_full} (跳过 {skipped_full}), "
        f"部署文件更新 {updated_deploy} (跳过 {skipped_deploy})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
