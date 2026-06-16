"""从 RuntimeConfig schema 生成所有配置产物（单一真相来源）。

单一真相来源：car_pricing.config.export_env_schema()

本脚本从 schema 自动生成以下内容，新增变量时无需手改多份配置：
  - .env.example                              （覆盖生成）
  - Dockerfile 中的 ENV block                 （增量更新生成区域）
  - heroku.yml 中的 build.config 段           （增量更新生成区域）
  - vercel.json 中的 build.env 段             （JSON 结构增量更新）
  - Procfile 中的 export 变量声明             （增量更新生成区域）
  - fastapi-setup.sh 中的 export 变量声明     （增量更新生成区域）
  - CONFIG_REFERENCE.md 配置参考文档          （覆盖生成）

标记格式：
  # === <SECTION>_GENERATED_START ===
  ... (自动生成的内容，下次运行会被覆盖)
  # === <SECTION>_GENERATED_END ===

部署目标的默认值差异完全由 schema 中 EnvVarMeta.deployment_defaults 控制：
  - docker/heroku/procfile: 容器内绝对路径 /app/...
  - shell: 动态 $PROJECT_ROOT/...
  - vercel: 相对路径 ./...

运行方式：
    python scripts/generate_config_artifacts.py              # 全部生成（推荐）
    python scripts/generate_config_artifacts.py deploy       # 仅生成部署配置
    python scripts/generate_config_artifacts.py .env         # 仅生成 .env.example
    python scripts/generate_config_artifacts.py audit        # 仅审计部署文件
    python scripts/generate_config_artifacts.py docs         # 仅打印文档到 stdout
    python scripts/generate_config_artifacts.py dry-run      # 预览变更不写入
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from car_pricing.config import (  # noqa: E402
    EnvVarMeta,
    export_env_schema,
    env_schema_to_list,
    validate_env_coverage,
)


# ===== 生成区域标记常量（会自动加上 comment_prefix 和 === 装饰） =====
MARKER_START_FMT = "{prefix} === {section}_GENERATED_START ==="
MARKER_END_FMT = "{prefix} === {section}_GENERATED_END ==="

CATEGORY_TITLES = {
    "api": "API server configuration",
    "model": "Model file locations",
    "data": "Training data locations",
    "client": "Client configuration",
    "bentoml": "BentoML configuration",
    "general": "Other",
}

# 部署目标名（所有目标名必须与 schema 中的 deployment_defaults key 完全一致）
_DEPLOYMENT_TARGETS = {"docker", "heroku", "vercel", "shell", "procfile"}


# =============================================================================
# 核心工具：根据 schema 生成格式化后的变量名/默认值列表
# =============================================================================

def _resolve_value(meta: EnvVarMeta, target: str) -> str:
    """根据部署目标解析变量的默认值字符串（单一真相来源：schema 中的 deployment_defaults）。

    Args:
        meta: schema 中的变量元数据
        target: 部署目标名 (docker/heroku/vercel/shell/procfile)
    """
    if target not in _DEPLOYMENT_TARGETS:
        raise ValueError(f"未知部署目标: {target}，必须在 {sorted(_DEPLOYMENT_TARGETS)} 中")

    raw_value = meta.value_for(target)
    if raw_value == "<auto-resolved>":
        return str(raw_value)
    if meta.type == "int":
        return str(int(raw_value))
    return str(raw_value)


def generate_docker_env() -> str:
    """生成 Dockerfile 中的 ENV 段。"""
    schema_list = env_schema_to_list()
    lines: List[str] = []
    lines.append("# 由 scripts/generate_config_artifacts.py 从 car_pricing.config.export_env_schema() 自动生成")
    lines.append("# 请勿手动编辑此段，修改后运行生成脚本覆盖")
    lines.append(f"# 共 {len(schema_list)} 个环境变量（部署目标: docker）")
    lines.append("")

    current_category = None
    for meta in schema_list:
        if meta.category != current_category:
            current_category = meta.category
            lines.append(f"# === {CATEGORY_TITLES.get(current_category, current_category.upper())} ===")

        value = _resolve_value(meta, "docker")
        lines.append(f"ENV {meta.name}={value}")

    return "\n".join(lines) + "\n"


def generate_heroku_build_config() -> str:
    """生成 heroku.yml 中的 build.config 段。"""
    schema_list = env_schema_to_list()
    lines: List[str] = []
    lines.append("  # 由 scripts/generate_config_artifacts.py 从 car_pricing.config.export_env_schema() 自动生成")
    lines.append("  # 请勿手动编辑此段，修改后运行生成脚本覆盖")
    lines.append("  config:")

    for meta in schema_list:
        value = _resolve_value(meta, "heroku")
        lines.append(f'    {meta.name}: "{value}"')

    return "\n".join(lines) + "\n"


def generate_shell_export() -> str:
    """生成 shell 脚本中的 export 段（bash 语法）。"""
    schema_list = env_schema_to_list()
    lines: List[str] = []
    lines.append("# 由 scripts/generate_config_artifacts.py 从 car_pricing.config.export_env_schema() 自动生成")
    lines.append("# 请勿手动编辑此段，修改后运行生成脚本覆盖")
    lines.append(f"# 共 {len(schema_list)} 个环境变量（部署目标: shell）")
    lines.append("")

    for meta in schema_list:
        value = _resolve_value(meta, "shell")
        # shell 语法：export VAR_NAME=${VAR_NAME:-default}
        lines.append(f'export {meta.name}=${{{meta.name}:-{value}}}')

    return "\n".join(lines) + "\n"


def generate_procfile_export() -> str:
    """生成 Procfile 中的 export 段（带 web: 前缀，末尾有 && 续行符）。

    输出格式：
        web: export A=... && export B=... && export PORT_MAP && \

    Procfile 中标记段之后应该继续写续行（以缩进行开头），例如：
             export PYTHONPATH=... && \
             cd ... && \
             gunicorn ...
    """
    schema_list = env_schema_to_list()
    lines: List[str] = []

    # Procfile 中所有 export 在一行，用 && 连接
    exports = []
    for meta in schema_list:
        value = _resolve_value(meta, "procfile")
        if meta.name == "API_PORT":
            # Heroku 特殊：PORT → API_PORT 映射
            exports.append('API_PORT=${PORT:-${API_PORT:-8000}}')
        else:
            exports.append(f'{meta.name}=${{{meta.name}:-{value}}}')

    # 生成 web: 开头的行，末尾 && \ 表示后续还有续行
    export_str = " && ".join(f"export {e}" for e in exports)
    lines.append(f"web: {export_str} && \\")

    return "\n".join(lines) + "\n"


def generate_env_dict_for_vercel() -> Dict[str, str]:
    """生成 JSON 格式的 env 字典（用于 vercel.json）。"""
    schema_list = env_schema_to_list()
    return {
        meta.name: _resolve_value(meta, "vercel")
        for meta in schema_list
    }


def generate_env_example() -> str:
    """从 schema 生成 .env.example 内容。"""
    schema_list = env_schema_to_list()
    lines: List[str] = []
    lines.append("# Unified runtime configuration for Serving-Machine-Learning-Models")
    lines.append("# Single source of truth: car_pricing.config.export_env_schema()")
    lines.append("# Generated by: scripts/generate_config_artifacts.py .env")
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
        value = _resolve_value(meta, "vercel")  # 用本地默认值
        if meta.sensitive:
            lines.append(f"# {meta.name}=")
        else:
            lines.append(f"#{meta.name}={value}")
        lines.append("")

    return "\n".join(lines) + "\n"


def generate_docs_summary() -> str:
    """生成配置文档摘要（Markdown 格式）。"""
    schema_list = env_schema_to_list()
    lines: List[str] = []

    lines.append("# 运行时配置环境变量参考")
    lines.append("")
    lines.append("> 单一真相来源：`car_pricing.config.export_env_schema()`")
    lines.append("> 本文件由 `scripts/generate_config_artifacts.py docs` 自动生成，请勿手动修改")
    lines.append("")
    lines.append("## 概览")
    lines.append("")
    lines.append(f"共 **{len(schema_list)}** 个环境变量，按类别分组：")
    lines.append("")
    category_counts: Dict[str, int] = {}
    for meta in schema_list:
        category_counts[meta.category] = category_counts.get(meta.category, 0) + 1
    for cat, count in category_counts.items():
        title = CATEGORY_TITLES.get(cat, cat.upper())
        lines.append(f"- **{title}**: {count} 个变量")
    lines.append("")
    lines.append("## 详细说明")
    lines.append("")

    current_category = None
    for meta in schema_list:
        if meta.category != current_category:
            current_category = meta.category
            lines.append("")
            lines.append(f"### {CATEGORY_TITLES.get(current_category, current_category.upper())}")
            lines.append("")
            lines.append("| 变量名 | 默认值 | 类型 | 敏感 | 必需 | 说明 |")
            lines.append("|--------|--------|------|------|------|------|")

        default_str = str(meta.default)
        if meta.sensitive:
            default_str = "***"
        elif default_str == "<auto-resolved>":
            default_str = f"*{default_str}*"
        sensitive = "✅" if meta.sensitive else "❌"
        required = "✅" if meta.required_in_deployment else "❌"
        lines.append(
            f"| `{meta.name}` | `{default_str}` | {meta.type} | {sensitive} | {required} | {meta.description} |"
        )

    lines.append("")
    lines.append("## 部署目标默认值差异")
    lines.append("")
    lines.append("不同部署平台对路径等变量有不同的默认值约定：")
    lines.append("")
    lines.append("| 变量名 | Schema 本地默认 | Docker/Heroku | Shell 脚本 | Vercel |")
    lines.append("|--------|-----------------|---------------|------------|--------|")
    for meta in schema_list:
        local = str(meta.default) if not meta.sensitive else "***"
        docker_v = _resolve_value(meta, "docker")
        shell_v = _resolve_value(meta, "shell")
        vercel_v = _resolve_value(meta, "vercel")
        # 如果全部相同则合并显示
        if local == docker_v == shell_v == vercel_v:
            lines.append(f"| `{meta.name}` | `{local}` | 同左 | 同左 | 同左 |")
        else:
            lines.append(f"| `{meta.name}` | `{local}` | `{docker_v}` | `{shell_v}` | `{vercel_v}` |")

    lines.append("")
    lines.append("## 如何新增/修改配置")
    lines.append("")
    lines.append("1. 编辑 `car_pricing/config.py`，在 `export_env_schema()` 中添加/修改变量定义")
    lines.append("2. 如果需要全局 `_DEFAULT_*` 常量，在文件顶部添加")
    lines.append("3. 如果变量在不同部署目标有不同默认值（如绝对路径），在 `export_env_schema()` 中为该变量添加 `deployment_defaults` 字典")
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

    return "\n".join(lines) + "\n"


# =============================================================================
# 所有需要增量更新的部署文件及其配置
# =============================================================================

DEPLOYMENT_TEMPLATES = [
    {
        "path": "car_pricing_api/Dockerfile",
        "section": "DOCKER_ENV",
        "generator": generate_docker_env,
    },
    {
        "path": "fastapi/Dockerfile",
        "section": "DOCKER_ENV",
        "generator": generate_docker_env,
    },
    {
        "path": "car_pricing_api/heroku.yml",
        "section": "HEROKU_BUILD_CONFIG",
        "generator": generate_heroku_build_config,
    },
    {
        "path": "fastapi/heroku.yml",
        "section": "HEROKU_BUILD_CONFIG",
        "generator": generate_heroku_build_config,
    },
    {
        "path": "car_pricing_api/fastapi-setup.sh",
        "section": "SHELL_EXPORT",
        "generator": generate_shell_export,
    },
    {
        "path": "fastapi/fastapi-setup.sh",
        "section": "SHELL_EXPORT",
        "generator": generate_shell_export,
    },
    {
        "path": "Procfile",
        "section": "PROCFILE_EXPORT",
        "generator": generate_procfile_export,
    },
]

# JSON 格式的部署配置（vercel.json）
JSON_DEPLOYMENT_FILES = [
    {
        "path": "car_pricing_api/vercel.json",
        "json_path": ["build", "env"],
        "generator": generate_env_dict_for_vercel,
    },
    {
        "path": "fastapi/vercel.json",
        "json_path": ["build", "env"],
        "generator": generate_env_dict_for_vercel,
    },
]


# =============================================================================
# 文件更新函数：增量更新带有标记区域的文件
# =============================================================================

def _replace_marker_section(
    content: str,
    section: str,
    new_block: str,
    comment_prefix: str = "#",
) -> Tuple[str, bool]:
    """替换内容中的标记区域。

    如果标记不存在，则在文件末尾追加；如果存在则替换整个段（含标记行）。

    Returns:
        (更新后的内容, 是否发生了变更)
    """
    start_marker = MARKER_START_FMT.format(prefix=comment_prefix, section=section)
    end_marker = MARKER_END_FMT.format(prefix=comment_prefix, section=section)

    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)

    # 如果找不到标记，则在文件末尾追加
    if start_idx == -1 or end_idx == -1:
        if not content.endswith("\n"):
            content += "\n"
        content += "\n"
        content += f"{start_marker}\n"
        content += new_block
        content += f"{end_marker}\n"
        return content, True

    # 找到替换的起止位置（包含标记行本身）
    start_line_start = content.rfind("\n", 0, start_idx)
    if start_line_start == -1:
        start_line_start = 0
    else:
        start_line_start += 1  # 跳过换行符本身，指向行首

    end_line_end = content.find("\n", end_idx)
    if end_line_end == -1:
        end_line_end = len(content)
    else:
        end_line_end += 1  # 包含换行符

    new_section = (
        f"{start_marker}\n"
        f"{new_block}"
        f"{end_marker}\n"
    )

    new_content = content[:start_line_start] + new_section + content[end_line_end:]

    changed = new_content != content
    return new_content, changed


def _update_json_field(
    data: Dict,
    json_path: List[str],
    new_value: Dict,
) -> Tuple[Dict, bool]:
    """更新 JSON 数据中的嵌套字段。

    Returns:
        (更新后的数据, 是否发生了变更)
    """
    current = data
    for key in json_path[:-1]:
        if key not in current:
            current[key] = {}
        elif not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]

    final_key = json_path[-1]
    old_value = current.get(final_key, {})

    if old_value == new_value:
        return data, False

    current[final_key] = new_value
    return data, True


def update_deployment_file(
    template: Dict,
    dry_run: bool = False,
) -> Tuple[bool, str]:
    """更新单个部署配置文件（带标记段的文本文件）。

    Returns:
        (是否变更, 变更说明)
    """
    file_path = PROJECT_ROOT / template["path"]
    if not file_path.exists():
        return False, f"跳过（不存在）: {template['path']}"

    new_block = template["generator"]()
    content = file_path.read_text(encoding="utf-8")

    new_content, changed = _replace_marker_section(
        content,
        template["section"],
        new_block,
        comment_prefix="#",
    )

    if not changed:
        return False, f"✅ 无变更: {template['path']}"

    if dry_run:
        return True, f"[DRY-RUN] 将更新: {template['path']}"

    file_path.write_text(new_content, encoding="utf-8")
    return True, f"✅ 已更新: {template['path']}"


def update_json_deployment_file(
    template: Dict,
    dry_run: bool = False,
) -> Tuple[bool, str]:
    """更新 JSON 格式的部署配置文件。"""
    file_path = PROJECT_ROOT / template["path"]
    if not file_path.exists():
        return False, f"跳过（不存在）: {template['path']}"

    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return False, f"❌ JSON 解析失败: {template['path']}: {e}"

    new_env = template["generator"]()
    new_data, changed = _update_json_field(data, template["json_path"], new_env)

    if not changed:
        return False, f"✅ 无变更: {template['path']}"

    if dry_run:
        return True, f"[DRY-RUN] 将更新 JSON {'.'.join(template['json_path'])}: {template['path']}"

    file_path.write_text(
        json.dumps(new_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return True, f"✅ 已更新 JSON {'.'.join(template['json_path'])}: {template['path']}"


# =============================================================================
# 产物写入函数
# =============================================================================

def write_env_example() -> Path:
    """生成并写入 .env.example 文件。"""
    content = generate_env_example()
    target = PROJECT_ROOT / ".env.example"
    target.write_text(content, encoding="utf-8")
    print(f"✅ 已生成 {target.relative_to(PROJECT_ROOT)}")
    return target


def write_docs_summary() -> Path:
    """生成并写入配置参考文档。"""
    content = generate_docs_summary()
    target = PROJECT_ROOT / "CONFIG_REFERENCE.md"
    target.write_text(content, encoding="utf-8")
    print(f"✅ 已生成 {target.relative_to(PROJECT_ROOT)}")
    return target


# =============================================================================
# 审计函数
# =============================================================================

def audit_deployment_file(file_path: Path, file_type: str) -> Tuple[List[str], List[str], Set[str]]:
    """审计单个部署文件的环境变量覆盖。"""
    errors: List[str] = []
    warnings: List[str] = []
    found_vars: Set[str] = set()

    if not file_path.exists():
        return [f"文件不存在: {file_path}"], warnings, found_vars

    content = file_path.read_text(encoding="utf-8")
    schema = export_env_schema()

    for var_name in schema.keys():
        if re.search(rf"\b{var_name}\b", content):
            found_vars.add(var_name)

    missing = validate_env_coverage(list(found_vars))
    expected_vars = {
        name for name, meta in schema.items()
        if meta.required_in_deployment
    }

    if file_type in ("docker", "shell", "procfile"):
        actually_missing = [v for v in missing if v in expected_vars]
        if actually_missing:
            warnings.append(f"缺少变量声明: {', '.join(actually_missing)}")

    elif file_type == "heroku":
        if "build:" in content and "config:" in content:
            pass
        else:
            warnings.append("未找到 build.config 段")

    elif file_type == "vercel":
        try:
            data = json.loads(content)
            if "build" in data and "env" in data["build"]:
                pass
            else:
                warnings.append("未找到 build.env 段")
        except json.JSONDecodeError as e:
            errors.append(f"JSON 解析失败: {e}")

    return errors, warnings, found_vars


def audit_all_deployment_files() -> Dict[str, Dict]:
    """审计所有部署文件。"""
    results: Dict[str, Dict] = {}
    all_found: Set[str] = set()

    file_type_map = {
        "Dockerfile": "docker",
        "heroku.yml": "heroku",
        "vercel.json": "vercel",
        "fastapi-setup.sh": "shell",
        "Procfile": "procfile",
    }
    all_files = [t["path"] for t in DEPLOYMENT_TEMPLATES] + [t["path"] for t in JSON_DEPLOYMENT_FILES]

    print("\n" + "=" * 70)
    print("部署文件环境变量审计")
    print("=" * 70)

    for rel_path in sorted(set(all_files)):
        full_path = PROJECT_ROOT / rel_path
        file_type = file_type_map.get(Path(rel_path).name, "unknown")
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
        }

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


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="从 RuntimeConfig schema 生成所有配置产物（单一真相来源）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
单一真相来源：car_pricing.config.export_env_schema()

命令：
  all       (默认) 全部生成：.env.example + 部署配置 + CONFIG_REFERENCE.md
  deploy    仅更新部署配置文件（Dockerfile/heroku/vercel 等）
  .env      仅重新生成 .env.example
  audit     仅审计部署文件的环境变量覆盖
  docs      生成 CONFIG_REFERENCE.md 文档
  dry-run   预览所有变更，不写入文件
""",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=["all", "deploy", ".env", "audit", "docs", "dry-run"],
        help="要执行的操作",
    )
    args = parser.parse_args()

    schema = export_env_schema()
    print(f"✅ 从 car_pricing.config 导出 schema，共 {len(schema)} 个环境变量")
    for cat in sorted(set(m.category for m in schema.values())):
        vars_in_cat = [m.name for m in schema.values() if m.category == cat]
        print(f"   类别 {cat}: {', '.join(vars_in_cat)}")

    dry_run = args.command == "dry-run"

    # ===== 生成 .env.example =====
    if args.command in ("all", ".env", "deploy") or dry_run:
        if not dry_run:
            write_env_example()
        else:
            print("[DRY-RUN] 将生成 .env.example")

    # ===== 更新部署配置文件 =====
    if args.command in ("all", "deploy") or dry_run:
        print(f"\n--- 更新文本部署文件 (标记段增量更新) ---")
        for template in DEPLOYMENT_TEMPLATES:
            changed, msg = update_deployment_file(template, dry_run=dry_run)
            prefix = "🔄 " if changed else "   "
            print(f"{prefix}{msg}")

        print(f"\n--- 更新 JSON 部署文件 (结构增量更新) ---")
        for template in JSON_DEPLOYMENT_FILES:
            changed, msg = update_json_deployment_file(template, dry_run=dry_run)
            prefix = "🔄 " if changed else "   "
            print(f"{prefix}{msg}")

    # ===== 审计部署文件 =====
    if args.command in ("all", "audit") or dry_run:
        audit_all_deployment_files()

    # ===== 生成文档 =====
    if args.command == "docs":
        print(generate_docs_summary())
    elif args.command in ("all",) and not dry_run:
        write_docs_summary()

    if args.command in ("all",):
        print("\n" + "=" * 70)
        print("所有配置产物生成完成")
        print("=" * 70)
        print("验证一致性：")
        print("  python tests/test_config_consistency.py")
        print("=" * 70)


if __name__ == "__main__":
    main()
