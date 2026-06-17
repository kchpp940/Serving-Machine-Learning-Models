#!/usr/bin/env python3
"""从 pyproject.toml 生成各部署目录的 requirements.txt，并检查入口中无 sys.path.insert。

依赖来源唯一：根目录 pyproject.toml 中的 [project] dependencies 和
[project.optional-dependencies] 分组。生成的 requirements.txt 是具体的
依赖列表，在 Docker、Heroku、BentoML 等不同构建上下文中都能稳定解析。

同时扫描部署入口文件，确保不再依赖 sys.path.insert 来支撑裸导入。

用法：
    python scripts/generate_requirements.py           # 生成所有 requirements.txt + 检查
    python scripts/generate_requirements.py --verify  # 校验现有文件是否最新 + 检查
    python scripts/generate_requirements.py --dry-run # 仅打印不写入
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"

DEPLOYMENT_GROUPS: dict[str, tuple[str, ...]] = {
    "car_pricing_api": ("core", "api"),
    "fastapi": ("core", "api"),
    "streamlitapp": ("core", "streamlit"),
    "bentoml": ("core", "bentoml"),
    "flaskapp": ("core", "flask"),
}

SYS_PATH_INSERT_PATTERN = re.compile(r"sys\.path\.insert\s*\(")

ENTRY_FILES: dict[str, list[str]] = {
    "car_pricing_api": [
        "app.py",
        "routers/prediction.py",
        "routers/schema.py",
        "routers/meta.py",
        "models.py",
        "train.py",
        "services/model_service.py",
    ],
    "fastapi": [
        "app.py",
        "train.py",
    ],
    "bentoml": [
        "service.py",
        "bentosklearn.py",
    ],
    "flaskapp": [
        "utils.py",
    ],
    "streamlitapp": [
        "streamlit_app.py",
    ],
}

HEADER = """\
# ============================================================
# AUTO-GENERATED from ../../pyproject.toml — 不要手动修改
# Run `python scripts/generate_requirements.py` to regenerate
# ============================================================

"""


def parse_pyproject() -> dict:
    """解析 pyproject.toml，优先使用 tomllib（Py3.11+），否则 tomli。"""
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib  # type: ignore

    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)


def collect_deps(data: dict, groups: tuple[str, ...]) -> list[str]:
    """根据分组收集依赖列表，保持 pyproject 中的声明顺序。"""
    project = data["project"]
    all_deps: list[str] = []
    optional = project.get("optional-dependencies", {})

    for group in groups:
        if group == "core":
            all_deps.extend(project.get("dependencies", []))
        else:
            if group not in optional:
                raise KeyError(
                    f"[project.optional-dependencies] 中不存在分组: {group}\n"
                    f"可用分组: {list(optional)}"
                )
            all_deps.extend(optional[group])

    seen: set[str] = set()
    unique: list[str] = []
    for dep in all_deps:
        key = dep.split(">")[0].split("<")[0].split("=")[0].split("!")[0].split("[")[0].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(dep)
    return unique


def render_requirements(deps: list[str]) -> str:
    content = HEADER
    for dep in deps:
        content += f"{dep}\n"
    return content


def write_requirements(target_dir: str, deps: list[str], dry_run: bool = False) -> tuple[Path, str, bool]:
    """写入 requirements.txt；返回 (路径, 内容, 是否与现有文件一致)。"""
    out_path = ROOT / target_dir / "requirements.txt"
    content = render_requirements(deps)

    if out_path.exists():
        existing = out_path.read_text(encoding="utf-8")
        same = existing == content
    else:
        same = False

    if not dry_run and not same:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")

    return out_path, content, same


def print_diff(path: Path, expected: str) -> None:
    actual = path.read_text(encoding="utf-8") if path.exists() else ""
    diff = difflib.unified_diff(
        actual.splitlines(keepends=True),
        expected.splitlines(keepends=True),
        fromfile=str(path) + " (current)",
        tofile=str(path) + " (expected)",
    )
    sys.stdout.writelines(diff)


def check_sys_path_insert() -> list[tuple[str, int, str]]:
    """扫描入口文件，返回包含 sys.path.insert 的 (相对路径, 行号, 行内容) 列表。"""
    violations: list[tuple[str, int, str]] = []

    for dir_name, files in ENTRY_FILES.items():
        for fname in files:
            fpath = ROOT / dir_name / fname
            if not fpath.exists():
                continue
            for lineno, line in enumerate(fpath.read_text(encoding="utf-8").splitlines(), start=1):
                if SYS_PATH_INSERT_PATTERN.search(line):
                    rel = f"{dir_name}/{fname}"
                    violations.append((rel, lineno, line.strip()))

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true", help="仅校验，不写入；不一致时报错退出")
    parser.add_argument("--dry-run", action="store_true", help="仅打印预期内容，不写入")
    args = parser.parse_args()

    if not PYPROJECT.exists():
        print(f"[ERROR] 找不到 pyproject.toml: {PYPROJECT}", file=sys.stderr)
        return 2

    data = parse_pyproject()
    exit_code = 0

    # ── 阶段 1：生成 / 校验 requirements.txt ──
    for target_dir, groups in DEPLOYMENT_GROUPS.items():
        try:
            deps = collect_deps(data, groups)
        except KeyError as e:
            print(f"[ERROR] {target_dir}: {e}", file=sys.stderr)
            return 2

        path, content, same = write_requirements(target_dir, deps, dry_run=args.dry_run or args.verify)

        tag = "[OK]" if same else "[UPDATED]" if not args.verify else "[MISMATCH]"
        print(f"{tag} {path.relative_to(ROOT)}  ({' + '.join(groups)})")

        if args.dry_run:
            print(f"    --- 预期内容 ---")
            for line in content.splitlines():
                print(f"    {line}")
            print()

        if args.verify and not same:
            exit_code = 1
            print(f"    !!! 文件过期，请运行 `python scripts/generate_requirements.py` 更新")
            print_diff(path, content)
            print()

    # ── 阶段 2：检查入口文件中无 sys.path.insert ──
    print()
    violations = check_sys_path_insert()
    if violations:
        exit_code = 1
        print("[FAIL] 以下入口文件仍包含 sys.path.insert：")
        for rel, lineno, line in violations:
            print(f"  {rel}:{lineno}  {line}")
        print()
        print("请改用包绝对导入（如 from car_pricing_api.models import ...），")
        print("确保可编辑安装后所有模块可稳定导入，无需手动调整 sys.path。")
    else:
        print("[OK] 所有入口文件均无 sys.path.insert，依赖可编辑安装稳定导入。")

    # ── 汇总 ──
    if args.verify:
        if exit_code == 0:
            print("\n[OK] 所有 requirements.txt 均为最新，入口文件无 sys.path.insert。")
        else:
            print("\n[FAIL] 校验未通过，请修复上述问题。", file=sys.stderr)
    elif not args.dry_run:
        print(f"\n完成。生成/更新了 {len(DEPLOYMENT_GROUPS)} 个部署目录的 requirements.txt。")
        print("提示：在 Dockerfile / 启动脚本中使用以下统一安装模式：")
        print("    pip install -r requirements.txt")
        print("    pip install --no-deps -e ../..   # 或构建上下文中的项目根目录")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
