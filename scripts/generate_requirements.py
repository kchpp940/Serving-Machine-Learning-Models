#!/usr/bin/env python3
"""从 pyproject.toml 生成各部署目录的 requirements.txt。

依赖来源唯一：根目录 pyproject.toml 中的 [project] dependencies 和
[project.optional-dependencies] 分组。生成的 requirements.txt 是具体的
依赖列表，在 Docker、Heroku、BentoML 等不同构建上下文中都能稳定解析。

用法：
    python scripts/generate_requirements.py           # 生成所有 requirements.txt
    python scripts/generate_requirements.py --verify  # 校验现有文件是否最新
    python scripts/generate_requirements.py --dry-run # 仅打印不写入
"""
from __future__ import annotations

import argparse
import difflib
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

    if args.verify:
        if exit_code == 0:
            print("\n[OK] 所有 requirements.txt 均为最新。")
        else:
            print("\n[FAIL] 部分 requirements.txt 与 pyproject.toml 不一致。", file=sys.stderr)
    elif not args.dry_run:
        print(f"\n完成。生成/更新了 {len(DEPLOYMENT_GROUPS)} 个部署目录的 requirements.txt。")
        print("提示：在 Dockerfile / 启动脚本中使用以下统一安装模式：")
        print("    pip install -r requirements.txt")
        print("    pip install --no-deps -e ../..   # 或构建上下文中的项目根目录")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
