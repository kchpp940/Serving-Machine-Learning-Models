#!/usr/bin/env python3
"""
Streamlit 架构约束自动校验脚本。

两条硬性限制：
1. streamlitapp/state.py  不能 import car_pricing.api_client，
   不能出现 requests / URL（http://、https://）/ API endpoint（/schema 等）。
2. streamlitapp/pages/*.py  不能直接写 st.session_state，
   只能通过 state  模块读写。

执行方式：
    cd streamlitapp && python3 lint_architecture.py
    # 或从项目根目录：
    python3 -m streamlitapp.lint_architecture
"""
from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# --- Rule 1: state.py 限制 ---

STATE_FORBIDDEN_IMPORTS = [
    "car_pricing",
    "car_pricing.api_client",
    "requests",
]

STATE_FORBIDDEN_SUBSTRINGS = [
    "http://",
    "https://",
    "/schema",
    "/predict",
    "/health",
    "/status",
    "/metadata",
]

# --- Rule 2: pages/*.py 限制 ---

PAGES_FORBIDDEN_ATTRIBUTE = "session_state"  # st.session_state 中 session_state


# --- 数据结构 ---

@dataclass
class Violation:
    file: str
    line: int
    code: str
    message: str


@dataclass
class RuleResult:
    name: str
    file: str
    violations: List[Violation] = field(default_factory=list)
    passed: bool = True


# --- 具体检查实现 ---

def _read_source(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


def _iter_source_lines(source: str):
    for i, line in enumerate(source.splitlines(), 1):
        yield i, line.rstrip("\n")


def check_state_module(file_path: Path) -> RuleResult:
    result = RuleResult("state.py 禁止协议/副作用依赖", str(file_path))
    source = _read_source(file_path)
    tree = ast.parse(source, filename=str(file_path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for forbidden in STATE_FORBIDDEN_IMPORTS:
                    if alias.name == forbidden or alias.name.startswith(forbidden + "."):
                        result.violations.append(Violation(
                            file=str(file_path),
                            line=node.lineno,
                            code=source.splitlines()[node.lineno - 1],
                            message=f"禁止 import '{alias.name}'，状态层不能依赖协议层",
                        ))
                        result.passed = False
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for forbidden in STATE_FORBIDDEN_IMPORTS:
                    if node.module == forbidden or node.module.startswith(forbidden + "."):
                        result.violations.append(Violation(
                            file=str(file_path),
                            line=node.lineno,
                            code=source.splitlines()[node.lineno - 1],
                            message=f"禁止 from '{node.module}' import，状态层不能依赖协议层",
                        ))
                        result.passed = False

    for lineno, line in _iter_source_lines(source):
        for substr in STATE_FORBIDDEN_SUBSTRINGS:
            if substr in line and not line.lstrip().startswith("#"):
                # 允许出现在注释里
                result.violations.append(Violation(
                    file=str(file_path),
                    line=lineno,
                    code=line,
                    message=f"禁止出现 '{substr}'，状态层不能维护 URL / endpoint",
                ))
                result.passed = False

    return result


def check_pages_module(file_path: Path) -> RuleResult:
    result = RuleResult(f"pages/*.py 禁止直接 st.session_state", str(file_path))
    source = _read_source(file_path)
    tree = ast.parse(source, filename=str(file_path))

    # st.session_state 是 ast.Attribute: value=Name('st'), attr='session_state'
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == PAGES_FORBIDDEN_ATTRIBUTE
            and isinstance(node.value, ast.Name)
            and node.value.id == "st"
        ):
            line_text = source.splitlines()[node.lineno - 1]
            # 更宽松：只要是 st.session_state 开头的任何访问都禁
            if "st.session_state" in line_text:
                result.violations.append(Violation(
                    file=str(file_path),
                    line=node.lineno,
                    code=line_text,
                    message=(
                        "禁止直接访问 st.session_state，"
                        "必须通过 state 模块的 getter/setter 读写"
                    ),
                ))
                result.passed = False

    return result


# --- 报告输出 ---

def _print_results(results: List[RuleResult]) -> None:
    for r in results:
        icon = "✅ PASS" if r.passed else "❌ FAIL"
        print(f"\n[{icon}] {r.name}")
        print(f"     File: {r.file}")
        for v in r.violations:
            print(f"     - Line {v.line}: {v.message}")
            print(f"         Code: {v.code.strip()}")


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    state_file = script_dir / "state.py"
    pages_dir = script_dir / "pages"

    results: List[RuleResult] = []

    if not state_file.exists():
        print(f"❌ 找不到 {state_file}")
        return 2

    results.append(check_state_module(state_file))

    if not pages_dir.is_dir():
        print(f"❌ 找不到 {pages_dir}/ 目录")
        return 2

    for py_file in sorted(pages_dir.glob("*.py")):
        if py_file.name == "__init__.py":
            continue
        results.append(check_pages_module(py_file))

    _print_results(results)

    total_files = len(results)
    fail_files = sum(1 for r in results if not r.passed)
    total_violations = sum(len(r.violations) for r in results)

    print("\n" + "=" * 60)
    if fail_files == 0:
        print(f"🎉 全部通过：{total_files} 个文件，{total_violations} 条违规")
        return 0
    else:
        print(
            f"❌ 架构违规：{fail_files}/{total_files} 个文件不通过，"
            f"共 {total_violations} 条违规"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
