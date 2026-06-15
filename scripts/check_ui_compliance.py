#!/usr/bin/env python3
"""UI 入口合规检查脚本。

检查 streamlitapp/ 和 androidapp/ 下的 Python 文件，确保不出现：
  1. `import requests` 或 `from requests import ...`
  2. `http://` 或 `https://` URL 字面量（注释/文档字符串除外）
  3. 手动校验响应字段：`if "xxx" not in data`、`if data.get("xxx") is None` 等
  4. 硬编码字段清单：`FEATURE_ORDER = [...]`、`feature_order = [...]`、字段名 list 字面量
  5. 环境变量读取部署配置：`os.environ.get("API_*")`、`os.environ["API_*"]`

用法:
    python scripts/check_ui_compliance.py
    python scripts/check_ui_compliance.py --verbose

退出码: 0=合规，1=违规
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
UI_PATHS = [
    PROJECT_ROOT / "streamlitapp",
    PROJECT_ROOT / "androidapp",
]

PYTHON_GLOB = "**/*.py"


class Violation:
    def __init__(self, rule: str, file_path: Path, lineno: int, detail: str):
        self.rule = rule
        self.file_path = file_path
        self.lineno = lineno
        self.detail = detail

    def __str__(self) -> str:
        rel = self.file_path.relative_to(PROJECT_ROOT)
        return f"[{self.rule}] {rel}:{self.lineno}: {self.detail}"


# ---------- 检查规则 ----------

def check_requests_imports(tree: ast.AST, file_path: Path) -> List[Violation]:
    """禁止直接 import requests。"""
    violations: List[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "requests" or alias.name.startswith("requests."):
                    violations.append(Violation(
                        rule="NO_REQUESTS_IMPORT",
                        file_path=file_path,
                        lineno=node.lineno,
                        detail=f"`import {alias.name}` 应通过 car_pricing.api_client 调用",
                    ))
        elif isinstance(node, ast.ImportFrom):
            if node.module and (node.module == "requests" or node.module.startswith("requests.")):
                violations.append(Violation(
                    rule="NO_REQUESTS_IMPORT",
                    file_path=file_path,
                    lineno=node.lineno,
                    detail=f"`from {node.module} import ...` 应通过 car_pricing.api_client 调用",
                ))
    return violations


def check_url_literals(tree: ast.AST, file_path: Path) -> List[Violation]:
    """禁止 http:// / https:// 字面量（注释/文档字符串除外）。"""
    violations: List[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            s = node.value.strip()
            if s.startswith("http://") or s.startswith("https://"):
                violations.append(Violation(
                    rule="NO_URL_LITERAL",
                    file_path=file_path,
                    lineno=getattr(node, "lineno", 0),
                    detail=f"URL 字面量 {s!r} 应下沉到 car_pricing.api_client.create_client()",
                ))
    return violations


def check_response_validation(tree: ast.AST, file_path: Path) -> List[Violation]:
    """禁止客户端侧手动校验响应字段（应下沉 client 层）。

    检查模式:
      - `if "xxx" not in data`
      - `if data.get("xxx") is None`
      - `xxx = data["xxx"]` （不拦，这是正常取 dict 值）
    """
    violations: List[Violation] = []

    for node in ast.walk(tree):
        # `if "prediction" not in body`
        if isinstance(node, ast.Compare):
            if any(isinstance(op, ast.NotIn) for op in node.ops):
                if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
                    field_name = node.left.value
                    violations.append(Violation(
                        rule="NO_RESPONSE_VALIDATION",
                        file_path=file_path,
                        lineno=node.lineno,
                        detail=f"手动检查字段 {field_name!r} 存在性，应下沉 client 层",
                    ))

        # `if body.get("prediction") is None`
        if isinstance(node, ast.If):
            test = node.test
            if isinstance(test, ast.Compare):
                for comp in test.ops:
                    if isinstance(comp, ast.Is):
                        left = test.left
                        if (isinstance(left, ast.Call) and isinstance(left.func, ast.Attribute)
                                and left.func.attr == "get"):
                            field_name = None
                            if left.args and isinstance(left.args[0], ast.Constant):
                                field_name = left.args[0].value
                            violations.append(Violation(
                                rule="NO_RESPONSE_VALIDATION",
                                file_path=file_path,
                                lineno=node.lineno,
                                detail=f"手动检查字段 {field_name!r} 是否为 None，应下沉 client 层",
                            ))

    return violations


def check_hardcoded_feature_lists(tree: ast.AST, file_path: Path) -> List[Violation]:
    """禁止硬编码特征清单。

    检查模式:
      - 赋值语句右侧是包含特征名字符串的 list 字面量
      - 变量名包含 `feature_order` / `numeric_features` / `categorical_features`
    """
    violations: List[Violation] = []
    feature_names = {"enginesize", "curbweight", "horsepower", "highwaympg",
                     "carwidth", "wheelbase", "drivewheel", "citympg",
                     "boreratio", "cylindernumber", "price"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id.lower()
                    # 变量名是特征清单相关
                    if any(k in name for k in ["feature_order", "numeric_feature",
                                                "categorical_feature", "field_list"]):
                        value = node.value
                        if isinstance(value, ast.List) and value.elts:
                            # 检查 list 元素是否是特征名字符串
                            feature_str_count = 0
                            for elt in value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    if elt.value in feature_names:
                                        feature_str_count += 1
                            if feature_str_count >= 2:
                                violations.append(Violation(
                                    rule="NO_HARDCODED_FEATURES",
                                    file_path=file_path,
                                    lineno=node.lineno,
                                    detail=f"硬编码特征清单 `{target.id}`，应从 client/schema 获取",
                                ))

    return violations


def check_deployment_env_vars(tree: ast.AST, file_path: Path) -> List[Violation]:
    """禁止 UI 层读取 API_* 环境变量（应下沉 client 层）。"""
    violations: List[Violation] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            # os.environ.get("API_xxx")
            if (isinstance(func, ast.Attribute) and func.attr == "get"
                    and isinstance(func.value, ast.Attribute)
                    and isinstance(func.value.value, ast.Name)
                    and func.value.value.id == "os"
                    and func.value.attr == "environ"):
                if node.args and isinstance(node.args[0], ast.Constant):
                    env_name = node.args[0].value
                    if isinstance(env_name, str) and env_name.startswith("API_"):
                        violations.append(Violation(
                            rule="NO_DEPLOYMENT_ENV_VARS",
                            file_path=file_path,
                            lineno=node.lineno,
                            detail=f"读取部署环境变量 {env_name!r}，应下沉 car_pricing.api_client.create_client()",
                        ))

            # os.environ["API_xxx"]
            elif (isinstance(func, ast.Subscript)
                    and isinstance(func.value, ast.Attribute)
                    and isinstance(func.value.value, ast.Name)
                    and func.value.value.id == "os"
                    and func.value.attr == "environ"):
                subscript = func.slice
                if isinstance(subscript, ast.Constant) and isinstance(subscript.value, str):
                    env_name = subscript.value
                    if env_name.startswith("API_"):
                        violations.append(Violation(
                            rule="NO_DEPLOYMENT_ENV_VARS",
                            file_path=file_path,
                            lineno=node.lineno,
                            detail=f"读取部署环境变量 {env_name!r}，应下沉 car_pricing.api_client.create_client()",
                        ))

    return violations


# ---------- 主流程 ----------

ALL_CHECKS = [
    check_requests_imports,
    check_url_literals,
    check_response_validation,
    check_hardcoded_feature_lists,
    check_deployment_env_vars,
]

RULE_DESCRIPTIONS: Dict[str, str] = {
    "NO_REQUESTS_IMPORT": "禁止直接 import requests，应通过 car_pricing.api_client 调用",
    "NO_URL_LITERAL": "禁止 URL 字面量，应下沉到 car_pricing.api_client.create_client()",
    "NO_RESPONSE_VALIDATION": "禁止客户端侧校验响应字段，应下沉到 client 层",
    "NO_HARDCODED_FEATURES": "禁止硬编码特征清单，应从 client/schema 获取",
    "NO_DEPLOYMENT_ENV_VARS": "禁止读取 API_* 部署配置环境变量，应下沉 create_client()",
}


def scan_file(file_path: Path) -> List[Violation]:
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file_path))
    violations: List[Violation] = []
    for check_fn in ALL_CHECKS:
        violations.extend(check_fn(tree, file_path))
    return violations


def scan_directory(dir_path: Path) -> List[Violation]:
    violations: List[Violation] = []
    for py_file in dir_path.glob(PYTHON_GLOB):
        violations.extend(scan_file(py_file))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="UI 入口合规检查")
    parser.add_argument("--verbose", "-v", action="store_true", help="输出详细规则说明")
    args = parser.parse_args()

    if args.verbose:
        print("=" * 70)
        print("UI 入口合规检查规则")
        print("=" * 70)
        for rule, desc in RULE_DESCRIPTIONS.items():
            print(f"  {rule}: {desc}")
        print()

    all_violations: List[Violation] = []
    for ui_path in UI_PATHS:
        if ui_path.exists():
            all_violations.extend(scan_directory(ui_path))

    if not all_violations:
        print("✅ 所有 UI 入口合规，未发现违规。")
        return 0

    print(f"❌ 发现 {len(all_violations)} 处违规:")
    print("-" * 70)
    for v in sorted(all_violations, key=lambda x: (str(x.file_path), x.lineno)):
        print(f"  {v}")
    print("-" * 70)
    if args.verbose:
        for rule in sorted({v.rule for v in all_violations}):
            print(f"  {rule}: {RULE_DESCRIPTIONS[rule]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
