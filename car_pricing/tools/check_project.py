from __future__ import annotations

import ast
import importlib
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from car_pricing.runtime_config import RuntimeConfig, ArtifactPaths
from car_pricing.feature_schema import FEATURE_ORDER, FIELD_DISPLAY_NAMES


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def format(self, verbose: bool = False) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"[{status}] {self.name}: {self.message}"]
        if verbose and self.details:
            for k, v in self.details.items():
                lines.append(f"       {k}: {v}")
        return "\n".join(lines)


@dataclass
class CheckReport:
    results: List[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.results.append(result)

    @property
    def passed(self) -> List[CheckResult]:
        return [r for r in self.results if r.passed]

    @property
    def failed(self) -> List[CheckResult]:
        return [r for r in self.results if not r.passed]

    def summary(self) -> str:
        total = len(self.results)
        ok = len(self.passed)
        fail = len(self.failed)
        return f"Health Check Summary: {ok}/{total} passed, {fail} failed"

    def format(self, verbose: bool = False) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append("  Car Pricing Project Health Check")
        lines.append("=" * 70)
        lines.append("")
        for r in self.results:
            lines.append(r.format(verbose=verbose))
        lines.append("")
        lines.append("-" * 70)
        lines.append(self.summary())
        lines.append("-" * 70)
        return "\n".join(lines)

    def exit_code(self) -> int:
        return 0 if not self.failed else 1


def _safe_import(module_name: str) -> Tuple[Optional[Any], Optional[str]]:
    try:
        mod = importlib.import_module(module_name)
        return mod, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# 1. Package import boundary checks
# ---------------------------------------------------------------------------

def check_core_package_imports(report: CheckReport) -> None:
    core_modules = [
        "car_pricing",
        "car_pricing.feature_schema",
        "car_pricing.model_runtime",
        "car_pricing.runtime_config",
        "car_pricing.api_client",
        "car_pricing.versioning",
        "car_pricing.model_lineage",
    ]
    for mod_name in core_modules:
        mod, err = _safe_import(mod_name)
        report.add(CheckResult(
            name=f"import.{mod_name}",
            passed=err is None,
            message=f"{mod_name} {'导入成功' if err is None else '导入失败: ' + str(err)}",
        ))


def check_third_party_fastapi_not_shadowed(report: CheckReport, config: RuntimeConfig) -> None:
    repo_root_str = config.repo_root
    legacy_dir = config.legacy_fastapi_dir

    original_path = list(sys.path)
    try:
        if repo_root_str in sys.path:
            sys.path.remove(repo_root_str)
        if legacy_dir in sys.path:
            sys.path.remove(legacy_dir)

        import fastapi as real_fastapi
        real_path = Path(real_fastapi.__file__).resolve()
        is_real_package = "site-packages" in str(real_path) or "dist-packages" in str(real_path)
        report.add(CheckResult(
            name="import.fastapi_not_shadowed",
            passed=is_real_package,
            message=(
                f"fastapi 解析到: {real_path} "
                f"({'第三方包' if is_real_package else '本地目录遮蔽!'})"
            ),
            details={"resolved_path": str(real_path)},
        ))

        if repo_root_str in sys.path:
            sys.path.insert(0, repo_root_str)
            importlib.invalidate_caches()
            try:
                import fastapi as maybe_shadowed
                shadowed_path = Path(maybe_shadowed.__file__).resolve()
                would_shadow = "site-packages" not in str(shadowed_path) and "dist-packages" not in str(shadowed_path)
                report.add(CheckResult(
                    name="import.fastapi_shim_risk",
                    passed=not would_shadow,
                    message=(
                        f"将 repo 根加入 sys.path[0] 后, fastapi 会解析到本地目录 {shadowed_path}"
                        " — 旧 fastapi shim 目录遮蔽第三方包的风险存在"
                        if would_shadow else
                        "即使 repo 根在 sys.path[0], fastapi 仍解析到第三方包"
                    ),
                ))
            except Exception as e:
                report.add(CheckResult(
                    name="import.fastapi_shim_risk",
                    passed=False,
                    message=f"检测遮蔽风险时出错: {e}",
                ))
    finally:
        sys.path[:] = original_path
        importlib.invalidate_caches()


def check_legacy_shim_path_manipulation(report: CheckReport, config: RuntimeConfig) -> None:
    legacy_app = Path(config.legacy_fastapi_dir) / "app.py"
    if not legacy_app.exists():
        report.add(CheckResult(
            name="shim.path_manipulation",
            passed=True,
            message="旧 fastapi shim 目录不存在, 跳过检查",
        ))
        return

    try:
        source = legacy_app.read_text(encoding="utf-8")
        tree = ast.parse(source)
        path_inserts = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (isinstance(func, ast.Attribute)
                        and isinstance(func.value, ast.Attribute)
                        and func.value.attr == "path"
                        and func.attr == "insert"):
                    path_inserts += 1

        report.add(CheckResult(
            name="shim.path_manipulation",
            passed=path_inserts <= 1,
            message=(
                f"旧 fastapi shim app.py 中 sys.path.insert 调用次数={path_inserts}"
                + ("" if path_inserts <= 1 else " — 过多的路径注入可能遮蔽第三方包")
            ),
            details={"path_inserts": path_inserts, "file": str(legacy_app)},
        ))
    except Exception as e:
        report.add(CheckResult(
            name="shim.path_manipulation",
            passed=False,
            message=f"分析旧 shim 时出错: {type(e).__name__}: {e}",
        ))


# ---------------------------------------------------------------------------
# 2. Runtime config + Artifact path checks
# ---------------------------------------------------------------------------

def check_runtime_config(report: CheckReport, config: RuntimeConfig) -> None:
    report.add(CheckResult(
        name="config.runtime_config",
        passed=True,
        message=f"RuntimeConfig 加载成功 (repo_root={config.repo_root})",
        details={
            "model_dir": config.model_dir,
            "data_csv_path": config.data_csv_path,
            "api_base_url": config.api_base_url,
        },
    ))


def check_data_file(report: CheckReport, config: RuntimeConfig) -> None:
    csv_path = Path(config.data_csv_path)
    report.add(CheckResult(
        name="config.data_file",
        passed=csv_path.exists(),
        message=f"训练数据 CSV {'存在' if csv_path.exists() else '缺失'}: {csv_path}",
        details={"path": str(csv_path)},
    ))


def check_artifact_paths(report: CheckReport, config: RuntimeConfig) -> None:
    artifacts = config.artifact_paths()

    report.add(CheckResult(
        name="artifact.model_file_exists",
        passed=artifacts.model_exists(),
        message=f"模型文件 {'存在' if artifacts.model_exists() else '缺失'}: {artifacts.model_path}",
        details={"path": artifacts.model_path},
    ))

    metadata_ok = False
    metadata_content: Dict[str, Any] = {}
    if artifacts.metadata_exists():
        try:
            with open(artifacts.metadata_path, "r", encoding="utf-8") as f:
                metadata_content = json.load(f)
            metadata_ok = True
        except Exception as e:
            report.add(CheckResult(
                name="artifact.metadata_readable",
                passed=False,
                message=f"metadata JSON 解析失败: {e}",
                details={"path": artifacts.metadata_path},
            ))
            return
    else:
        report.add(CheckResult(
            name="artifact.metadata_readable",
            passed=False,
            message=f"metadata 文件缺失: {artifacts.metadata_path}",
            details={"path": artifacts.metadata_path},
        ))
        return

    report.add(CheckResult(
        name="artifact.metadata_readable",
        passed=metadata_ok,
        message=f"metadata 可读，包含字段: {sorted(metadata_content.keys())}",
        details={"path": artifacts.metadata_path},
    ))

    required_meta = {"model_name", "schema_version"}
    missing_meta = required_meta - set(metadata_content.keys())
    report.add(CheckResult(
        name="artifact.metadata_required_fields",
        passed=not missing_meta,
        message=f"metadata 必需字段 {'完整' if not missing_meta else '缺失: ' + str(missing_meta)}",
        details={"missing": list(missing_meta)},
    ))


def check_model_loadable(report: CheckReport, config: RuntimeConfig) -> None:
    from car_pricing.model_runtime import CarPriceModel

    artifacts = config.artifact_paths()
    if not artifacts.model_exists():
        report.add(CheckResult(
            name="artifact.model_loadable",
            passed=False,
            message=f"跳过模型加载检查 (文件缺失): {artifacts.model_path}",
        ))
        return

    try:
        model = CarPriceModel.from_joblib(artifacts.model_path)
        model.schema.validate()
        report.add(CheckResult(
            name="artifact.model_loadable",
            passed=True,
            message=f"模型加载成功 (mode={model.mode}, n_features={model.schema.n_features()})",
            details={
                "mode": model.mode,
                "n_features": model.schema.n_features(),
                "feature_order": model.feature_order,
            },
        ))
        report.add(CheckResult(
            name="artifact.schema_validation",
            passed=True,
            message=f"FeatureSchema 校验通过, schema_version={model.schema.schema_version()}",
        ))
    except Exception as e:
        report.add(CheckResult(
            name="artifact.model_loadable",
            passed=False,
            message=f"模型加载失败: {type(e).__name__}: {e}",
            details={"traceback": traceback.format_exc(limit=2)},
        ))


# ---------------------------------------------------------------------------
# 3. API route contract checks — reuse FastAPI app and Pydantic models
# ---------------------------------------------------------------------------

def _collect_fastapi_routes(app) -> List[Tuple[str, str]]:
    routes = []
    for route in app.routes:
        methods = getattr(route, "methods", None) or {"GET"}
        path = getattr(route, "path", "")
        for m in methods:
            if m in {"HEAD", "OPTIONS"}:
                continue
            routes.append((m, path))
    return sorted(routes)


def check_fastapi_app_imports(report: CheckReport, config: RuntimeConfig) -> None:
    api_dir = str(Path(config.repo_root) / "car_pricing_api")
    original_cwd = os.getcwd()
    try:
        os.chdir(api_dir)
        sys.path.insert(0, api_dir)
        importlib.invalidate_caches()

        app_mod, err = _safe_import("app")
        if err or app_mod is None:
            report.add(CheckResult(
                name="api.app_import",
                passed=False,
                message=f"car_pricing_api/app.py 导入失败: {err}",
            ))
            return

        if not hasattr(app_mod, "app"):
            report.add(CheckResult(
                name="api.app_import",
                passed=False,
                message="app 模块中未找到 FastAPI 实例 `app`",
            ))
            return

        report.add(CheckResult(
            name="api.app_import",
            passed=True,
            message="FastAPI app 导入成功",
        ))

        from car_pricing.api_client import CarPricingClient

        client = CarPricingClient(config=config)
        expected_routes = {
            ("GET", "/"),
            ("GET", "/health"),
            ("GET", "/status"),
            ("GET", "/metadata"),
            ("GET", "/schema"),
            ("POST", "/predict"),
        }
        routes = _collect_fastapi_routes(app_mod.app)
        actual_routes = set(routes)
        missing = expected_routes - actual_routes
        extra = actual_routes - expected_routes

        report.add(CheckResult(
            name="api.routes_contract",
            passed=not missing,
            message=(
                f"路由契约检查: 缺失 {sorted(missing) if missing else '无'}"
                + (f"; 额外路由 {sorted(extra)}" if extra else "")
            ),
            details={
                "expected": sorted(expected_routes),
                "actual": sorted(routes),
                "missing": sorted(missing),
                "extra": sorted(extra),
            },
        ))
    finally:
        os.chdir(original_cwd)
        try:
            sys.path.remove(api_dir)
        except ValueError:
            pass
        importlib.invalidate_caches()


def check_pydantic_schema_consistency(report: CheckReport, config: RuntimeConfig) -> None:
    api_dir = str(Path(config.repo_root) / "car_pricing_api")
    original_cwd = os.getcwd()
    try:
        os.chdir(api_dir)
        sys.path.insert(0, api_dir)
        importlib.invalidate_caches()

        models_mod, err = _safe_import("models")
        if err or models_mod is None:
            report.add(CheckResult(
                name="api.pydantic_consistency",
                passed=False,
                message=f"car_pricing_api/models.py 导入失败: {err}",
            ))
            return

        CarPrediction = getattr(models_mod, "CarPrediction", None)
        if CarPrediction is None:
            report.add(CheckResult(
                name="api.pydantic_consistency",
                passed=False,
                message="models 模块中未找到 CarPrediction",
            ))
            return

        pydantic_fields = list(CarPrediction.__fields__.keys())
        expected_fields = list(FEATURE_ORDER)

        if pydantic_fields == expected_fields:
            report.add(CheckResult(
                name="api.pydantic_consistency",
                passed=True,
                message=f"CarPrediction 字段与 FEATURE_ORDER 完全一致 ({len(pydantic_fields)} 个字段)",
                details={"fields": pydantic_fields},
            ))
        else:
            missing_in_pydantic = [f for f in expected_fields if f not in pydantic_fields]
            extra_in_pydantic = [f for f in pydantic_fields if f not in expected_fields]
            report.add(CheckResult(
                name="api.pydantic_consistency",
                passed=False,
                message=(
                    f"CarPrediction 字段与 FEATURE_ORDER 不一致 — "
                    f"pydantic 缺失: {missing_in_pydantic}, pydantic 多余: {extra_in_pydantic}"
                ),
            ))
    finally:
        os.chdir(original_cwd)
        try:
            sys.path.remove(api_dir)
        except ValueError:
            pass
        importlib.invalidate_caches()


# ---------------------------------------------------------------------------
# 4. Frontend call boundary checks — reuse CarPricingClient + FeatureSchema
# ---------------------------------------------------------------------------

def check_streamlit_boundary(report: CheckReport, config: RuntimeConfig) -> None:
    st_path = Path(config.streamlit_dir) / "streamlit_app.py"
    if not st_path.exists():
        report.add(CheckResult(
            name="frontend.streamlit_exists",
            passed=False,
            message=f"Streamlit app 不存在: {st_path}",
        ))
        return

    report.add(CheckResult(
        name="frontend.streamlit_exists",
        passed=True,
        message=f"Streamlit app 存在: {st_path}",
    ))

    try:
        source = st_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        hardcoded_schemas = []
        imports_api_client = False
        uses_requests_directly = False
        imports_feature_schema = False
        api_endpoints_used = set()
        KNOWN_ENDPOINTS = ("/schema", "/predict", "/health", "/metadata", "/status")

        def _scan_string_for_endpoints(s: str) -> None:
            for ep in KNOWN_ENDPOINTS:
                if ep in s:
                    api_endpoints_used.add(ep)

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module == "car_pricing.api_client":
                    imports_api_client = True
                if node.module and node.module == "car_pricing.feature_schema":
                    imports_feature_schema = True
                if node.module and node.module.startswith("car_pricing"):
                    for alias in node.names:
                        if "client" in alias.name.lower():
                            imports_api_client = True
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "requests":
                        uses_requests_directly = True

            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ("DEFAULT_SCHEMA", "FIELD_DISPLAY_NAMES"):
                        hardcoded_schemas.append(target.id)

            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                _scan_string_for_endpoints(node.value)
            elif isinstance(node, ast.JoinedStr):
                for v in node.values:
                    if isinstance(v, ast.Constant) and isinstance(v.value, str):
                        _scan_string_for_endpoints(v.value)

            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                    if func.value.id in ("re", "requests") and func.attr in ("get", "post"):
                        uses_requests_directly = True

        if hardcoded_schemas:
            report.add(CheckResult(
                name="frontend.hardcoded_schema",
                passed=False,
                message=(
                    f"Streamlit 硬编码了 {hardcoded_schemas}, 应从共享 FeatureSchema 或 /schema 端点获取"
                ),
                details={"hardcoded_vars": hardcoded_schemas},
            ))
        else:
            report.add(CheckResult(
                name="frontend.hardcoded_schema",
                passed=True,
                message="Streamlit 未发现硬编码 schema",
            ))

        expected_endpoints = {"/schema", "/predict"}
        if imports_api_client and not api_endpoints_used:
            import car_pricing.api_client as _ac
            ac_source = Path(_ac.__file__).read_text(encoding="utf-8")
            ac_tree = ast.parse(ac_source)
            for node in ast.walk(ac_tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    _scan_string_for_endpoints(node.value)
                elif isinstance(node, ast.JoinedStr):
                    for v in node.values:
                        if isinstance(v, ast.Constant) and isinstance(v.value, str):
                            _scan_string_for_endpoints(v.value)
        missing_eps = expected_endpoints - api_endpoints_used
        report.add(CheckResult(
            name="frontend.api_endpoints",
            passed=not missing_eps,
            message=(
                f"Streamlit 使用的 API 端点: {sorted(api_endpoints_used)}"
                + (f"; 未使用: {sorted(missing_eps)}" if missing_eps else "")
            ),
            details={"used": sorted(api_endpoints_used), "missing": sorted(missing_eps)},
        ))

        shared_client_ok = imports_api_client and not uses_requests_directly
        if shared_client_ok:
            client_msg = "Streamlit 通过 car_pricing.api_client 共享客户端进行 API 调用"
        elif uses_requests_directly:
            client_msg = "Streamlit 直接使用 requests 调用 API, 建议抽离共享 client 统一维护端点与契约"
        else:
            client_msg = "Streamlit 未检测到 API 调用方式"
        report.add(CheckResult(
            name="frontend.shared_client",
            passed=shared_client_ok,
            message=client_msg,
            details={
                "imports_api_client": imports_api_client,
                "uses_requests_directly": uses_requests_directly,
            },
        ))

        uses_shared_display_names = imports_feature_schema and "FIELD_DISPLAY_NAMES" not in hardcoded_schemas
        report.add(CheckResult(
            name="frontend.shared_display_names",
            passed=uses_shared_display_names,
            message=(
                "Streamlit 使用 car_pricing.feature_schema.FIELD_DISPLAY_NAMES"
                if uses_shared_display_names else
                "Streamlit 未使用共享 FIELD_DISPLAY_NAMES"
            ),
        ))

    except Exception as e:
        report.add(CheckResult(
            name="frontend.ast_analysis",
            passed=False,
            message=f"分析 Streamlit 源码失败: {type(e).__name__}: {e}",
            details={"traceback": traceback.format_exc(limit=2)},
        ))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_checks(config: Optional[RuntimeConfig] = None) -> CheckReport:
    if config is None:
        config = RuntimeConfig.from_env()

    report = CheckReport()

    checks: List[Tuple[str, Callable]] = [
        ("core_imports", lambda: check_core_package_imports(report)),
        ("runtime_config", lambda: check_runtime_config(report, config)),
        ("data_file", lambda: check_data_file(report, config)),
        ("artifact_paths", lambda: check_artifact_paths(report, config)),
        ("model_loadable", lambda: check_model_loadable(report, config)),
        ("fastapi_shadow", lambda: check_third_party_fastapi_not_shadowed(report, config)),
        ("shim_path", lambda: check_legacy_shim_path_manipulation(report, config)),
        ("fastapi_app", lambda: check_fastapi_app_imports(report, config)),
        ("pydantic_consistency", lambda: check_pydantic_schema_consistency(report, config)),
        ("streamlit_boundary", lambda: check_streamlit_boundary(report, config)),
    ]

    for name, fn in checks:
        try:
            fn()
        except Exception as e:
            report.add(CheckResult(
                name=f"check.{name}",
                passed=False,
                message=f"检查函数异常: {type(e).__name__}: {e}",
                details={"traceback": traceback.format_exc(limit=3)},
            ))

    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m car_pricing.tools.check_project",
        description="Car Pricing 项目工程健康检查",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细信息")
    parser.add_argument("--json", action="store_true", dest="as_json", help="以 JSON 格式输出")
    parser.add_argument("--repo-root", default=None, help="项目根目录 (默认自动检测)")
    args = parser.parse_args()

    config = RuntimeConfig.from_env(repo_root=args.repo_root)
    report = run_all_checks(config)

    if args.as_json:
        out = {
            "summary": {
                "total": len(report.results),
                "passed": len(report.passed),
                "failed": len(report.failed),
            },
            "checks": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "details": r.details,
                }
                for r in report.results
            ],
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(report.format(verbose=args.verbose))

    return report.exit_code()


if __name__ == "__main__":
    sys.exit(main())
