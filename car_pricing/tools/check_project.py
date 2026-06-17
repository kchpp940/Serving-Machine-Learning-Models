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

from car_pricing.feature_schema import FEATURE_ORDER


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


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
        "car_pricing.versioning",
        "car_pricing.model_lineage",
    ]
    for mod_name in core_modules:
        _, err = _safe_import(mod_name)
        report.add(CheckResult(
            name=f"import.{mod_name}",
            passed=err is None,
            message=f"{mod_name} {'导入成功' if err is None else '导入失败: ' + str(err)}",
        ))


def check_third_party_fastapi_not_shadowed(report: CheckReport) -> None:
    repo_root_str = str(REPO_ROOT)
    legacy_dir = str(REPO_ROOT / "fastapi")

    original_path = list(sys.path)
    try:
        for p in (repo_root_str, legacy_dir):
            if p in sys.path:
                sys.path.remove(p)

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
                    "即使 repo 根在 sys.path[0], fastapi 仍解析到第三方包"
                    if not would_shadow else
                    f"repo 根加入 sys.path[0] 后, fastapi 会被本地目录遮蔽 ({shadowed_path})"
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


def check_legacy_shim_path_manipulation(report: CheckReport) -> None:
    legacy_app = REPO_ROOT / "fastapi" / "app.py"
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


def check_api_error_contract(report: CheckReport) -> None:
    api_client_mod, err = _safe_import("car_pricing.api_client")
    if err or api_client_mod is None:
        report.add(CheckResult(
            name="api_client.error_type_import",
            passed=False,
            message=f"无法导入 car_pricing.api_client: {err}",
        ))
        return

    ApiError = getattr(api_client_mod, "ApiError", None)
    if ApiError is None:
        report.add(CheckResult(
            name="api_client.error_type_import",
            passed=False,
            message="car_pricing.api_client 中未导出 ApiError",
        ))
        return

    report.add(CheckResult(
        name="api_client.error_type_import",
        passed=True,
        message="ApiError 从 car_pricing.api_client 正常导出",
    ))

    required_fields = {"message", "source", "status_code", "detail", "context"}
    required_methods = {"display"}

    actual_fields = set()
    if hasattr(ApiError, "__dataclass_fields__"):
        actual_fields = set(ApiError.__dataclass_fields__.keys())
    actual_methods = {m for m in dir(ApiError) if not m.startswith("_") and callable(getattr(ApiError, m, None))}

    missing_fields = required_fields - actual_fields
    missing_methods = required_methods - actual_methods

    report.add(CheckResult(
        name="api_client.error_type_structure",
        passed=not missing_fields and not missing_methods,
        message=(
            "ApiError 契约字段和方法完整"
            if not missing_fields and not missing_methods else
            f"ApiError 契约不完整: 缺失字段 {sorted(missing_fields)}, 缺失方法 {sorted(missing_methods)}"
        ),
        details={
            "fields": sorted(actual_fields),
            "methods": sorted(actual_methods),
            "missing_fields": sorted(missing_fields),
            "missing_methods": sorted(missing_methods),
        },
    ))

    CarPricingClient = getattr(api_client_mod, "CarPricingClient", None)
    if CarPricingClient is None:
        return

    methods_to_check = ["fetch_schema", "predict", "health", "metadata", "status"]
    contract_ok = True
    contract_details = {}
    for method_name in methods_to_check:
        method = getattr(CarPricingClient, method_name, None)
        if method is None:
            contract_ok = False
            contract_details[method_name] = "method missing"
            continue
        annotations = getattr(method, "__annotations__", {})
        return_ann = annotations.get("return", None)
        if return_ann is None:
            contract_ok = False
            contract_details[method_name] = "missing return annotation"
            continue
        return_str = str(return_ann)
        if "ApiError" not in return_str:
            contract_ok = False
            contract_details[method_name] = f"return type does not include ApiError: {return_str}"
        else:
            contract_details[method_name] = "OK"

    report.add(CheckResult(
        name="api_client.client_return_contract",
        passed=contract_ok,
        message=(
            "CarPricingClient 所有方法返回类型均包含 ApiError"
            if contract_ok else
            "CarPricingClient 部分方法返回契约不包含 ApiError"
        ),
        details=contract_details,
    ))


# ---------------------------------------------------------------------------
# 2. Artifact checks — reuse ModelService for paths and loading
# ---------------------------------------------------------------------------

def _import_model_service():
    api_dir = str(REPO_ROOT / "car_pricing_api")
    sys.path.insert(0, api_dir)
    sys.path.insert(0, str(REPO_ROOT))
    try:
        importlib.invalidate_caches()
        services_mod, err = _safe_import("services.model_service")
        if err or services_mod is None:
            return None, err
        return services_mod.ModelService, None
    finally:
        try:
            sys.path.remove(api_dir)
        except ValueError:
            pass
        try:
            sys.path.remove(str(REPO_ROOT))
        except ValueError:
            pass


def check_data_file(report: CheckReport) -> None:
    csv_path = REPO_ROOT / "Data" / "cars.csv"
    report.add(CheckResult(
        name="config.data_file",
        passed=csv_path.exists(),
        message=f"训练数据 CSV {'存在' if csv_path.exists() else '缺失'}: {csv_path}",
        details={"path": str(csv_path)},
    ))


def check_artifact_paths_via_model_service(report: CheckReport) -> None:
    ModelService, err = _import_model_service()
    if err or ModelService is None:
        report.add(CheckResult(
            name="artifact.model_service_import",
            passed=False,
            message=f"无法导入 ModelService: {err}",
        ))
        return

    report.add(CheckResult(
        name="artifact.model_service_import",
        passed=True,
        message="ModelService 导入成功",
    ))

    service = ModelService()

    model_path = Path(service._model_path)
    metadata_path = Path(service._metadata_path)

    report.add(CheckResult(
        name="artifact.model_file_exists",
        passed=model_path.exists(),
        message=f"模型文件 {'存在' if model_path.exists() else '缺失'}: {model_path}",
        details={"path": str(model_path), "source": "ModelService._model_path"},
    ))

    if not metadata_path.exists():
        report.add(CheckResult(
            name="artifact.metadata_readable",
            passed=False,
            message=f"metadata 文件缺失: {metadata_path}",
            details={"path": str(metadata_path), "source": "ModelService._metadata_path"},
        ))
        return

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_content = json.load(f)
    except Exception as e:
        report.add(CheckResult(
            name="artifact.metadata_readable",
            passed=False,
            message=f"metadata JSON 解析失败: {e}",
            details={"path": str(metadata_path)},
        ))
        return

    report.add(CheckResult(
        name="artifact.metadata_readable",
        passed=True,
        message=f"metadata 可读, 字段: {sorted(metadata_content.keys())}",
        details={"path": str(metadata_path)},
    ))

    required_meta = {"model_name", "schema_version"}
    missing_meta = required_meta - set(metadata_content.keys())
    report.add(CheckResult(
        name="artifact.metadata_required_fields",
        passed=not missing_meta,
        message=f"metadata 必需字段 {'完整' if not missing_meta else '缺失: ' + str(missing_meta)}",
        details={"missing": list(missing_meta)},
    ))


def check_model_loadable(report: CheckReport) -> None:
    from car_pricing.model_runtime import CarPriceModel

    ModelService, _ = _import_model_service()
    if ModelService is None:
        report.add(CheckResult(
            name="artifact.model_loadable",
            passed=False,
            message="跳过模型加载检查 (ModelService 不可用)",
        ))
        return

    service = ModelService()
    model_path = service._model_path
    if not Path(model_path).exists():
        report.add(CheckResult(
            name="artifact.model_loadable",
            passed=False,
            message=f"跳过模型加载检查 (文件缺失): {model_path}",
        ))
        return

    try:
        model = CarPriceModel.from_joblib(model_path)
        model.schema.validate()
        report.add(CheckResult(
            name="artifact.model_loadable",
            passed=True,
            message=f"模型加载成功 (mode={model.mode}, n_features={model.schema.n_features()})",
            details={
                "mode": model.mode,
                "n_features": model.schema.n_features(),
                "feature_order": model.feature_order,
                "source": "CarPriceModel.from_joblib(ModelService._model_path)",
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
# 3. API route contract checks — derive contract from FastAPI app.routes
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


def _import_fastapi_app():
    api_dir = str(REPO_ROOT / "car_pricing_api")
    original_cwd = os.getcwd()
    try:
        os.chdir(api_dir)
        sys.path.insert(0, api_dir)
        sys.path.insert(0, str(REPO_ROOT))
        importlib.invalidate_caches()
        app_mod, err = _safe_import("app")
        if err or app_mod is None:
            return None, err
        if not hasattr(app_mod, "app"):
            return None, "app 模块中未找到 FastAPI 实例 `app`"
        return app_mod.app, None
    finally:
        os.chdir(original_cwd)
        for p in (api_dir, str(REPO_ROOT)):
            try:
                sys.path.remove(p)
            except ValueError:
                pass
        importlib.invalidate_caches()


def check_fastapi_app_and_contract(report: CheckReport) -> None:
    app, err = _import_fastapi_app()
    if err or app is None:
        report.add(CheckResult(
            name="api.app_import",
            passed=False,
            message=f"car_pricing_api/app.py 导入失败: {err}",
        ))
        return

    report.add(CheckResult(
        name="api.app_import",
        passed=True,
        message="FastAPI app 导入成功",
    ))

    contract_routes = _collect_fastapi_routes(app)
    contract_set = set(contract_routes)

    report.add(CheckResult(
        name="api.routes_contract_derived",
        passed=True,
        message=f"从 FastAPI app.routes 导出契约路由 {len(contract_set)} 条",
        details={"contract_routes": contract_routes},
    ))

    expected_from_routers = set()
    api_dir = str(REPO_ROOT / "car_pricing_api")
    original_cwd = os.getcwd()
    try:
        os.chdir(api_dir)
        sys.path.insert(0, api_dir)
        sys.path.insert(0, str(REPO_ROOT))
        importlib.invalidate_caches()
        for router_mod_name in ("routers.meta", "routers.prediction", "routers.schema"):
            mod, merr = _safe_import(router_mod_name)
            if merr or mod is None:
                continue
            router = getattr(mod, "router", None)
            if router is None:
                continue
            for r in router.routes:
                methods = getattr(r, "methods", None) or {"GET"}
                path = getattr(r, "path", "")
                prefix = getattr(router, "prefix", "")
                for m in methods:
                    if m in {"HEAD", "OPTIONS"}:
                        continue
                    expected_from_routers.add((m, prefix + path))
    finally:
        os.chdir(original_cwd)
        for p in (api_dir, str(REPO_ROOT)):
            try:
                sys.path.remove(p)
            except ValueError:
                pass
        importlib.invalidate_caches()

    app_decorated = set()
    for r in app.routes:
        if getattr(r, "name", None) in {"running", "favicon"}:
            methods = getattr(r, "methods", None) or {"GET"}
            path = getattr(r, "path", "")
            for m in methods:
                if m in {"HEAD", "OPTIONS"}:
                    continue
                app_decorated.add((m, path))

    auto_docs = {("GET", "/docs"), ("GET", "/redoc"), ("GET", "/openapi.json"), ("GET", "/docs/oauth2-redirect")}
    actual_business = contract_set - auto_docs
    full_expected = expected_from_routers | app_decorated
    missing_in_app = full_expected - actual_business
    extra_in_app = actual_business - full_expected

    report.add(CheckResult(
        name="api.routes_contract_stable",
        passed=not missing_in_app,
        message=(
            "app.routes 与 router 注册的路由完全一致"
            if not missing_in_app and not extra_in_app else
            f"路由契约不一致: 缺失 {sorted(missing_in_app)}, 额外 {sorted(extra_in_app)}"
        ),
        details={
            "router_expected": sorted(expected_from_routers),
            "app_decorated": sorted(app_decorated),
            "app_actual_business": sorted(actual_business),
            "auto_docs_excluded": sorted(auto_docs),
            "missing": sorted(missing_in_app),
            "extra": sorted(extra_in_app),
        },
    ))


def check_pydantic_schema_consistency(report: CheckReport) -> None:
    api_dir = str(REPO_ROOT / "car_pricing_api")
    original_cwd = os.getcwd()
    try:
        os.chdir(api_dir)
        sys.path.insert(0, api_dir)
        sys.path.insert(0, str(REPO_ROOT))
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
                message=f"CarPrediction 字段与 FeatureSchema.FEATURE_ORDER 完全一致 ({len(pydantic_fields)} 个字段)",
                details={"fields": pydantic_fields, "source": "FeatureSchema.FEATURE_ORDER"},
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
        for p in (api_dir, str(REPO_ROOT)):
            try:
                sys.path.remove(p)
            except ValueError:
                pass
        importlib.invalidate_caches()


# ---------------------------------------------------------------------------
# 4. Frontend call boundary checks
# ---------------------------------------------------------------------------

def check_streamlit_boundary(report: CheckReport) -> None:
    st_path = REPO_ROOT / "streamlitapp" / "streamlit_app.py"
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
        imports_shared_client = False
        imports_shared_schema = False
        uses_requests_directly = False
        api_endpoints_used = set()
        KNOWN_ENDPOINTS = ("/schema", "/predict", "/health", "/metadata", "/status")

        def _scan_string_for_endpoints(s: str) -> None:
            for ep in KNOWN_ENDPOINTS:
                if ep in s:
                    api_endpoints_used.add(ep)

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("car_pricing"):
                    for alias in node.names:
                        if "client" in alias.name.lower() or "Client" in alias.name:
                            imports_shared_client = True
                        if "schema" in alias.name.lower() or "Schema" in alias.name or "FEATURE" in alias.name or "FIELD_" in alias.name:
                            imports_shared_schema = True
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

        report.add(CheckResult(
            name="frontend.hardcoded_schema",
            passed=not hardcoded_schemas,
            message=(
                "Streamlit 未发现硬编码 schema/field_names"
                if not hardcoded_schemas else
                f"Streamlit 硬编码了 {hardcoded_schemas}, 应从共享 FeatureSchema 或 /schema 端点获取"
            ),
            details={"hardcoded_vars": hardcoded_schemas},
        ))

        report.add(CheckResult(
            name="frontend.shared_schema_import",
            passed=imports_shared_schema,
            message=(
                "Streamlit 从 car_pricing 导入共享 schema 常量"
                if imports_shared_schema else
                "Streamlit 未从 car_pricing 导入共享 schema/字段常量"
            ),
        ))

        report.add(CheckResult(
            name="frontend.shared_client",
            passed=imports_shared_client and not uses_requests_directly,
            message=(
                "Streamlit 通过 car_pricing 共享客户端进行 API 调用"
                if imports_shared_client and not uses_requests_directly else
                ("Streamlit 直接使用 requests 调用 API, 建议抽离共享 client 统一维护端点与契约"
                 if uses_requests_directly else
                 "Streamlit 未使用 requests 也未使用共享 client")
            ),
            details={
                "imports_shared_client": imports_shared_client,
                "uses_requests_directly": uses_requests_directly,
            },
        ))

        expected_endpoints = {"/schema", "/predict"}
        if imports_shared_client and not api_endpoints_used:
            from car_pricing import api_client as _ac_mod
            ac_source = Path(_ac_mod.__file__).read_text(encoding="utf-8")
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
            name="frontend.api_endpoints_used",
            passed=not missing_eps,
            message=(
                f"Streamlit 引用的 API 端点: {sorted(api_endpoints_used)}"
                + (f"; 未引用: {sorted(missing_eps)}" if missing_eps else "")
            ),
            details={"used": sorted(api_endpoints_used), "missing": sorted(missing_eps)},
        ))

        imports_api_error = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("car_pricing"):
                    for alias in node.names:
                        if alias.name == "ApiError":
                            imports_api_error = True

        uses_isinstance_apierror = False
        structured_attr_accesses: Dict[str, int] = {}
        string_errors: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "isinstance":
                    for arg in node.args:
                        if isinstance(arg, ast.Name) and arg.id == "ApiError":
                            uses_isinstance_apierror = True
                        if isinstance(arg, ast.Attribute) and arg.attr == "ApiError":
                            uses_isinstance_apierror = True
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id in ("err", "schema_error", "error", "api_err"):
                    structured_attr_accesses[node.attr] = structured_attr_accesses.get(node.attr, 0) + 1
            if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.IsNot):
                if isinstance(node.left, ast.Name) and node.left.id in ("err", "schema_error", "error"):
                    if isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value is None:
                        string_errors.append(f"使用 `{node.left.id} is not None` 作为错误判定, 建议 isinstance(..., ApiError)")

        structured_attr_required = {"message", "source", "status_code", "detail", "display"}
        used_structured = set(structured_attr_accesses.keys()) & structured_attr_required

        report.add(CheckResult(
            name="frontend.structured_error_import",
            passed=imports_api_error,
            message=(
                "Streamlit 从 car_pricing 导入了 ApiError"
                if imports_api_error else
                "Streamlit 未导入 ApiError, 无法消费结构化错误"
            ),
        ))

        report.add(CheckResult(
            name="frontend.structured_error_usage",
            passed=uses_isinstance_apierror and bool(used_structured) and not string_errors,
            message=(
                (
                    "Streamlit 使用 isinstance(..., ApiError) 判定错误"
                    f" 并访问结构化属性: {sorted(used_structured)}"
                )
                if uses_isinstance_apierror and bool(used_structured) and not string_errors else
                (
                    "Streamlit 未正确消费结构化错误: "
                    + ("缺少 isinstance(ApiError); " if not uses_isinstance_apierror else "")
                    + ("缺少结构化属性访问; " if not used_structured else f"已访问 {sorted(used_structured)}; ")
                    + (f"检测到字符串判定: {string_errors}" if string_errors else "")
                )
            ),
            details={
                "uses_isinstance_apierror": uses_isinstance_apierror,
                "structured_accesses": structured_attr_accesses,
                "string_error_patterns": string_errors,
            },
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

ALL_CHECKS: List[Callable[[CheckReport], None]] = [
    check_core_package_imports,
    check_data_file,
    check_artifact_paths_via_model_service,
    check_model_loadable,
    check_third_party_fastapi_not_shadowed,
    check_legacy_shim_path_manipulation,
    check_api_error_contract,
    check_fastapi_app_and_contract,
    check_pydantic_schema_consistency,
    check_streamlit_boundary,
]


def run_all_checks() -> CheckReport:
    report = CheckReport()
    for fn in ALL_CHECKS:
        try:
            fn(report)
        except Exception as e:
            report.add(CheckResult(
                name=f"check.{fn.__name__}",
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
    args = parser.parse_args()

    report = run_all_checks()

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
