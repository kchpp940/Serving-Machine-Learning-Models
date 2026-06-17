from __future__ import annotations

import ast
import importlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class CheckResult:
    name: str
    category: str
    passed: bool
    message: str
    details: List[str] = field(default_factory=list)

    def format(self, verbose: bool = False) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"[{status}] {self.category} :: {self.name} — {self.message}"]
        if verbose and self.details:
            for d in self.details:
                lines.append(f"       -> {d}")
        return "\n".join(lines)


class ProjectChecker:
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or PROJECT_ROOT
        self.results: List[CheckResult] = []

    def _record(self, result: CheckResult) -> CheckResult:
        self.results.append(result)
        return result

    def run_all(self, verbose: bool = False) -> int:
        self.results = []
        checkers = [
            ("package", self.check_package_imports),
            ("artifacts", self.check_artifacts),
            ("model_runtime", self.check_model_runtime),
            ("api_contract", self.check_api_contract),
            ("streamlit_boundary", self.check_streamlit_boundary),
            ("fastapi_shim", self.check_fastapi_shim),
        ]
        for category, fn in checkers:
            try:
                fn()
            except Exception as e:
                self._record(CheckResult(
                    name=fn.__name__,
                    category=category,
                    passed=False,
                    message=f"检查器异常: {type(e).__name__}: {e}",
                ))
        self._print_report(verbose)
        return 0 if all(r.passed for r in self.results) else 1

    def _print_report(self, verbose: bool) -> None:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        print("=" * 70)
        print(f"car_pricing 工程健康检查  —  {passed}/{total} 通过, {failed} 失败")
        print("=" * 70)
        for r in self.results:
            print(r.format(verbose=verbose))
        print("=" * 70)
        if failed:
            print(f"共 {failed} 项未通过，请根据上面的 FAIL 记录修复。")
        else:
            print("所有检查项通过，工程配置完整。")

    # ------------------------------------------------------------------
    # 1. 包导入边界
    # ------------------------------------------------------------------
    def check_package_imports(self) -> None:
        self._check_core_package_import()
        self._check_all_exports_consistent()

    def _check_core_package_import(self) -> None:
        details: List[str] = []
        ok = True
        modules = [
            "car_pricing",
            "car_pricing.feature_schema",
            "car_pricing.model_runtime",
            "car_pricing.versioning",
            "car_pricing.model_lineage",
        ]
        for mod_name in modules:
            try:
                importlib.import_module(mod_name)
                details.append(f"导入成功: {mod_name}")
            except Exception as e:
                ok = False
                details.append(f"导入失败: {mod_name} — {type(e).__name__}: {e}")
        self._record(CheckResult(
            name="core_package_import",
            category="package",
            passed=ok,
            message="核心包及子模块可导入" if ok else "存在模块无法导入",
            details=details,
        ))

    def _check_all_exports_consistent(self) -> None:
        import car_pricing
        import types
        declared = set(car_pricing.__all__)
        actual = {
            name for name in dir(car_pricing)
            if not name.startswith("_") and not isinstance(getattr(car_pricing, name, None), types.ModuleType)
        }
        missing = declared - actual
        extra = actual - declared
        details = []
        if missing:
            details.append(f"__all__ 声明但未实际导出: {sorted(missing)}")
        if extra:
            details.append(f"实际导出但未在 __all__ 中声明: {sorted(extra)}")
        self._record(CheckResult(
            name="all_exports_consistent",
            category="package",
            passed=not missing and not extra,
            message="__all__ 与实际导出一致" if not (missing or extra) else "__all__ 与实际导出不一致",
            details=details,
        ))

    # ------------------------------------------------------------------
    # 2. Artifact 路径
    # ------------------------------------------------------------------
    def check_artifacts(self) -> None:
        self._check_models_directory()
        self._check_model_metadata_readable()

    def _models_dir(self) -> Path:
        return self.project_root / "car_pricing_api" / "models"

    def _check_models_directory(self) -> None:
        models_dir = self._models_dir()
        model_path = models_dir / "sklearn_gbr.pkl"
        metadata_path = models_dir / "model_metadata.json"
        details = []
        ok = True
        for p in (models_dir, model_path, metadata_path):
            if p.exists():
                details.append(f"存在: {p.relative_to(self.project_root)}")
            else:
                ok = False
                details.append(f"缺失: {p.relative_to(self.project_root)}")
        self._record(CheckResult(
            name="artifact_paths",
            category="artifacts",
            passed=ok,
            message="模型文件与元数据文件齐全" if ok else "存在缺失的 artifact 文件",
            details=details,
        ))

    def _check_model_metadata_readable(self) -> None:
        metadata_path = self._models_dir() / "model_metadata.json"
        if not metadata_path.exists():
            self._record(CheckResult(
                name="metadata_readable",
                category="artifacts",
                passed=False,
                message="metadata 文件缺失，无法解析",
                details=[str(metadata_path.relative_to(self.project_root))],
            ))
            return
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            required = ["model_name", "model_type", "schema_version", "model_artifact_hash"]
            missing = [k for k in required if k not in data]
            details = [f"已解析字段: {sorted(data.keys())}"]
            if missing:
                details.append(f"缺失关键字段: {missing}")
            self._record(CheckResult(
                name="metadata_readable",
                category="artifacts",
                passed=not missing,
                message="metadata JSON 可读且字段完整" if not missing else "metadata 字段不完整",
                details=details,
            ))
        except Exception as e:
            self._record(CheckResult(
                name="metadata_readable",
                category="artifacts",
                passed=False,
                message=f"metadata 解析失败: {type(e).__name__}: {e}",
            ))

    # ------------------------------------------------------------------
    # 3. 模型运行时
    # ------------------------------------------------------------------
    def check_model_runtime(self) -> None:
        self._check_model_service_load()
        self._check_model_schema_valid()
        self._check_prediction_smoke()

    def _get_model_service(self):
        sys.path.insert(0, str(self.project_root / "car_pricing_api"))
        from services.model_service import ModelService
        ModelService._instance = None
        return ModelService.get_instance()

    def _check_model_service_load(self) -> None:
        try:
            service = self._get_model_service()
            service.load()
            model = service.model
            self._record(CheckResult(
                name="model_service_load",
                category="model_runtime",
                passed=True,
                message=f"ModelService 加载成功，模式={model.mode}",
                details=[
                    f"feature_order: {model.feature_order}",
                    f"n_features_in_: {getattr(model.model, 'n_features_in_', 'N/A')}",
                ],
            ))
        except Exception as e:
            self._record(CheckResult(
                name="model_service_load",
                category="model_runtime",
                passed=False,
                message=f"ModelService 加载失败: {type(e).__name__}: {e}",
            ))

    def _check_model_schema_valid(self) -> None:
        try:
            service = self._get_model_service()
            service.model.schema.validate()
            lineage = service.get_lineage()
            schema_version = service.model.schema.schema_version()
            details = [
                f"schema_version: {schema_version}",
                f"lineage.schema_version: {lineage.schema_version}",
            ]
            passed = lineage.schema_version == schema_version or bool(lineage.schema_version)
            self._record(CheckResult(
                name="model_schema_valid",
                category="model_runtime",
                passed=passed,
                message="Schema 校验通过，lineage 可读" if passed else "Schema 或 lineage 异常",
                details=details,
            ))
        except Exception as e:
            self._record(CheckResult(
                name="model_schema_valid",
                category="model_runtime",
                passed=False,
                message=f"Schema 校验失败: {type(e).__name__}: {e}",
            ))

    def _check_prediction_smoke(self) -> None:
        try:
            from car_pricing.feature_schema import FIELD_DEFAULT_VALUES, FEATURE_ORDER
            service = self._get_model_service()
            sample = {f: FIELD_DEFAULT_VALUES[f] for f in FEATURE_ORDER}
            result = service.model.predict_raw(sample)
            value = float(result[0])
            self._record(CheckResult(
                name="prediction_smoke",
                category="model_runtime",
                passed=True,
                message=f"冒烟预测成功，prediction={value:.2f}",
                details=[f"输入样本使用 FIELD_DEFAULT_VALUES (feature_count={len(FEATURE_ORDER)})"],
            ))
        except Exception as e:
            self._record(CheckResult(
                name="prediction_smoke",
                category="model_runtime",
                passed=False,
                message=f"冒烟预测失败: {type(e).__name__}: {e}",
            ))

    # ------------------------------------------------------------------
    # 4. API 路由契约
    # ------------------------------------------------------------------
    def check_api_contract(self) -> None:
        self._check_fastapi_routes()
        self._check_pydantic_schema_consistency()

    def _check_fastapi_routes(self) -> None:
        try:
            sys.path.insert(0, str(self.project_root / "car_pricing_api"))
            from app import app as fastapi_app
            registered = {(frozenset(route.methods), route.path) for route in fastapi_app.routes if hasattr(route, "methods")}
            expected = [
                ({"GET"}, "/schema"),
                ({"GET"}, "/status"),
                ({"GET"}, "/metadata"),
                ({"GET"}, "/health"),
                ({"POST"}, "/predict"),
                ({"GET"}, "/"),
            ]
            missing = []
            for methods, path in expected:
                if not any(methods <= set(reg_methods) and path == reg_path for reg_methods, reg_path in registered):
                    missing.append(f"{next(iter(methods))} {path}")
            details = [f"已注册路由: {sorted(f'{next(iter(m))} {p}' for m, p in registered)}"]
            if missing:
                details.append(f"缺失路由: {missing}")
            self._record(CheckResult(
                name="fastapi_routes",
                category="api_contract",
                passed=not missing,
                message="所有期望路由已注册" if not missing else f"缺失 {len(missing)} 个路由",
                details=details,
            ))
        except Exception as e:
            self._record(CheckResult(
                name="fastapi_routes",
                category="api_contract",
                passed=False,
                message=f"FastAPI 路由检查失败: {type(e).__name__}: {e}",
            ))

    @staticmethod
    def _pydantic_field_type(field_info) -> type:
        annotation = getattr(field_info, "annotation", None)
        if annotation is not None:
            return annotation
        outer = getattr(field_info, "outer_type_", None)
        return outer

    @staticmethod
    def _pydantic_fields(model_cls) -> dict:
        if hasattr(model_cls, "model_fields"):
            return model_cls.model_fields
        return model_cls.__fields__

    def _check_pydantic_schema_consistency(self) -> None:
        try:
            from car_pricing.feature_schema import FEATURE_ORDER, NUMERIC_FEATURES, CATEGORICAL_FEATURES
            sys.path.insert(0, str(self.project_root / "car_pricing_api"))
            from models import CarPrediction
            fields_map = self._pydantic_fields(CarPrediction)
            pydantic_fields = set(fields_map.keys())
            expected = set(FEATURE_ORDER)
            missing_fields = expected - pydantic_fields
            extra_fields = pydantic_fields - expected
            details = [
                f"FEATURE_ORDER ({len(expected)}): {sorted(expected)}",
                f"CarPrediction 字段 ({len(pydantic_fields)}): {sorted(pydantic_fields)}",
            ]
            type_ok = True
            for f in NUMERIC_FEATURES:
                if f in fields_map:
                    field_type = self._pydantic_field_type(fields_map[f])
                    if field_type not in (float, int):
                        type_ok = False
                        details.append(f"数值字段 {f} 类型应为 float/int，实际为 {field_type}")
            for f in CATEGORICAL_FEATURES:
                if f in fields_map:
                    field_type = self._pydantic_field_type(fields_map[f])
                    if field_type is not str:
                        type_ok = False
                        details.append(f"分类字段 {f} 类型应为 str，实际为 {field_type}")
            passed = not missing_fields and not extra_fields and type_ok
            if missing_fields:
                details.append(f"Pydantic 缺失字段: {sorted(missing_fields)}")
            if extra_fields:
                details.append(f"Pydantic 多余字段: {sorted(extra_fields)}")
            self._record(CheckResult(
                name="pydantic_schema_consistency",
                category="api_contract",
                passed=passed,
                message="Pydantic 模型与 feature_schema 一致" if passed else "Pydantic 模型与 feature_schema 不一致",
                details=details,
            ))
        except Exception as e:
            self._record(CheckResult(
                name="pydantic_schema_consistency",
                category="api_contract",
                passed=False,
                message=f"Pydantic 一致性检查失败: {type(e).__name__}: {e}",
            ))

    # ------------------------------------------------------------------
    # 5. Streamlit 前端调用边界
    # ------------------------------------------------------------------
    def check_streamlit_boundary(self) -> None:
        self._check_streamlit_only_http_client()
        self._check_streamlit_url_from_env()
        self._check_streamlit_endpoints_match_api()

    def _streamlit_path(self) -> Path:
        return self.project_root / "streamlitapp" / "streamlit_app.py"

    def _parse_streamlit_ast(self) -> Optional[ast.Module]:
        path = self._streamlit_path()
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return ast.parse(f.read())

    def _check_streamlit_only_http_client(self) -> None:
        tree = self._parse_streamlit_ast()
        if tree is None:
            self._record(CheckResult(
                name="streamlit_only_http_client",
                category="streamlit_boundary",
                passed=False,
                message="streamlit_app.py 不存在",
            ))
            return
        forbidden_prefixes = (
            "car_pricing",
            "sklearn",
            "joblib",
            "bentoml",
            "services",
            "models",
        )
        imports: List[str] = []
        bad_imports: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
                    if any(alias.name == p or alias.name.startswith(p + ".") for p in forbidden_prefixes):
                        bad_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                imports.append(f"from {mod}")
                if any(mod == p or mod.startswith(p + ".") for p in forbidden_prefixes):
                    bad_imports.append(f"from {mod}")
        details = [f"已导入模块: {sorted(set(imports))}"]
        if bad_imports:
            details.append(f"违规导入 (应仅使用 HTTP 客户端): {sorted(set(bad_imports))}")
        has_requests = any("requests" in imp or "re" == imp.split()[-1] for imp in imports)
        if not has_requests:
            details.append("警告: 未检测到 requests 导入，前端可能未通过 HTTP 调用")
        self._record(CheckResult(
            name="streamlit_only_http_client",
            category="streamlit_boundary",
            passed=not bad_imports,
            message="Streamlit 仅使用 HTTP 客户端调用 API" if not bad_imports else "Streamlit 存在直连推理代码的导入",
            details=details,
        ))

    def _check_streamlit_url_from_env(self) -> None:
        tree = self._parse_streamlit_ast()
        if tree is None:
            self._record(CheckResult(
                name="streamlit_url_from_env",
                category="streamlit_boundary",
                passed=False,
                message="streamlit_app.py 不存在",
            ))
            return
        found_env_base = False
        found_env_timeout = False
        details = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id == "API_BASE_URL":
                            src = ast.unparse(node)
                            details.append(f"API_BASE_URL 定义: {src}")
                            if "os.environ.get" in src or "os.getenv" in src:
                                found_env_base = True
                        if target.id == "REQUEST_TIMEOUT":
                            src = ast.unparse(node)
                            details.append(f"REQUEST_TIMEOUT 定义: {src}")
                            if "os.environ.get" in src or "os.getenv" in src:
                                found_env_timeout = True
        self._record(CheckResult(
            name="streamlit_url_from_env",
            category="streamlit_boundary",
            passed=found_env_base and found_env_timeout,
            message="API_BASE_URL 与 REQUEST_TIMEOUT 均从环境变量读取" if found_env_base and found_env_timeout
                    else "存在硬编码配置项，应全部从环境变量读取",
            details=details,
        ))

    def _check_streamlit_endpoints_match_api(self) -> None:
        tree = self._parse_streamlit_ast()
        if tree is None:
            self._record(CheckResult(
                name="streamlit_endpoints_match_api",
                category="streamlit_boundary",
                passed=False,
                message="streamlit_app.py 不存在",
            ))
            return
        expected_endpoints = {"/schema", "/predict"}
        found_endpoints: set[str] = set()
        details = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                s = node.value.strip()
                if s in expected_endpoints:
                    found_endpoints.add(s)
        for ep in sorted(found_endpoints):
            details.append(f"检测到端点引用: {ep}")
        missing = expected_endpoints - found_endpoints
        if missing:
            details.append(f"未引用的预期端点: {sorted(missing)}")
        passed = not missing
        self._record(CheckResult(
            name="streamlit_endpoints_match_api",
            category="streamlit_boundary",
            passed=passed,
            message="Streamlit 调用端点与 API 契约一致" if passed else "Streamlit 未覆盖所有 API 契约端点",
            details=details,
        ))

    # ------------------------------------------------------------------
    # 6. 旧 fastapi shim 边界
    # ------------------------------------------------------------------
    def check_fastapi_shim(self) -> None:
        self._check_fastapi_dir_not_pkg_in_sys_path()
        self._check_old_fastapi_shim_no_local_models()

    def _check_fastapi_dir_not_pkg_in_sys_path(self) -> None:
        fastapi_dir = self.project_root / "fastapi"
        if not fastapi_dir.exists():
            self._record(CheckResult(
                name="fastapi_dir_not_pkg",
                category="fastapi_shim",
                passed=True,
                message="旧 fastapi/ 目录不存在，无遮蔽风险",
            ))
            return
        details = []
        has_init = (fastapi_dir / "__init__.py").exists()
        has_models_py = (fastapi_dir / "models.py").exists()
        has_services_py = (fastapi_dir / "services.py").exists()
        if has_init:
            details.append("检测到 fastapi/__init__.py：存在将整个目录视为包遮蔽第三方 fastapi 的风险")
        if has_models_py:
            details.append("检测到 fastapi/models.py：可能被 app.py 内的 'from models import' 解析为本地模块")
        if has_services_py:
            details.append("检测到 fastapi/services.py：同上本地遮蔽风险")
        passed = not has_init and not has_models_py and not has_services_py
        if passed:
            details.append("目录结构合规，无 __init__.py 且无本地 models/services 模块遮蔽")
        self._record(CheckResult(
            name="fastapi_dir_not_pkg",
            category="fastapi_shim",
            passed=passed,
            message="旧 fastapi/ 目录未配置成可遮蔽第三方的 Python 包" if passed else "旧 fastapi/ 目录可能遮蔽第三方 fastapi 或本地子模块",
            details=details,
        ))

    def _check_old_fastapi_shim_no_local_models(self) -> None:
        fastapi_dir = self.project_root / "fastapi"
        app_path = fastapi_dir / "app.py"
        if not app_path.exists():
            self._record(CheckResult(
                name="old_fastapi_shim_no_local_models",
                category="fastapi_shim",
                passed=True,
                message="旧 fastapi/app.py 不存在",
            ))
            return
        with open(app_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        local_imports = []
        bad_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod in ("models", "services", "routers"):
                    local_imports.append(f"from {mod} import ...")
                    target_path = fastapi_dir / f"{mod}.py"
                    if not target_path.exists() and not (fastapi_dir / mod / "__init__.py").exists():
                        bad_imports.append(f"from {mod} import ... — fastapi/ 下无对应模块，将依赖 sys.path 注入")
        passed = not bad_imports
        details = local_imports or ["未检测到本地模块导入"]
        if bad_imports:
            details.append(f"存在无对应模块文件的导入（依赖 sys.path hack）: {bad_imports}")
        self._record(CheckResult(
            name="old_fastapi_shim_no_local_models",
            category="fastapi_shim",
            passed=passed,
            message="旧 fastapi shim 导入结构清晰" if passed else "旧 fastapi shim 使用依赖 sys.path 注入的本地导入，可能遮蔽第三方包",
            details=details,
        ))


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        prog="python -m car_pricing.tools.check_project",
        description="car_pricing 工程健康检查：包边界、artifact、API 契约、前端边界等",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="输出详细检查信息")
    parser.add_argument("--project-root", type=str, default=None, help="项目根目录，默认自动检测")
    args = parser.parse_args(argv)
    root = Path(args.project_root).resolve() if args.project_root else None
    checker = ProjectChecker(project_root=root)
    return checker.run_all(verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
