from __future__ import annotations

import ast
import importlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class CheckStatus:
    PASS = "pass"
    FAIL = "fail"


class CheckCategory:
    PACKAGE = "package"
    ARTIFACTS = "artifacts"
    MODEL_RUNTIME = "model_runtime"
    API_CONTRACT = "api_contract"
    STREAMLIT_BOUNDARY = "streamlit_boundary"
    FASTAPI_SHIM = "fastapi_shim"


@dataclass
class CheckResult:
    name: str
    category: str
    status: str
    message: str
    details: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.status not in (CheckStatus.PASS, CheckStatus.FAIL):
            raise ValueError(f"status 必须为 '{CheckStatus.PASS}' 或 '{CheckStatus.FAIL}'，收到 '{self.status}'")

    @property
    def passed(self) -> bool:
        return self.status == CheckStatus.PASS

    @classmethod
    def ok(cls, name: str, category: str, message: str, details: Optional[List[str]] = None) -> "CheckResult":
        return cls(name=name, category=category, status=CheckStatus.PASS, message=message, details=details or [])

    @classmethod
    def fail(cls, name: str, category: str, message: str, details: Optional[List[str]] = None) -> "CheckResult":
        return cls(name=name, category=category, status=CheckStatus.FAIL, message=message, details=details or [])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "status": self.status,
            "message": self.message,
            "details": list(self.details),
        }

    def format_text(self, verbose: bool = False) -> str:
        tag = "PASS" if self.passed else "FAIL"
        lines = [f"[{tag}] {self.category} :: {self.name} — {self.message}"]
        if verbose and self.details:
            for d in self.details:
                lines.append(f"       -> {d}")
        return "\n".join(lines)


@dataclass
class CheckReport:
    results: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results) if self.results else False

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        return self.total - self.passed_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": "car_pricing.tools.check_project",
            "passed": self.passed,
            "summary": {
                "total": self.total,
                "passed": self.passed_count,
                "failed": self.failed_count,
            },
            "results": [r.to_dict() for r in self.results],
        }

    def format_text(self, verbose: bool = False) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append(f"car_pricing 工程健康检查  —  {self.passed_count}/{self.total} 通过, {self.failed_count} 失败")
        lines.append("=" * 70)
        for r in self.results:
            lines.append(r.format_text(verbose=verbose))
        lines.append("=" * 70)
        if self.failed_count:
            lines.append(f"共 {self.failed_count} 项未通过，请根据上面的 FAIL 记录修复。")
        else:
            lines.append("所有检查项通过，工程配置完整。")
        return "\n".join(lines)

    def format_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class ProjectChecker:
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or PROJECT_ROOT
        self._report = CheckReport()

    def _record(self, result: CheckResult) -> CheckResult:
        self._report.results.append(result)
        return result

    def run(self) -> CheckReport:
        self._report = CheckReport()
        checkers = [
            (CheckCategory.PACKAGE, self.check_package_imports),
            (CheckCategory.ARTIFACTS, self.check_artifacts),
            (CheckCategory.MODEL_RUNTIME, self.check_model_runtime),
            (CheckCategory.API_CONTRACT, self.check_api_contract),
            (CheckCategory.STREAMLIT_BOUNDARY, self.check_streamlit_boundary),
            (CheckCategory.FASTAPI_SHIM, self.check_fastapi_shim),
        ]
        for category, fn in checkers:
            try:
                fn()
            except Exception as e:
                self._record(CheckResult.fail(
                    name=fn.__name__,
                    category=category,
                    message=f"检查器异常: {type(e).__name__}: {e}",
                ))
        return self._report

    def run_all(self, verbose: bool = False, output_json: bool = False) -> int:
        report = self.run()
        if output_json:
            print(report.format_json())
        else:
            print(report.format_text(verbose=verbose))
        return 0 if report.passed else 1

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
            category=CheckCategory.PACKAGE,
            status=CheckStatus.PASS if ok else CheckStatus.FAIL,
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
        ok = not missing and not extra
        self._record(CheckResult(
            name="all_exports_consistent",
            category=CheckCategory.PACKAGE,
            status=CheckStatus.PASS if ok else CheckStatus.FAIL,
            message="__all__ 与实际导出一致" if ok else "__all__ 与实际导出不一致",
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
            category=CheckCategory.ARTIFACTS,
            status=CheckStatus.PASS if ok else CheckStatus.FAIL,
            message="模型文件与元数据文件齐全" if ok else "存在缺失的 artifact 文件",
            details=details,
        ))

    def _check_model_metadata_readable(self) -> None:
        metadata_path = self._models_dir() / "model_metadata.json"
        if not metadata_path.exists():
            self._record(CheckResult.fail(
                name="metadata_readable",
                category=CheckCategory.ARTIFACTS,
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
                category=CheckCategory.ARTIFACTS,
                status=CheckStatus.PASS if not missing else CheckStatus.FAIL,
                message="metadata JSON 可读且字段完整" if not missing else "metadata 字段不完整",
                details=details,
            ))
        except Exception as e:
            self._record(CheckResult.fail(
                name="metadata_readable",
                category=CheckCategory.ARTIFACTS,
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
            self._record(CheckResult.ok(
                name="model_service_load",
                category=CheckCategory.MODEL_RUNTIME,
                message=f"ModelService 加载成功，模式={model.mode}",
                details=[
                    f"feature_order: {model.feature_order}",
                    f"n_features_in_: {getattr(model.model, 'n_features_in_', 'N/A')}",
                ],
            ))
        except Exception as e:
            self._record(CheckResult.fail(
                name="model_service_load",
                category=CheckCategory.MODEL_RUNTIME,
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
                category=CheckCategory.MODEL_RUNTIME,
                status=CheckStatus.PASS if passed else CheckStatus.FAIL,
                message="Schema 校验通过，lineage 可读" if passed else "Schema 或 lineage 异常",
                details=details,
            ))
        except Exception as e:
            self._record(CheckResult.fail(
                name="model_schema_valid",
                category=CheckCategory.MODEL_RUNTIME,
                message=f"Schema 校验失败: {type(e).__name__}: {e}",
            ))

    def _check_prediction_smoke(self) -> None:
        try:
            from car_pricing.feature_schema import FIELD_DEFAULT_VALUES, FEATURE_ORDER
            service = self._get_model_service()
            sample = {f: FIELD_DEFAULT_VALUES[f] for f in FEATURE_ORDER}
            result = service.model.predict_raw(sample)
            value = float(result[0])
            self._record(CheckResult.ok(
                name="prediction_smoke",
                category=CheckCategory.MODEL_RUNTIME,
                message=f"冒烟预测成功，prediction={value:.2f}",
                details=[f"输入样本使用 FIELD_DEFAULT_VALUES (feature_count={len(FEATURE_ORDER)})"],
            ))
        except Exception as e:
            self._record(CheckResult.fail(
                name="prediction_smoke",
                category=CheckCategory.MODEL_RUNTIME,
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
                category=CheckCategory.API_CONTRACT,
                status=CheckStatus.PASS if not missing else CheckStatus.FAIL,
                message="所有期望路由已注册" if not missing else f"缺失 {len(missing)} 个路由",
                details=details,
            ))
        except Exception as e:
            self._record(CheckResult.fail(
                name="fastapi_routes",
                category=CheckCategory.API_CONTRACT,
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
                category=CheckCategory.API_CONTRACT,
                status=CheckStatus.PASS if passed else CheckStatus.FAIL,
                message="Pydantic 模型与 feature_schema 一致" if passed else "Pydantic 模型与 feature_schema 不一致",
                details=details,
            ))
        except Exception as e:
            self._record(CheckResult.fail(
                name="pydantic_schema_consistency",
                category=CheckCategory.API_CONTRACT,
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
            self._record(CheckResult.fail(
                name="streamlit_only_http_client",
                category=CheckCategory.STREAMLIT_BOUNDARY,
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
            category=CheckCategory.STREAMLIT_BOUNDARY,
            status=CheckStatus.PASS if not bad_imports else CheckStatus.FAIL,
            message="Streamlit 仅使用 HTTP 客户端调用 API" if not bad_imports else "Streamlit 存在直连推理代码的导入",
            details=details,
        ))

    def _check_streamlit_url_from_env(self) -> None:
        tree = self._parse_streamlit_ast()
        if tree is None:
            self._record(CheckResult.fail(
                name="streamlit_url_from_env",
                category=CheckCategory.STREAMLIT_BOUNDARY,
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
        ok = found_env_base and found_env_timeout
        self._record(CheckResult(
            name="streamlit_url_from_env",
            category=CheckCategory.STREAMLIT_BOUNDARY,
            status=CheckStatus.PASS if ok else CheckStatus.FAIL,
            message="API_BASE_URL 与 REQUEST_TIMEOUT 均从环境变量读取" if ok
                    else "存在硬编码配置项，应全部从环境变量读取",
            details=details,
        ))

    def _check_streamlit_endpoints_match_api(self) -> None:
        tree = self._parse_streamlit_ast()
        if tree is None:
            self._record(CheckResult.fail(
                name="streamlit_endpoints_match_api",
                category=CheckCategory.STREAMLIT_BOUNDARY,
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
        ok = not missing
        self._record(CheckResult(
            name="streamlit_endpoints_match_api",
            category=CheckCategory.STREAMLIT_BOUNDARY,
            status=CheckStatus.PASS if ok else CheckStatus.FAIL,
            message="Streamlit 调用端点与 API 契约一致" if ok else "Streamlit 未覆盖所有 API 契约端点",
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
            self._record(CheckResult.ok(
                name="fastapi_dir_not_pkg",
                category=CheckCategory.FASTAPI_SHIM,
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
        ok = not has_init and not has_models_py and not has_services_py
        if ok:
            details.append("目录结构合规，无 __init__.py 且无本地 models/services 模块遮蔽")
        self._record(CheckResult(
            name="fastapi_dir_not_pkg",
            category=CheckCategory.FASTAPI_SHIM,
            status=CheckStatus.PASS if ok else CheckStatus.FAIL,
            message="旧 fastapi/ 目录未配置成可遮蔽第三方的 Python 包" if ok else "旧 fastapi/ 目录可能遮蔽第三方 fastapi 或本地子模块",
            details=details,
        ))

    def _check_old_fastapi_shim_no_local_models(self) -> None:
        fastapi_dir = self.project_root / "fastapi"
        app_path = fastapi_dir / "app.py"
        if not app_path.exists():
            self._record(CheckResult.ok(
                name="old_fastapi_shim_no_local_models",
                category=CheckCategory.FASTAPI_SHIM,
                message="旧 fastapi/app.py 不存在",
            ))
            return
        with open(app_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        bare_imports = []
        sys_path_hacks = []
        details = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod in ("models", "services", "routers"):
                    names = ", ".join(a.name for a in node.names)
                    bare_imports.append(f"from {mod} import {names}")
            if isinstance(node, ast.Call):
                func = node.func
                if (isinstance(func, ast.Attribute) and func.attr == "insert"
                        and isinstance(func.value, ast.Attribute) and func.value.attr == "path"
                        and isinstance(func.value.value, ast.Name) and func.value.value.id == "sys"):
                    sys_path_hacks.append("sys.path.insert(...)")
        if bare_imports:
            details.append(f"裸模块导入: {bare_imports}")
        if sys_path_hacks:
            details.append(f"sys.path hack: {sys_path_hacks}")
        ok = not bare_imports and not sys_path_hacks
        if ok:
            details.append("所有导入均使用显式包路径，无 sys.path hack")
        self._record(CheckResult(
            name="old_fastapi_shim_no_local_models",
            category=CheckCategory.FASTAPI_SHIM,
            status=CheckStatus.PASS if ok else CheckStatus.FAIL,
            message="旧 fastapi shim 导入结构清晰" if ok else "旧 fastapi shim 存在裸模块导入或 sys.path hack，应改用显式包路径",
            details=details,
        ))


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        prog="python -m car_pricing.tools.check_project",
        description="car_pricing 工程健康检查：包边界、artifact、API 契约、前端边界等",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="输出详细检查信息")
    parser.add_argument("--json", action="store_true", dest="output_json", help="以 JSON 格式输出")
    parser.add_argument("--project-root", type=str, default=None, help="项目根目录，默认自动检测")
    args = parser.parse_args(argv)
    root = Path(args.project_root).resolve() if args.project_root else None
    checker = ProjectChecker(project_root=root)
    return checker.run_all(verbose=args.verbose, output_json=args.output_json)


if __name__ == "__main__":
    raise SystemExit(main())
