from __future__ import annotations

import sys
import os
import importlib.util
from typing import List, Optional


def clean_sys_path_for_import(package_name: str, local_dir_conflict: Optional[str] = None) -> None:
    _site_packages_dirs = []
    for p in sys.path:
        if p and ("site-packages" in p or "dist-packages" in p or "/lib/" in p):
            _site_packages_dirs.append(p)

    _cwd = os.getcwd()
    _script_dir = os.path.dirname(sys.argv[0]) if sys.argv else ""
    _script_dir_abs = os.path.abspath(_script_dir) if _script_dir else ""

    _paths_to_remove = [
        "",
        ".",
        _cwd,
    ]

    if _script_dir_abs and _script_dir_abs != _cwd:
        _paths_to_remove.append(_script_dir_abs)

    if local_dir_conflict:
        _local_abs = os.path.abspath(local_dir_conflict)
        _parent_dir = os.path.dirname(_local_abs)
        if _parent_dir and _parent_dir not in _paths_to_remove:
            _paths_to_remove.append(_parent_dir)

    for p in list(_paths_to_remove):
        while p in sys.path:
            sys.path.remove(p)

    for p in _site_packages_dirs:
        if p not in sys.path:
            sys.path.append(p)


def verify_no_local_conflict(package_name: str, local_dir_conflict: str) -> None:
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        raise ImportError(f"{package_name} package is not installed.")

    spec_origin = os.path.abspath(spec.origin) if spec.origin else ""
    local_abs = os.path.abspath(local_dir_conflict)

    if local_abs in spec_origin:
        raise ImportError(
            f"Local directory ({local_abs}) conflicts with {package_name} package. "
            f"The import resolution is picking up the local directory instead of "
            f"the installed package. Please run from a different location or "
            f"rename the local directory."
        )


def import_bentoml_safe(project_root: str):
    local_bentoml_dir = os.path.join(project_root, "bentoml")

    clean_sys_path_for_import("bentoml", local_bentoml_dir)

    import bentoml
    import bentoml.picklable_model
    from bentoml.io import NumpyNdarray, PandasDataFrame, JSON

    verify_no_local_conflict("bentoml", local_bentoml_dir)

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    return bentoml, bentoml.picklable_model, NumpyNdarray, PandasDataFrame, JSON


def import_mlflow_safe():
    clean_sys_path_for_import("mlflow")

    import mlflow
    import mlflow.sklearn

    return mlflow, mlflow.sklearn
