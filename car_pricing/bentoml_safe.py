from __future__ import annotations

import sys
import os
import importlib


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_LOCAL_BENTOML_DIR = os.path.join(_PROJECT_ROOT, "bentoml")


def _import_bentoml_package():
    _original_path = list(sys.path)
    _original_modules = set(sys.modules.keys())

    _cwd = os.getcwd()
    _bentoml_parent = os.path.dirname(_LOCAL_BENTOML_DIR)

    def _is_system_lib(p):
        if not p:
            return False
        if "site-packages" in p or "dist-packages" in p:
            return True
        if "/python3" in p or "/lib/python" in p:
            return True
        if p == os.path.dirname(os.__file__):
            return True
        return False

    for p in list(sys.path):
        if not _is_system_lib(p):
            if p == _LOCAL_BENTOML_DIR or p == _bentoml_parent or p == _cwd or p == "" or p == ".":
                sys.path.remove(p)

    if "bentoml" in sys.modules:
        del sys.modules["bentoml"]

    try:
        import bentoml as _bm

        _loaded_origin = os.path.abspath(_bm.__file__) if _bm.__file__ else ""
        if _LOCAL_BENTOML_DIR in _loaded_origin:
            raise ImportError(
                f"Local bentoml/ directory conflicts with package import"
            )

        return _bm, _original_path
    except Exception:
        sys.path = _original_path
        for mod_name in list(sys.modules.keys()):
            if mod_name not in _original_modules and mod_name.startswith("bentoml"):
                del sys.modules[mod_name]
        raise


bentoml, _saved_path = _import_bentoml_package()


import bentoml.picklable_model
from bentoml.io import NumpyNdarray, PandasDataFrame, JSON


sys.path = _saved_path
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _get_bentoml_major_minor():
    try:
        parts = bentoml.__version__.split(".")
        return (int(parts[0]), int(parts[1]))
    except Exception:
        return (0, 0)


_bentoml_version = _get_bentoml_major_minor()


_picklable_model = bentoml.picklable_model


class _CompatIORedirector:
    def __init__(self, real_model):
        self._real = real_model

    def get(self, tag):
        return self._real.get(tag)

    def load_model(self, tag):
        return self._real.load_model(tag)

    def save_model(self, *args, **kwargs):
        return self._real.save_model(*args, **kwargs)


picklable_model = _CompatIORedirector(_picklable_model)


class _ServiceCompat:
    def __init__(self, name, runners=None, **kwargs):
        self._name = name
        self._runners = runners or []
        self._apis = {}
        self._inner_service = None

        if _bentoml_version >= (1, 4):
            self._build_new_style_service()
        else:
            self._service = bentoml.Service(name, runners=runners, **kwargs)

    def _build_new_style_service(self):
        class _ServiceClass:
            def __init__(self_inner):
                self_inner._predictors = []
                for runner in self._runners:
                    pass

        service_decorator = bentoml.service(name=self._name)
        self._inner_class = service_decorator(_ServiceClass)
        self._service = self._inner_class

    @property
    def name(self):
        if _bentoml_version >= (1, 4):
            return self._service.name
        return self._service.name

    @property
    def apis(self):
        if _bentoml_version >= (1, 4):
            return list(self._service.apis.values())
        return list(self._service.apis.values())

    def api(self, input, output):
        def decorator(func):
            if _bentoml_version >= (1, 4):
                api_decorator = bentoml.api(
                    input_spec=type(input),
                    output_spec=type(output),
                )
                wrapped = api_decorator(func)
                setattr(self._inner_class, func.__name__, wrapped)
            else:
                self._service.api(input=input, output=output)(func)
            self._apis[func.__name__] = func
            return func
        return decorator


def Service(name, runners=None, **kwargs):
    return _ServiceCompat(name, runners=runners, **kwargs)


def restore_project_path():
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)


restore_project_path()


__all__ = [
    "bentoml",
    "picklable_model",
    "NumpyNdarray",
    "PandasDataFrame",
    "JSON",
    "Service",
    "restore_project_path",
]
