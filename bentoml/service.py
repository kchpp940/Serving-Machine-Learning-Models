import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
from typing import Any

import bentoml
from bentoml.io import NumpyNdarray, PandasDataFrame, JSON
from pydantic import BaseModel

from model_runtime import CarPriceModel, _is_bundle


BENTOML_MODEL_TAG = "gbr_bundle:latest"
BENTOML_LEGACY_TAG = "gbr:latest"

# ============================================================
# 1. 旧版 gbr tag 只存了 sklearn 对象 → 用 load_runner 跑；
# 2. 新版 gbr_bundle tag 存的是 dict bundle → 取 raw 交给 CarPriceModel。
#
# 为了同时兼容两种 BentoML 存储格式，我们不在 runner 层 decode，
# 而是在 service 层处理：
#   - PandasDataFrame / JSON 输入 → 交给 CarPriceModel 编码 + 预测
#   - NumpyNdarray 输入（已经编码好的 10 列） → 直接喂给模型
# ============================================================

def _load_runtime_from_bentoml() -> CarPriceModel:
    """按优先级尝试从 bentoml 模型库加载，最后回退到保存裸模型。"""
    for tag in (BENTOML_MODEL_TAG, BENTOML_LEGACY_TAG):
        try:
            store = bentoml.models.get(tag)
            raw = store.to_runner().model
            if _is_bundle(raw) or hasattr(raw, "predict"):
                return CarPriceModel.from_sklearn_object(raw)
        except Exception:
            continue
    raise RuntimeError(
        f"BentoML 模型库中既没有 {BENTOML_MODEL_TAG} 也没有 {BENTOML_LEGACY_TAG}"
    )


# ===== Runner 定义（兼容 bundle / legacy） =====
try:
    _raw_obj = None
    for tag in (BENTOML_MODEL_TAG, BENTOML_LEGACY_TAG):
        try:
            store = bentoml.models.get(tag)
            _raw_obj = store.to_runner().model
            break
        except Exception:
            continue

    if _raw_obj is None:
        raise RuntimeError("BentoML 无可用模型")

    if _is_bundle(_raw_obj):
        # 新版：BentoML 里存了 bundle dict，但 runner 只能跑带 predict 的对象
        # 所以把真正的 sklearn model 拿出来当 runner 用
        _sklearn_model = _raw_obj["model"]
    else:
        _sklearn_model = _raw_obj

    predictor = bentoml.sklearn.save_model(
        "gbr_runner_model",
        _sklearn_model,
    ).to_runner()
    predictor_runner = predictor
except Exception as exc:  # pragma: no cover - 开发环境不一定有 bentoml 模型
    predictor_runner = None
    print(f"[BentoML] runner 未加载（{exc}），service 层将使用 CarPriceModel 独立推理")

# ===== 推理服务 =====
service = bentoml.Service("gbr", runners=[predictor_runner] if predictor_runner else [])

_cached_runtime: CarPriceModel | None = None


def _get_runtime() -> CarPriceModel:
    global _cached_runtime
    if _cached_runtime is None:
        _cached_runtime = _load_runtime_from_bentoml()
    return _cached_runtime


class CarPredictionRequest(BaseModel):
    enginesize: float
    curbweight: float
    horsepower: float
    highwaympg: float
    carwidth: float
    wheelbase: float
    drivewheel: object
    citympg: float
    boreratio: float
    cylindernumber: object


# ---- API 1: 推荐，JSON 请求体（分类字段允许可读字符串） ----
@service.api(input=JSON(pydantic_model=CarPredictionRequest), output=NumpyNdarray())
def predict_json(request: CarPredictionRequest) -> np.ndarray:
    runtime = _get_runtime()
    values = request.dict()
    encoded = runtime.encode_dict(values)
    return runtime.predict_encoded(encoded)


# ---- API 2: DataFrame 输入（批量） ----
@service.api(input=PandasDataFrame(), output=NumpyNdarray())
def predict(df: pd.DataFrame) -> np.ndarray:
    runtime = _get_runtime()
    return runtime.predict_dataframe(df)


# ---- API 3: Numpy 数组输入（客户端已自行编码的 10 列） ----
@service.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict_numpy(data: np.ndarray) -> np.ndarray:
    if predictor_runner is not None:
        return predictor_runner.predict.run(data)
    runtime = _get_runtime()
    return runtime.model.predict(data)
