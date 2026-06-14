"""兼容转发层：旧代码 `from model_runtime import ...` 仍然可用。

新代码请使用 `from car_pricing import CarPriceModel, load_model, ...`。
"""

from car_pricing.model_runtime import *  # noqa: F401,F403
from car_pricing.model_runtime import (  # noqa: F401
    CarPriceModel,
    _is_bundle,
    _DEFAULT_FEATURE_ORDER_LEGACY,
)
