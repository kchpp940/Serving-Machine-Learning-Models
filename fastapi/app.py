import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from models import CarPrediction, INTERFACE_FIELDS

from car_pricing.model_runtime import CarPriceModel
from car_pricing.feature_schema import (
    find_model_path,
    SchemaMismatchError,
    run_startup_self_check,
    pydantic_major_version,
)

STRICT_SELF_CHECK = True


app = FastAPI(
    title="Car Price Prediction API",
    description="""An API that utilises a Machine Learning model to predict the price of a given car make and model based on various features.""",
    version="0.0.1",
    debug=True,
)

_model: CarPriceModel = None
_self_check_result: dict = None


def get_model() -> CarPriceModel:
    global _model
    if _model is None:
        local_dir = os.path.join(os.path.dirname(__file__), "models")
        model_path = find_model_path(local_dir=local_dir)
        _model = CarPriceModel.from_joblib(model_path)
        _model.validate_service(INTERFACE_FIELDS)
    return _model


def get_self_check_result() -> dict:
    global _self_check_result
    if _self_check_result is None:
        _self_check_result = run_startup_self_check(
            pydantic_model_cls=CarPrediction,
            strict=STRICT_SELF_CHECK,
        )
    return _self_check_result


@app.on_event("startup")
async def startup_event():
    print("=" * 70)
    print(" Car Price Prediction API - 启动自检")
    print("=" * 70)
    print(f"  Pydantic 主版本:     {pydantic_major_version()}")

    self_check = get_self_check_result()
    print(f"  Pydantic 当前版本:   {self_check['pydantic_version']}")
    print(f"  FastAPI 当前版本:    {self_check['fastapi_version']}")
    print(f"  Feature 字段数:      {len(self_check['feature_order'])}")
    print()

    if self_check["dependency_issues"]:
        print("  [依赖版本警告]")
        for issue in self_check["dependency_issues"]:
            print(f"    - {issue}")
    else:
        print("  依赖版本检查: PASSED")

    if self_check["schema_issues"]:
        print("  [Schema 生成自检失败]")
        for issue in self_check["schema_issues"]:
            print(f"    - {issue}")
    else:
        print("  Schema 生成自检: PASSED")

    if not self_check["passed"] and self_check["strict"]:
        print()
        print("FATAL - 自检未通过，服务拒绝启动。")
        print("请检查 pydantic / fastapi 版本是否在支持矩阵内，")
        print("或确认 FeatureSchema 与 Pydantic 模型生成逻辑一致。")
        print("=" * 70)
        raise RuntimeError("Startup self-check failed: " + "; ".join(
            self_check["dependency_issues"] + self_check["schema_issues"]
        ))

    try:
        model = get_model()
        info = model.model_info()
        print()
        print("  模型加载: PASSED")
        print(f"    mode:             {info['mode']}")
        print(f"    n_features_in_:   {info['n_features_in_']}")
        print(f"    schema_features:  {info['schema_n_features']}")
        print(f"    feature_order:    {info['feature_order']}")
        print(f"    interface_fields: {INTERFACE_FIELDS}")
        print("  Schema 校验: PASSED")
        print()
        print(" 全部启动检查通过 ✅")
        print("=" * 70)
    except SchemaMismatchError as e:
        print(f"FATAL - Schema mismatch on startup: {e}")
        print("=" * 70)
        raise
    except Exception as e:
        print(f"FATAL - Failed to load model on startup: {e}")
        print("=" * 70)
        raise


@app.get("/", response_class=PlainTextResponse)
async def running():
    note = """
Car Price Prediction API 🙌🏻
Note: add "/docs" to the URL to get the Swagger UI Docs or "/redoc"
  """
    return note


favicon_path = "favicon.png"


@app.get("/favicon.png", include_in_schema=False)
async def favicon():
    from fastapi.responses import FileResponse
    return FileResponse(favicon_path)


@app.get("/self_check")
async def self_check():
    return get_self_check_result()


@app.get("/schema")
async def get_schema():
    model = get_model()
    return model.schema_export()


@app.get("/model_info")
async def get_model_info():
    model = get_model()
    return model.model_info()


@app.post("/predict")
def predict(data: CarPrediction):
    try:
        model = get_model()
        predictions = model.predict_from_pydantic(data)
        value = float(predictions[0])
        return {"predicted_price": value}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except SchemaMismatchError as e:
        raise HTTPException(status_code=500, detail=f"Schema 不一致: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")
