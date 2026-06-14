import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

FEATURE_COLUMNS = [
    "enginesize",
    "curbweight",
    "horsepower",
    "highwaympg",
    "carwidth",
    "wheelbase",
    "drivewheel",
    "citympg",
    "boreratio",
    "cylindernumber",
]

TARGET_COLUMN = "price"


def _is_string_dtype(dtype) -> bool:
    return pd.api.types.is_string_dtype(dtype) or dtype == "O"


def fit_label_encoders(data: pd.DataFrame) -> dict:
    cat_cols = [col for col in data.columns if _is_string_dtype(data[col].dtype)]
    encoders = {}
    for col in cat_cols:
        lb = LabelEncoder()
        lb.fit(data[col].astype(str))
        encoders[col] = lb
    return encoders


def apply_label_encoders(data: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    out = data.copy()
    for col, lb in encoders.items():
        out[col] = lb.transform(out[col].astype(str))
    return out


def build_model_bundle(data_path: str):
    dataset = pd.read_csv(
        data_path,
        usecols=FEATURE_COLUMNS + [TARGET_COLUMN],
    )

    feature_df = dataset[FEATURE_COLUMNS]
    target = dataset[TARGET_COLUMN]

    encoders = fit_label_encoders(feature_df)
    encoded_features = apply_label_encoders(feature_df, encoders)

    X_train, X_test, y_train, y_test = train_test_split(
        encoded_features, target, random_state=0
    )

    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    bundle = {
        "model": model,
        "feature_order": FEATURE_COLUMNS,
        "categorical_encoders": encoders,
        "target_column": TARGET_COLUMN,
    }
    return bundle


if __name__ == "__main__":
    bundle = build_model_bundle("../Data/cars.csv")

    model = bundle["model"]
    n_expected = len(bundle["feature_order"])
    assert getattr(model, "n_features_in_", n_expected) == n_expected, (
        f"模型 n_features_in_={model.n_features_in_} 与 feature_order 长度 {n_expected} 不一致"
    )

    joblib.dump(bundle, "./models/sklearn_gbr.pkl")
    print(
        f"已保存模型包: feature_order={bundle['feature_order']}, "
        f"分类编码器列= {list(bundle['categorical_encoders'].keys())}"
    )
