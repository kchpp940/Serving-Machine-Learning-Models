import bentoml
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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


if __name__ == "__main__":
    dataset = pd.read_csv(
        "./Data/cars.csv",
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

    n_expected = len(FEATURE_COLUMNS)
    assert getattr(model, "n_features_in_", n_expected) == n_expected

    bundle = {
        "model": model,
        "feature_order": FEATURE_COLUMNS,
        "categorical_encoders": encoders,
        "target_column": TARGET_COLUMN,
    }
    bentoml.sklearn.save("gbr_bundle", bundle)
    print(
        f"已保存 bentoml 模型包: feature_order={FEATURE_COLUMNS}, "
        f"分类编码器列={list(encoders.keys())}"
    )
