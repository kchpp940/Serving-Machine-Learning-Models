import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import bentoml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from car_pricing.feature_schema import (
    load_training_data,
    prepare_training_data,
    bundle_model,
    save_bundle,
    SHARED_MODEL_PATH,
)


def main():
    csv_path = os.path.join(os.path.dirname(__file__), "Data", "cars.csv")
    df = load_training_data(csv_path)

    X, y, schema = prepare_training_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    schema.validate_model_input(model)

    bundle = bundle_model(model, schema)

    bentoml.sklearn.save("gbr", bundle)
    print(f"[bentoml] Model saved to BentoML store")

    save_bundle(bundle, SHARED_MODEL_PATH)
    print(f"[shared]  Model saved to {SHARED_MODEL_PATH}")

    print(f"Feature order: {schema.feature_order}")
    print(f"Number of features: {schema.n_features()}")
    print(f"Model n_features_in_: {model.n_features_in_}")


if __name__ == "__main__":
    main()
