import os

import bentoml
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from car_pricing.feature_schema import (
    load_training_data,
    prepare_training_data,
    bundle_model,
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

    bentoml.sklearn.save_model("gbr", bundle)

    print(f"Model saved to BentoML")
    print(f"Feature order: {schema.feature_order}")
    print(f"Number of features: {schema.n_features()}")
    print(f"Model n_features_in_: {model.n_features_in_}")


if __name__ == "__main__":
    main()
