import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import joblib

from car_pricing.feature_schema import (
    load_training_data,
    prepare_training_data,
    bundle_model,
)
from car_pricing.artifact_paths import ArtifactPaths
from car_pricing.config import RuntimeConfig


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config = RuntimeConfig.default(base_dir=script_dir)
    paths = ArtifactPaths.from_config(config=config, base_dir=script_dir)

    paths.assert_training_data_file_exists()
    paths.ensure_model_dir()

    df = load_training_data(paths.training_data_file)

    X, y, schema = prepare_training_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    schema.validate_model_input(model)

    bundle = bundle_model(model, schema)

    joblib.dump(bundle, paths.model_file)

    print(f"Model saved to {paths.model_file}")
    print(f"Feature order: {schema.feature_order}")
    print(f"Number of features: {schema.n_features()}")
    print(f"Model n_features_in_: {model.n_features_in_}")


if __name__ == "__main__":
    main()
