import joblib
import numpy as np

from schema import MODEL_FEATURE_ORDER


def predict_price(features: dict) -> np.ndarray:
    feature_values = [features[name] for name in MODEL_FEATURE_ORDER]
    data = np.array([feature_values])
    model = joblib.load("./models/sklearn_gbr.pkl")
    predictions = model.predict(data)
    return predictions
