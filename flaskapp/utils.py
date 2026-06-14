import numpy as np

from model_bundle_loader import get_bundle


def predict_price(features: dict) -> np.ndarray:
    bundle = get_bundle()
    return bundle.predict(features)
