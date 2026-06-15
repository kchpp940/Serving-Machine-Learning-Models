from __future__ import annotations

import hashlib
import json
import os
import pickle
from typing import Optional, Any

from car_pricing.feature_schema import FeatureSchema


def compute_schema_version(schema: FeatureSchema) -> str:
    schema_dict = schema.to_dict()
    for col in schema_dict.get("categorical_encoders", {}):
        schema_dict["categorical_encoders"][col]["classes"] = sorted(
            schema_dict["categorical_encoders"][col]["classes"]
        )
    payload = json.dumps(schema_dict, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def compute_data_version(csv_path: Optional[str] = None) -> str:
    if csv_path is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        candidates = [
            os.path.join(base_dir, "Data", "cars.csv"),
            os.path.join(base_dir, "bentoml", "Data", "cars.csv"),
            os.path.join(base_dir, "mlflow", "data", "cars.csv"),
        ]
        for path in candidates:
            if os.path.exists(path):
                csv_path = path
                break
    if csv_path is None or not os.path.exists(csv_path):
        return "unknown"
    h = hashlib.sha1()
    with open(csv_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def compute_artifact_hash(obj: Any) -> str:
    payload = pickle.dumps(obj)
    return hashlib.sha256(payload).hexdigest()
