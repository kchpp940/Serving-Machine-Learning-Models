from __future__ import annotations

import hashlib
import os
from typing import Optional, Any

try:
    import pandas as pd
except ImportError:
    pd = None


def compute_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dataframe_hash(df) -> str:
    if pd is None:
        raise RuntimeError("需要 pandas 才能使用 compute_dataframe_hash")
    csv_content = df.to_csv(index=False)
    return hashlib.sha256(csv_content.encode("utf-8")).hexdigest()[:12]


def compute_data_version(csv_path: str) -> str:
    return compute_file_hash(csv_path)[:12]


def verify_artifact_hash(file_path: str, expected_hash: str) -> bool:
    if not os.path.exists(file_path):
        return False
    actual_hash = compute_file_hash(file_path)
    return actual_hash == expected_hash


def compute_schema_hash(schema_dict: dict) -> str:
    import json
    raw = json.dumps(schema_dict, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def compute_object_hash(obj: Any) -> str:
    import pickle
    pickled = pickle.dumps(obj)
    return hashlib.sha256(pickled).hexdigest()
