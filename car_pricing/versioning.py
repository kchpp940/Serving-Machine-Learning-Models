from __future__ import annotations

import hashlib
import json
import os
from typing import Optional, Union
import pandas as pd


def _stable_hash(obj: Union[dict, list, str, bytes]) -> str:
    if isinstance(obj, bytes):
        raw = obj
    elif isinstance(obj, str):
        raw = obj.encode("utf-8")
    else:
        raw = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def compute_schema_version(schema_dict: dict) -> str:
    signature = {
        "feature_order": schema_dict.get("feature_order", []),
        "numeric_features": schema_dict.get("numeric_features", []),
        "categorical_features": schema_dict.get("categorical_features", []),
        "target_column": schema_dict.get("target_column", ""),
        "categorical_options": {},
    }
    categorical_options = schema_dict.get("categorical_options", {})
    categorical_encoders = schema_dict.get("categorical_encoders", {})

    for f in signature["categorical_features"]:
        if f in categorical_options:
            opts = categorical_options[f]
            signature["categorical_options"][f] = [
                opt.get("form_value", opt) if isinstance(opt, dict) else opt
                for opt in opts
            ]
        elif f in categorical_encoders:
            signature["categorical_options"][f] = list(
                categorical_encoders[f].get("classes", [])
            )
        else:
            signature["categorical_options"][f] = []

    return _stable_hash(signature)


def compute_data_version_from_df(df: pd.DataFrame) -> str:
    summary = {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "row_hash": hashlib.sha256(
            pd.util.hash_pandas_object(df).values.tobytes()
        ).hexdigest()[:16],
    }
    return _stable_hash(summary)


def compute_data_version_from_file(csv_path: str) -> str:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"数据文件不存在: {csv_path}")
    stat = os.stat(csv_path)
    file_signature = {
        "path": os.path.abspath(csv_path),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
    }
    with open(csv_path, "rb") as f:
        file_hash = hashlib.sha256()
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            file_hash.update(chunk)
    file_signature["content_hash"] = file_hash.hexdigest()[:16]
    return _stable_hash(file_signature)


def compute_data_version(source: Union[str, pd.DataFrame]) -> str:
    if isinstance(source, pd.DataFrame):
        return compute_data_version_from_df(source)
    elif isinstance(source, str):
        return compute_data_version_from_file(source)
    else:
        raise TypeError(f"不支持的数据源类型: {type(source)}")


def attach_versions_to_schema(
    schema_dict: dict,
    data_source: Optional[Union[str, pd.DataFrame]] = None,
) -> dict:
    result = dict(schema_dict)
    result["schema_version"] = compute_schema_version(schema_dict)
    if data_source is not None:
        result["data_version"] = compute_data_version(data_source)
    return result
