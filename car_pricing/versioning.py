from __future__ import annotations

import hashlib
import os
from typing import Optional


def compute_file_hash(path: str, algorithm: str = "sha256", chunk_size: int = 65536) -> str:
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_data_version(csv_path: str) -> str:
    return compute_file_hash(csv_path)[:16]


def compute_model_hash(model_dir: str, model_filename: str = "sklearn_gbr.pkl") -> Optional[str]:
    model_path = os.path.join(model_dir, model_filename)
    if not os.path.exists(model_path):
        return None
    return compute_file_hash(model_path)
